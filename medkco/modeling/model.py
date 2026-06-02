import os
import numpy as np
from tqdm import tqdm

from .dictionary import definitions, definitions_CXR, definitions_OCT

import torch
import torchvision
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer, logging
from ..pretraining.data.SPCL_dataloader import *
from ..pretraining.data.dataloader import get_loader

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Bio_ClinicalBERT_PATH = './Bio_ClinicalBERT'


class MedKCOModel(torch.nn.Module):
    def __init__(self, vision_type='resnet_v1', bert_type=Bio_ClinicalBERT_PATH, vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None,
                 out_path=None, image_size=512, caption="An Optical coherence tomography(OCT) photograph of [CLS]", projection=True,
                 norm_features=True, test_code=False, modality="CFP"):
        ### caption for test
        # CFP: A fundus photograph of [CLS]
        # OCT: An Optical coherence tomography(OCT) photograph of [CLS]
        # CXR: A chest x-ray photograph of [CLS]
        super().__init__()
        if test_code:
            global device
            device = "cuda:0"

        self.modality = modality
        self.image_size = image_size
        self.caption = caption
        self.from_checkpoint = from_checkpoint
        self.weights_path = weights_path
        self.out_path = out_path

        self.projection = projection
        self.norm_features = norm_features
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.vision_type = vision_type
        self.bert_type = bert_type
        self.vision_pretrained = vision_pretrained
        self.logit_scale_init_value = logit_scale_init_value
        self.vision_model = VisionModel(vision_type=self.vision_type, pretrained=self.vision_pretrained,
                                        proj_dim=self.proj_dim, proj_bias=self.proj_bias, projection=self.projection,
                                        norm=self.norm_features)
        self.text_model = TextModel(bert_type=self.bert_type, proj_dim=self.proj_dim, proj_bias=self.proj_bias,
                                    projection=self.projection, norm=self.norm_features)
        self.logit_scale = torch.nn.Parameter(torch.log(torch.tensor(1/self.logit_scale_init_value)))

        if from_checkpoint:
            self.load_from_pretrained(self.weights_path)
        self.to(device)

    def load_from_pretrained(self, weights_path=None):
        state_dict = torch.load(weights_path, map_location="cuda:0")
        self.load_state_dict(state_dict, strict=False)
        print('load model weight from:', weights_path)


    #########################################
    # model train
    #########################################
    def self_paced_asymmetry_clip_loss(self, logits_per_text, target_pseudo, beta=1.):
        caption_loss = self.ce_loss(logits_per_text, target_pseudo)
        image_loss = self.ce_loss(logits_per_text.T, target_pseudo)
        loss = (caption_loss * beta + image_loss) / 2.0
        return loss


    def ce_loss(self, pred_logit, ref):
        ce_loss = torch.nn.functional.cross_entropy(pred_logit, ref)
        return ce_loss


    def compute_logits(self, img_emb, text_emb):
        # similarity compute
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text


    def reduce_tensor(self, tensor: torch.Tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()
        return rt


    def fit(self, datalaoders, epochs=30, lr=5e-4, weight_decay=1e-5, scheduler=True, warmup_epoch=1, store_num=5,
            transforms=None, local_rank=None, test_code=False, SPCL=False, args=None):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        if scheduler:
            from medkco.pretraining.utils import get_scheduler_per_iteration
            scheduler = get_scheduler_per_iteration(optimizer, lr, warmup_epoch, len(datalaoders["train"]))
        else:
            scheduler = None

        epoch = 1
        while epoch <= epochs:
            loss_epoch = self.train_epoch(datalaoders["train"], optimizer, scheduler, transforms, epoch, test_code=test_code, epochs=epochs)

            if local_rank==0:
                print('Epoch=%d: ave_loss=%2.5f' % (epoch, loss_epoch))
            if (epoch % store_num == 0) & ((local_rank==0 & test_code==False) or test_code == True):
                if self.out_path is not None:
                    if not os.path.isdir(self.out_path):
                        os.mkdir(self.out_path)
                    torch.save(self.state_dict(), self.out_path + self.vision_type + '_epoch' + str(epoch) + '.pth')
            epoch += 1

            if SPCL:
                if epoch == 6:
                    datalaoders = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                                             datasets=args.datasets, balance=args.balance, batch_size=args.batch_size,
                                             num_workers=args.num_workers, banned_categories=args.banned_categories,
                                             caption=self.caption, augment_description=args.augment_description,
                                             test_code=args.test_code, SPCL=args.SPCL, epoch=epoch, modality=self.modality)

                if epoch == 11:
                    datalaoders = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                                             datasets=args.datasets, balance=args.balance, batch_size=args.batch_size,
                                             num_workers=args.num_workers, banned_categories=args.banned_categories,
                                             caption=self.caption, augment_description=args.augment_description,
                                             test_code=args.test_code, SPCL=args.SPCL, epoch=epoch, modality=self.modality)

                if epoch == 16:
                    datalaoders = get_loader_MM(args.data_root_path, args.balance, args.batch_size, args.num_workers,
                                                self.caption, args.augment_description, args.test_code, modality=self.modality)


                if epoch == 21:
                    datalaoders = get_loader_MM(args.data_root_path, args.balance, args.batch_size, args.num_workers,
                                                self.caption, args.augment_description, args.test_code, epoch=epoch,
                                                modality=self.modality)


    def train_epoch(self, loader, optimizer, scheduler=None, transforms=None, epoch=1, test_code=False, epochs=25):
        self.train()
        max_grad_norm, scaler = 1, torch.cuda.amp.GradScaler()
        loss_ave = 0.0

        if not test_code:
            loader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=False)
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)
            text_tokens = self.text_model.tokenize(list(batch["report"][0]))
            input_ids = text_tokens["input_ids"].to(device).to(torch.long)
            attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)

            coocurrence = np.array(
                [[iDesc == iiDesc for iDesc in batch["sel_category"]] for iiDesc in batch["sel_category"]], np.float32)
            target = torch.tensor(coocurrence / coocurrence.sum(-1)).to(device).to(torch.float32)

            with autocast():
                print("\nExtracting features...")
                if transforms is not None:
                    images = transforms(images)
                img_embeds = self.vision_model(images)
                text_embeds = self.text_model(input_ids, attention_mask)

                logits_per_text= self.compute_logits(img_embeds, text_embeds)
                beta = 0 + (1/epochs) * epoch
                loss = self.self_paced_asymmetry_clip_loss(logits_per_text, target, beta=beta).to(device)

                if not test_code:
                    loss = self.reduce_tensor(loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_ave += loss.item()
            torch.cuda.empty_cache()
            epoch_iterator.set_description(
                "Epoch=%d: Training (%d / %d Steps) " % (epoch, step + 1, len(loader)) +
                "- loss_value: " + str(round(loss.item(), 3))
            )

            if scheduler is not None:
                scheduler.step()

        self.eval()
        return loss_ave / len(loader)


    #########################################
    # prediction
    #########################################
    def forward(self, image, text):
        self.eval()
        image = self.preprocess_image(image)
        text_input_ids, text_attention_mask = self.preprocess_text(text)

        with torch.no_grad():
            img_embeds = self.vision_model(image)
            text_embeds = self.text_model(text_input_ids, text_attention_mask)
            logits = self.compute_logits(img_embeds, text_embeds).t()
            probs = logits.softmax(dim=-1)
        return probs.cpu().numpy(), logits.cpu().numpy()


    def preprocess_image(self, image):
        if image.dtype != np.float32:
            image = np.float32(image)
        if image.max() > 0:
            image /= 255
        if len(image.shape) > 2:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)
        image = torch.tensor(image)
        sizes = image.shape[-2:]
        max_size = max(sizes)
        scale = max_size / self.image_size
        image = torchvision.transforms.Resize((int(image.shape[-2] / scale), int(image.shape[-1] / scale)))(image)
        image = torch.nn.functional.pad(image, (0, self.image_size - image.shape[-1], 0, self.image_size - image.shape[-2], 0, 0))
        image = image.to(torch.float32).to(device)
        return image


    def preprocess_text(self, text):
        prompts = [self.caption.replace("[CLS]", category) for category in text]
        text_tokens = self.text_model.tokenize(prompts)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)
        return input_ids, attention_mask


    def preprocess_report(self, text):
        text_tokens = self.text_model.tokenize(text)
        input_ids = text_tokens["input_ids"].to(device).to(torch.long)
        attention_mask = text_tokens["attention_mask"].to(device).to(torch.long)
        return input_ids, attention_mask


    def compute_text_embeddings(self, categories, domain_knowledge=False, modality="CFP"):
        text_embeds_dict = {}
        for iKey in range(len(categories)):
            if modality=="CFP" and domain_knowledge and categories[iKey] in list(definitions.keys()):
                descriptions = definitions[categories[iKey]]
                if categories[iKey] not in descriptions:
                    descriptions.append(categories[iKey])
            elif modality=="OCT" and domain_knowledge and categories[iKey] in list(definitions_OCT.keys()):
                descriptions = definitions_OCT[categories[iKey]]
                if categories[iKey] not in descriptions:
                    descriptions.append(categories[iKey])
            elif modality=="CXR" and domain_knowledge and categories[iKey] in list(definitions_CXR.keys()):
                descriptions = definitions_CXR[categories[iKey]]['pos']
                if categories[iKey] not in descriptions:
                    descriptions.append(categories[iKey])
            else:
                descriptions = [categories[iKey]]

            with torch.no_grad():
                descriptions = [self.caption.replace("[CLS]", iDescription) for iDescription in descriptions]
                text_token = self.text_model.tokenizer(descriptions, truncation=True, padding=True, return_tensors='pt')
                input_ids = text_token["input_ids"].to(device).to(torch.long)
                attention_mask = text_token["attention_mask"].to(device).to(torch.long)
                text_embeds = self.text_model(input_ids, attention_mask)
            text_embeds_dict[categories[iKey]] = text_embeds.mean(0).unsqueeze(0)

        text_embeds = torch.concat(list(text_embeds_dict.values()))

        return text_embeds_dict, text_embeds


#########################################
# model architecture
#########################################
class VisionModel(torch.nn.Module):
    def __init__(self, vision_type='resnet', pretrained=True, proj_dim=512, proj_bias=False, projection=True,norm=True):
        super().__init__()
        self.proj_dim = proj_dim

        if vision_type not in ['resnet_v2']:
            print("Vision model should be one of resnet... using resnet.")
            vision_type = "resnet_v2"
        if vision_type == "resnet_v2":
            weights = 'IMAGENET1K_V2' if pretrained else None
            print("Pretrained weights: " + str(weights))
            self.model = torchvision.models.resnet50(weights=weights)
            self.vision_dim = 2048
            self.model.fc = torch.nn.Identity()

        if projection:
            self.out_dim = self.proj_dim
        self.projection_head_vision = ProjectionLayer(layer=torch.nn.Linear(self.vision_dim, self.proj_dim,bias=proj_bias)
                                                      , projection=projection, norm=norm)

    def forward(self, pixel_values):
        embed = self.model(pixel_values)
        embed = self.projection_head_vision(embed)
        return embed


class TextModel(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, projection=True,norm=True):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 256
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)
        self.projection_head_text = ProjectionLayer(layer=torch.nn.Linear(768, proj_dim, bias=proj_bias),
                                                    projection=projection, norm=norm)

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2],output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)
        embed = self.projection_head_text(embed)
        return embed


class ProjectionLayer(torch.nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()
        self.apply_projection = projection
        self.norm_modality = bool(projection * norm)
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):
        if self.norm_modality:
            x = x / x.norm(dim=-1, keepdim=True)
        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = x / x.norm(dim=-1, keepdim=True)
        return x