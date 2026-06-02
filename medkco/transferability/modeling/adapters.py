import copy
import torch
import numpy as np
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class AdapterWrapper(object):
    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.model.eval()

    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()
        epoch_iterator = tqdm(data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)
        X, Y = [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)
            with torch.no_grad():
                if transforms is not None:
                    images = transforms(images)
                x = self.model.vision_model(images)
            X.extend(x.cpu().detach().numpy())
            Y.extend(batch["label"].numpy())
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def predict(self, loader, transforms=None):
        return


class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, modality="CFP"):
        super().__init__(model)
        if len(targets)>0:
            self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()), domain_knowledge=domain_knowledge, modality=modality)


# ZS
class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, modality="CFP"):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, modality=modality)

    def predict(self, loader, transforms=None, dataset = None):
        X, refs = self.extract_vision_features(loader)
        X = torch.tensor(X).to(device)
        with torch.no_grad():
            score = torch.matmul(X, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()
        preds = torch.softmax(score, dim=-1)
        preds = preds.detach().cpu().numpy()
        return refs, preds


# Retrieval
class Retrieval(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge)

    def predict(self, loader, transforms=None, dataset = None):
        self.model.eval()
        epoch_iterator = tqdm(loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)
        X, Y, text = [], [], []

        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)
            with torch.no_grad():
                if transforms is not None:
                    images = transforms(images)
                image_embed = self.model.vision_model(images)

                report = batch["label"]
                for i in report:
                    text.append(i)
                text_input_ids, text_attention_mask = self.model.preprocess_report(report)
                text_embeds = self.model.text_model(text_input_ids, text_attention_mask)
            X.extend(image_embed.cpu().detach().numpy())
            Y.extend(text_embeds.cpu().detach().numpy())
        X = np.array(X)
        Y = np.array(Y)
        return X, Y, text