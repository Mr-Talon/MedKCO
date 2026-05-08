import copy
import random
import torch
import math
import numpy as np
from tqdm import tqdm

from medkco.pretraining.data.transforms import augmentations_pretraining


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


"""
纯视觉适配器 只使用图像编码器
pure vision adapter uses only the image encoder
"""
# 适配器的父类    初始化；实现训练接口；训练、预测虚函数；抽取视觉特征和标签
# Initialization of the adapter's parent class;
# Implement training interface; Training and predicting virtual functions; Extract visual features and labels
class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        '''
        targets：类别  category     tta：测试阶段数据增强  test time data augmentation   fta：训练时数据增强  train/fit time data augmentation
        '''
        self.model = copy.deepcopy(model)
        self.model.eval()                               # 冻结编码器参数  Freezing encoder parameter
        self.num_targets = len(targets)                 # 类别个数  Number of classes
        self.tta = tta
        self.fta = fta
        self.number_augmentations = 20                  # 训练增强次数  enhancement times for fta/tta

    # 获取视觉特征和标签  Get visual features and labels
    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()
        epoch_iterator = tqdm(data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)

        # 对于后续的适配器 输入是CLIP视觉编码器得到的特征向量 输出是类别号
        # adapter input is the feature vector of CLIP visual encoder and the output is the class number
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

    # 训练的接口：进行数据增强、抽视觉特征、调用训练函数
    # Training interface:  data augmentation, extract visual features, call training functions
    def fit(self, loaders, transforms=None, dataset=None):
        data_loader = loaders["train"]                                                         # 训练集  img path 标签/mask

        # 是否使用训练增强策略 增加训练数据  use augmentation strategies to increase training data?
        if self.fta:
            transforms = augmentations_pretraining
        # 获取视觉特征  get visual features
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)                                                       # 合并成一维  Merge into one dimension
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)

        self.train(X, Y, dataset)

    # 训练用虚函数  Virtual functions for training
    def train(self, X, Y, dataset):
        """
        虚函数 由具体适配器实现  Implemented by a specific adapter
        """
        return

    # 预测用虚函数  Virtual functions for prediction
    def predict(self, loader, transforms=None):
        """
        虚函数 由具体适配器实现  Implemented by a specific adapter
        """
        return


"""
多模态适配器   Multimodal adapter
"""
# 多模态适配器父类      继承适配器父类；增加文本特征的抽取
# Multimodal adapter parent class inherits the adapter parent class; Increase the extraction of text features
class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # 输入类别名称    输出对应类别的文本特征（有/无领域知识）
        # Input category name. Output text characteristics corresponding to the category (with/without domain knowledge)
        if len(targets)>0:
            self.text_embeds_dict, self.text_embeds = model.compute_text_embeddings(list(targets.keys()), domain_knowledge=domain_knowledge)


# ZS
class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

    # 清空训练接口的操作  Clear the operation of the training interface
    def fit(self, loaders, transforms=None, dataset=None):
        return

    def predict(self, loader, transforms=None, dataset = None):
        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)                # 获取图像特征和标签  Get image features and labels
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.model.logit_scale.exp()    # 计算相似度 compute similarity     X:N * embd      text_embeds：K * embd
                scores.append(score_i.unsqueeze(-1))                                                                # score_i.unsqueeze(-1)：N * K * 1
            score = torch.concat(scores, -1).mean(-1)                                                               # score=logits     N * K
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = torch.matmul(X, self.text_embeds.t().to(device)) * self.model.logit_scale.exp()

        if dataset == 'Angiographic':
            preds = torch.sigmoid(score)
        else:
            preds = torch.softmax(score, dim=-1)                                                                        # softmax
        preds = preds.detach().cpu().numpy()
        return refs, preds


# Retrieval
class Retrieval(LanguageAdapterWrapper):
    def __init__(self, model, targets, domain_knowledge=False, tta=False, fta=False):
        super().__init__(model, targets, domain_knowledge=domain_knowledge, tta=tta, fta=fta)

    # 清空训练接口的操作  Clear the operation of the training interface
    def fit(self, loaders, transforms=None, dataset=None):
        return

    def predict(self, loader, transforms=None, dataset = None):
        # 抽特征
        self.model.eval()
        epoch_iterator = tqdm(loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True)

        X, Y, text = [], [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():
                # 图像特征
                if transforms is not None:
                    images = transforms(images)
                image_embed = self.model.vision_model(images)

                # 文本特征
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