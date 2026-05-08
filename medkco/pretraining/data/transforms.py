import numpy as np
import random
import torch
import copy

from PIL import Image, ImageFile
from torchvision.transforms import Resize
from medkco.modeling.dictionary import definitions, definitions_OCT, definitions_CXR
from kornia.augmentation import RandomHorizontalFlip, RandomAffine, ColorJitter

ImageFile.LOAD_TRUNCATED_IMAGES = True

# CFP/OCT
augmentations_pretraining = torch.nn.Sequential(RandomHorizontalFlip(p=0.5),
                                                RandomAffine(p=0.25, degrees=(-5, 5), scale=(0.9, 1)),
                                                ColorJitter(p=0.25, brightness=0.2, contrast=0.2))
# CXR
# augmentations_pretraining = torch.nn.Sequential(RandomHorizontalFlip(p=0.5),
#                                                 RandomAffine(p=1, degrees=(-10, 10), scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
#                                                 ColorJitter(p=1, brightness=(0.8, 1.2), contrast=(0.8, 1.2)))


# Used for prediction
class LoadImage():
    def __init__(self, target="image_path"):
        self.target = target
    def __call__(self, data):
        img = np.array(Image.open(data[self.target]), dtype=float)
        if np.max(img) > 1:
            img /= 255
        if len(img.shape) > 2:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.expand_dims(img, 0)
        if img.shape[0] > 3:
            img = img[1:, :, :]
        if "image" in self.target:
            if img.shape[0] < 3:
                img = np.repeat(img, 3, axis=0)
        data[self.target.replace("_path", "")] = img
        return data


# Used for prediction/pretrian (if not preprocessed)
class ImageScaling():
    def __init__(self, size=(512, 512), canvas=True, target="image"):
        self.size = size
        self.canvas = canvas
        self.target = target
        self.transforms = torch.nn.Sequential(
            Resize(self.size, antialias=True),
        )
    def __call__(self, data):
        img = torch.tensor(data[self.target])
        if not self.canvas or (img.shape[-1] == img.shape[-2]):
            img = self.transforms(img)
        else:
            sizes = img.shape[-2:]
            max_size = max(sizes)
            scale = max_size/self.size[0]
            img = Resize((int(img.shape[-2]/scale), int(img.shape[-1]/scale)), antialias=True)(img)
            img = torch.nn.functional.pad(img, (0, self.size[0] - img.shape[-1], 0, self.size[1] - img.shape[-2], 0, 0))
        data[self.target] = img
        return data


class ProduceDescription():
    def __init__(self, caption):
        self.caption = caption
    def __call__(self, data):
        atr_sample = random.sample(data['atributes'], 1)[0] if len(data['atributes']) > 0 else ""
        cat_sample = None
        if "CheXpert-v1.0" in data["image_name"]:
            cat_sample = data['categories']
            data["sel_category"] = str(cat_sample)
        else:
            # Used for CPF/OCT and testset of CXR
            cat_sample = random.sample(data['categories'], 1)[0] if len(data['categories']) > 0 else ""
            data["sel_category"] = cat_sample

        if "CheXpert-v1.0" in data["image_name"]:
            report = ""
            positive, negative, uncertain = cat_sample
            for pos in positive:
                cand = definitions_CXR[pos]["pos"]
                sentence = random.choice(cand)
                if len(sentence) > 0:
                    report+= " " + sentence
            for neg in negative:
                cand = definitions_CXR[neg]["neg"]
                sentence = random.choice(cand)
                if len(sentence) > 0:
                    report+= " " + sentence
            for unc in uncertain:
                cand = definitions_CXR[unc]["unc"]
                sentence = random.choice(cand)
                if len(sentence) > 0:
                    report+= " " + sentence
            data["report"] = [report]
        elif ('OCT17_MM_Retinal_OCT' in data["image_name"] or '39_MM_Retinal_dataset' in data["image_name"] or
              'mimic-cxr' in data["image_name"] or 'openi' in data["image_name"]):
            # description dataset
            data["report"] = [cat_sample]
        else:
            # CFP OCT label dataset
            data["report"] = [self.caption.replace("[ATR]",  atr_sample).replace("[CLS]",  cat_sample).replace("  ", " ")]
        return data


class AugmentDescription():
    def __init__(self, augment=False):
        self.augment = augment
    def __call__(self, data):
        if self.augment:
            if data["image_name"].split("/")[0] not in ["06_EYENET", "11_STARE", "08_ODIR-5K", "31_JICHI", "39_MM_Retinal_dataset",
                                                        "OCT17_MM_Retinal_OCT", "CheXpert-v1.0", "mimic-cxr", "openi"]:
                if data["sel_category"] in list(definitions_OCT.keys()):
                    prompts = [data["sel_category"]] + definitions_OCT[data["sel_category"]]
                    new_cat = random.sample(prompts, 1)[0]
                    data["report"][0] = data["report"][0].replace(data["sel_category"], new_cat)
                    data["augmented_category"] = new_cat
        return data


class CopyDict():
    def __call__(self, data):
        d = copy.deepcopy(data)
        return d


class SelectRelevantKeys():
    def __init__(self, target_keys=None):
        if target_keys is None:
            target_keys = ['image', 'report', 'sel_category']
        self.target_keys = target_keys
    def __call__(self, data):
        d = {key: data[key] for key in self.target_keys}
        return d