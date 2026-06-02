import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from medkco.pretraining.data.dataset import Dataset
from medkco.pretraining.data.transforms import LoadImage, ImageScaling, CopyDict


def get_dataloader_splits(dataframe_path, data_root_path, targets_dict, shots_train="80%", shots_val="0%",
                          shots_test="20%", balance=False, batch_size=8, num_workers=0, seed=0, task="classification",
                          size=(512, 512), batch_size_test=1, dataset=''):
    if task == "classification" or task == "Retrieval":
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling(size=size)])
    else:
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling()])

    data = []
    dataframe = pd.read_csv(dataframe_path)
    for i in range(len(dataframe)):
        sample_df = dataframe.loc[i, :].to_dict()
        data_i = {"image_path": data_root_path + sample_df["image"]}
        if task == "classification":
            data_i["label"] = targets_dict[eval(sample_df["categories"])[0]]
        elif task == "Retrieval":
            data_i["label"] = sample_df["caption"]
        data.append(data_i)
    random.seed(seed)
    random.shuffle(data)

    data_train, data_val, data_test = [], [], []
    if task == "classification":
        labels = [data_i["label"] for data_i in data]
        unique_labels = np.unique(labels)

        for iLabel in unique_labels:
            idx = list(np.squeeze(np.argwhere(labels == iLabel)))
            train_samples = get_shots(shots_train, len(idx))
            val_samples = get_shots(shots_val, len(idx))
            test_samples = get_shots(shots_test, len(idx))

            if dataset != 'COVIDx':
                [data_test.append(data[iidx]) for iidx in idx[:test_samples]]
                [data_train.append(data[iidx]) for iidx in idx[test_samples:test_samples+train_samples]]
                [data_val.append(data[iidx]) for iidx in idx[test_samples+train_samples:test_samples+train_samples+val_samples]]
            else:
                dataframe = pd.read_csv(dataframe_path.replace("train", "test"))
                for i in range(len(dataframe)):
                    sample_df = dataframe.loc[i, :].to_dict()
                    data_i = {"image_path": data_root_path + sample_df["image"]}
                    if task == "classification":
                        data_i["label"] = targets_dict[eval(sample_df["categories"])[0]]
                    data_test.append(data_i)
                [data_train.append(data[iidx]) for iidx in idx[test_samples:test_samples + train_samples]]
                [data_val.append(data[iidx]) for iidx in idx[test_samples + train_samples:test_samples + train_samples + val_samples]]

        if balance:
            data_train = balance_data(data_train)
    elif task == "Retrieval":
        data_test = data

    train_loader = get_loader(data_train, transforms, "train", batch_size, num_workers)
    val_loader = get_loader(data_val, transforms, "val", batch_size_test, num_workers)
    test_loader = get_loader(data_test, transforms, "test", batch_size_test, num_workers)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    return loaders


def get_loader(data, transforms, split, batch_size, num_workers):
    if len(data) == 0:
        loader = None
    else:
        dataset = Dataset(data=data, transform=transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle = split == "train", num_workers=num_workers, drop_last=False, pin_memory=False)
    return loader


def balance_data(data):
    labels = [iSample["label"] for iSample in data]
    unique_labels = np.unique(labels)                                   
    counts = np.bincount(labels)
    N_max = np.max(counts)
    data_out = []
    for iLabel in unique_labels:
        idx = list(np.argwhere(np.array(labels) == iLabel)[:, 0])
        if N_max-counts[iLabel] > 0:
            idx += random.choices(idx, k=N_max-counts[iLabel])
        [data_out.append(data[iidx]) for iidx in idx]
    return data_out


def get_shots(shots_str, N):
    if "%" in str(shots_str):
        shots_int = int(int(shots_str[:-1]) / 100 * N)
    else:
        shots_int = int(shots_str)
    return shots_int