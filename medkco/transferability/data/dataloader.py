"""
迁移部分    数据变换（预处理）；分割训练、验证、测试集；创建dataloader；平衡数据集类别
data transformation (preprocessing); Split train, valid and test sets;
Create dataloader; Balance dataset categories
"""

import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from medkco.pretraining.data.dataset import Dataset
from medkco.pretraining.data.transforms import LoadImage, ImageScaling, CopyDict


# 数据变换（预处理）；分割训练、验证、测试集；创建dataloader
# Data transformation (preprocessing); Split train, valid and test sets; Create dataloader
def get_dataloader_splits(dataframe_path, data_root_path, targets_dict, shots_train="80%", shots_val="0%",
                          shots_test="20%", balance=False, batch_size=8, num_workers=0, seed=0, task="classification",
                          size=(512, 512), batch_size_test=1, dataset=''):
    '''
    dataframe_path 各个数据集路径       data_root_path 原始数据路径         targets_dict 类别名称  category name
    seed=第K折实验 用于随机数生成
    '''

    # 数据变换（预处理）操作集合   Set of data transformation (preprocessing) operations
    if task == "classification" or task == "Retrieval":
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling(size=size)])
    else:
        transforms = Compose([CopyDict(), LoadImage(), ImageScaling()])

    # 读取数据字典中有用的部分（数据&标签）   Read the useful parts of the data dictionary (data & labels)
    # 输出：字典列表   Output: List of dictionaries
    data = []
    dataframe = pd.read_csv(dataframe_path)
    for i in range(len(dataframe)):
        sample_df = dataframe.loc[i, :].to_dict()                                      # 将每行变为字典  image,attributes,categories   Turn each line into a dictionary

        data_i = {"image_path": data_root_path + sample_df["image"]}                   # 构造需要的数据   Construct the required data
        if task == "classification":
            data_i["label"] = targets_dict[eval(sample_df["categories"])[0]]           # 类别名称=》编号   Category Name = Number
            if dataset == 'Angiographic' and '%' in shots_train:
                # 将标签变为k-hot向量 用于zs和lp
                khot_label = np.zeros(23, dtype=int)
                for cls in data_i["label"]:
                    khot_label[cls] = 1
                data_i["label"] = khot_label
            elif dataset == 'Angiographic' and '%' not in shots_train:
                # 单标签的情况还是返回标签值     fs过滤多标签数据
                if len(data_i["label"])>1:
                    continue
                else:
                    data_i["label"] = data_i["label"][0]
        elif task == "Retrieval":
            data_i["label"] = sample_df["caption"]

        data.append(data_i)

    random.seed(seed)
    random.shuffle(data)

    # 分割训练、验证、测试集       希望每个集合的类别分布一致
    # Splitting the train, valid, and test sets
    # expected to have a consistent distribution of categories for each set
    data_train, data_val, data_test = [], [], []
    if task == "classification":
        labels = [data_i["label"] for data_i in data]                                       # 数据集标签列表  Label List

        if dataset == 'Angiographic' and '%' in shots_train:
            # 多分类数据集单独处理
            # zs不需要对数据集划分 不做处理
            # fs由于有一个类别不足10个 去除多标签的样本 转为单标签分类问题 不需要特殊处理
            # finetune训练集按类划分的时候 多标签的算作一个新类划分 但不影响最后多分类

            unique_labels = targets_dict.values()

            # unique_labels转为khot的形式
            unique_labels_khot = []
            for label in unique_labels:
                khot_label = np.zeros(23, dtype=int)
                for cls in label:
                    khot_label[cls] = 1
                unique_labels_khot.append(khot_label)
            unique_labels = unique_labels_khot
        else:
            unique_labels = np.unique(labels)

        # 无划分 0 0 100   80 0 20
        # 有划分 0 0 100   100 0 0
        for iLabel in unique_labels:
            if dataset == 'Angiographic':
                idx = list(np.squeeze([i for i, label in enumerate(labels) if np.all(label == iLabel)]))
            else:
                idx = list(np.squeeze(np.argwhere(labels == iLabel)))

            train_samples = get_shots(shots_train, len(idx))
            val_samples = get_shots(shots_val, len(idx))
            test_samples = get_shots(shots_test, len(idx))

            if dataset != 'TAOP' and dataset != 'COVIDx':
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


# dataset + dataloader
def get_loader(data, transforms, split, batch_size, num_workers):

    if len(data) == 0:
        loader = None
    else:
        dataset = Dataset(data=data, transform=transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle = split == "train", num_workers=num_workers, drop_last=False, pin_memory=False)
    return loader


# 平衡训练集中不同类别的分布情况  Balance the distribution of different categories in the training set
def balance_data(data):
    labels = [iSample["label"] for iSample in data]                     # 所有样本的label  label of all sample
    unique_labels = np.unique(labels)                                   
    counts = np.bincount(labels)                                        # 统计每个类别的样本个数  Count the number of samples for each category

    N_max = np.max(counts)

    data_out = []
    for iLabel in unique_labels:
        idx = list(np.argwhere(np.array(labels) == iLabel)[:, 0])

        # 如果当前类别在训练集中较少 随机选一些样本（该类）进行重复，使得所有类别的数量一样
        # If the current category is less in the training set, some samples are randomly selected to be repeated,
        # so that the number of all categories is the same
        if N_max-counts[iLabel] > 0:
            idx += random.choices(idx, k=N_max-counts[iLabel])
        [data_out.append(data[iidx]) for iidx in idx]

    return data_out


# 返回样本个数  Return sample number
def get_shots(shots_str, N):
    # 输入百分比  Input percentage
    if "%" in str(shots_str):
        shots_int = int(int(shots_str[:-1]) / 100 * N)
    # 直接输入个数  Direct input number
    else:
        shots_int = int(shots_str)
    return shots_int
