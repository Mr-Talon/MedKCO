import pandas as pd

from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from medkco.pretraining.data.dataset import Dataset, UniformDataset
from medkco.pretraining.data.transforms import LoadImage, SelectRelevantKeys, CopyDict, ProduceDescription, AugmentDescription


def get_loader_MM(data_root_path, balance=False, batch_size=8, num_workers=0, caption="A fundus photograph of [CLS]",
                  augment_description=True, test_code=False, epoch=1, modality="CFP"):
    transforms = Compose([
        CopyDict(),
        LoadImage(),
        # ImageScaling(),
        ProduceDescription(caption=caption),
        AugmentDescription(augment=augment_description, modality=modality),
        SelectRelevantKeys()
    ])

    print("Setting assembly data...")
    data = []
    dataframe = ""
    if modality=="CFP":
        dataframe = pd.read_csv("./local_data/dataframes/pretrainingCFP/39_MM_Retinal_dataset.csv")
    elif modality=="OCT":
        dataframe = pd.read_csv("./local_data/dataframes/pretrainingOCT/OCT17_MM_Retinal_OCT.csv")
    elif modality=="CXR":
        dataframe = pd.read_csv("./local_data/dataframes/pretrainingCXR/mimic-cxr.csv")

    selected_id_list = range(len(dataframe))
    for i in selected_id_list:
        data_i = dataframe.loc[i, :].to_dict()

        # remove other stage of curriculum 2
        if (epoch == 16 and data_i['stage'] != 1) or (epoch == 21 and data_i['stage'] != 2):
            continue
        data_i["categories"] = [data_i["caption"]]
        data_i["atributes"] = [""]
        data_i["image_name"] = data_i["image"]
        data_i["image_path"] = data_root_path + data_i["image"]
        data.append(data_i)
    print('Total assembly data samples: {}'.format(len(data)))

    if balance:
        train_dataset = UniformDataset(data=data, transform=transforms)
    else:
        train_dataset = Dataset(data=data, transform=transforms)

    if not test_code:
        train_sampler = DistributedSampler(train_dataset)
    else:
        pass

    if test_code:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)

    dataloaders = {"train": train_loader}
    return dataloaders