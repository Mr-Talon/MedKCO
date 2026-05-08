import pandas as pd

from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from medkco.pretraining.data.dataset import Dataset, UniformDataset
from medkco.pretraining.data.transforms import LoadImage, SelectRelevantKeys, CopyDict, ProduceDescription, AugmentDescription


def get_loader(dataframes_path, data_root_path, datasets, balance=False, batch_size=8, num_workers=0,
               banned_categories=None, caption="A fundus photograph of [CLS]", augment_description=True,
               test_code=False, SPCL=False, epoch=1):
    # CFP
    easy_cls = ["hard exudates", "soft exudates", "microaneurysms", "haemorrhages", "media haze", "drusens", "tessellation",
                "laser scar", "optic disc cupping", "tortuous vessels", "asteroid hyalosis", "optic disc pallor", "exudates",
                "cotton wool spots", "colobomas", "preretinal haemorrhage", "myelinated nerve fibers", "tilted disc",
                "vitreous haemorrhage", "large optic cup", "optic atrophy", "fibrosis", "silicon oil", "scar", "nevus",
                "red small dots"]
    mid_cls = ["no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy", "severe diabetic retinopathy",
               "proliferative diabetic retinopathy", "age-related macular degeneration", "pathologic myopia",
               "branch retinal vein occlusion", "epiretinal membrane", "macular scar", "central retinal vein occlusion",
               "optic disc edema", "shunt", "retinal traction", "retinitis", "retinal pigment epithelium changes",
               "retinitis pigmentosa", "haemorrhagic retinopathy", "central retinal artery occlusion",
               "post traumatic choroidal rupture", "choroidal folds", "vasculitis", "branch retinal artery occlusion",
               "plaque", "collaterals", "maculopathy", "severe hypertensive retinopathy", "disc swelling and elevation",
               "dragged disk", "congenital disk abnormality", "peripheral retinal degeneration and break", "yellow-white spots flecks",
               "no proliferative diabetic retinopathy", "hypertensive retinopathy", "geographical age-related macular degeneration",
               "abnormal optic disc", "abnormal vessels", "abnormal macula", "macular edema", "increased cup disc",
               "a disease", "intraretinal microvascular abnormalities", "retina detachment", "normal"]
    hard_cls = ["diabetic macular edema", "no referable diabetic macular edema", "non clinically significant diabetic macular edema",
                "central serous retinopathy", "anterior ischemic optic neuropathy", "parafoveal telangiectasia", "chorioretinitis",
                "macular hole", "optic disc pit maculopathy", "haemorrhagic pigment epithelial detachment", "Vogt-Koyanagi syndrome",
                "glaucoma", "Bietti crystalline dystrophy", "neoplasm", "no glaucoma", "neovascular age-related macular degeneration",
                "cataract", "no cataract", "macroaneurysm", "cystoid macular edema", "acute central serous retinopathy",
                "chronic central serous retinopathy", "neovascularisation"]

    # OCT
    # easy_cls = ["macular hole stage3", "macular hole stage4", "vitreomacular Interface Disease", "epiretinal membrane"]
    # mid_cls = ["age related macular degeneration", "drusen", "diabetic macular edema", "macular hole stage1",
    #            "macular hole stage2", "normal", "macular hole", "central serous retinopathy", "choroidal neovascularization"]
    # hard_cls = ["glaucoma", "diabetic retinopathy", "retinal artery occlusion", "retinal vein occlusion"]

    # CXR
    # easy_cls =['Lung Opacity', 'Consolidation', 'Support Devices']
    # mid_cls =['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Edema', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    #           'Pleural Effusion', 'Emphysema', 'Hernia', 'Infiltration', 'Mass']
    # hard_cls = ['Lung Lesion', 'Pleural Other', 'Fracture', 'Fibrosis', 'Nodule', 'Pleural_Thickening']

    transforms = Compose([
        CopyDict(),
        LoadImage(),
        ProduceDescription(caption=caption),
        AugmentDescription(augment=augment_description),
        SelectRelevantKeys()
    ])

    print("Setting assembly data...")
    data = []
    for iDataset in datasets:
        # curriculum 2
        if SPCL and (iDataset == 'openi' or iDataset == 'mimic-cxr' or iDataset == '39_MM_Retinal_dataset' or iDataset == 'OCT17_MM_Retinal_OCT'):
            continue

        print("Processing data: " + iDataset)
        dataframe = pd.read_csv(dataframes_path + iDataset + ".csv")
        selected_id_list = range(len(dataframe))
        for i in selected_id_list:
            data_i = dataframe.loc[i, :].to_dict()
            if iDataset == 'openi' or iDataset == 'mimic-cxr' or iDataset == '39_MM_Retinal_dataset' or iDataset == 'OCT17_MM_Retinal_OCT':
                data_i["categories"] = [data_i["caption"]]
                data_i["atributes"] = [""]
            else:
                data_i["categories"] = eval(data_i["categories"])
                data_i["atributes"] = eval(data_i["atributes"])

                banned = False
                complexity = -1
                if banned_categories is not None:
                    cls = []
                    if iDataset == 'CheXpert-v1.0':
                        positive, negative, uncertain = data_i["categories"]
                        for i in positive:
                            cls.append(i)
                        for i in uncertain:
                            cls.append(i)
                    else:
                        cls = data_i["categories"]

                    for iCat in cls:
                        # Removes the banned category
                        if iCat in banned_categories:
                            banned = True
                        # remove other stage
                        if epoch == 1 and iCat not in easy_cls:
                            banned = True
                        if epoch == 6 or epoch == 11:
                            if (iCat not in easy_cls) and (iCat not in mid_cls):
                                complexity_i = 3
                            elif iCat in easy_cls:
                                complexity_i = 1
                            else:
                                complexity_i = 2
                            if complexity_i > complexity:
                                complexity = complexity_i
                    if (epoch == 6 and complexity != 2) or (epoch == 11 and complexity != 3):
                        banned = True
                if banned:
                    continue

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