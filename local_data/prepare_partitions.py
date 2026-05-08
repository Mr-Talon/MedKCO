import os
import json
import glob
import re
import random
from PIL import Image
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from constants import *


if not os.path.exists(PATH_DATAFRAME_PRETRAIN):
    os.mkdir(PATH_DATAFRAME_PRETRAIN)
if not os.path.exists(PATH_DATAFRAME_TRANSFERABILITY):
    os.mkdir(PATH_DATAFRAME_TRANSFERABILITY)
if not os.path.exists(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION):
    os.mkdir(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION)


def adequate_01_eyepacs():
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy"}
    path_dataset = "01_Eyepacs/"

    partitions = ["train", "valid"]
    data = []
    for iPartition in partitions:
        dataframe = pd.read_csv(PATH_DATASETS + path_dataset + iPartition + ".csv")
        for iFile in range(dataframe.shape[0]):
            image_path = path_dataset + iPartition + '/' + dataframe["image"][iFile] + '.jpeg'
            categories, atributes = [], []

            categories.append(labels_dr[dataframe["level"][iFile]])
            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "01_EYEPACS.csv")
    print(f'01_EYEPACS has {len(data)} images')


def adequate_03_idrid():
    path_dataset = "03_IDRiD/"
    data = []

    # A.Segmentation
    subpath = "A. Segmentation/"
    subpath_images = "1. Original Images/a. Training Set/"
    subpath_gt = "2. All Segmentation Groundtruths/a. Training Set/"
    annotations_paths = ["1. Microaneurysms", "2. Haemorrhages", "3. Hard Exudates", "4. Soft Exudates"]
    annotations_abbreviations = ["MA", "HE", "EX", "SE"]
    annotations_categories = ["microaneurysms", "haemorrhages", "hard exudates", "soft exudates"]

    files_segmentation = os.listdir(PATH_DATASETS + path_dataset + subpath + subpath_images)

    for iFile in files_segmentation:
        image_path = path_dataset + subpath + subpath_images + iFile

        categories = []
        atributes = []
        for i in range(len(annotations_categories)):
            annotation_path = PATH_DATASETS + path_dataset + subpath + subpath_gt + annotations_paths[i] + "/" \
                              + iFile.replace(".jpg", "_" + annotations_abbreviations[i] + ".tif")
            if os.path.isfile(annotation_path):
                categories.append(annotations_categories[i])

        data.append({"image": image_path,
                     "atributes": atributes,
                     "categories": categories})

    # B.Grading
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy"}
    labels_dme = {0: "no referable diabetic macular edema", 1: "non clinically significant diabetic macular edema",
                  2: "diabetic macular edema"}

    subpath = "B. Disease Grading/"
    subpath_images = "1. Original Images/a. Training Set/"
    dataframe = "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"

    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + subpath + dataframe)
    for iFile in range(dataframe.shape[0]):
        image_path = path_dataset + subpath + subpath_images + dataframe["Image name"][iFile] + ".jpg"
        categories = []
        atributes = []

        categories.append(labels_dr[dataframe["Retinopathy grade"][iFile]])
        categories.append(labels_dme[dataframe["Risk of macular edema "][iFile]])

        if os.path.isfile(PATH_DATASETS + image_path):
            data.append({"image": image_path,
                         "atributes": atributes,
                         "categories": categories})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "03_IDRID.csv")
    print(f'03_IDRiD has {len(data)} images')


def adequate_04_rfmid():
    template_diseases = {'DR': 'diabetic retinopathy', 'ARMD': 'age-related macular degeneration',
                         'MH': 'media haze', 'DN': 'drusens', 'MYA': 'myopia', 'BRVO': 'branch retinal vein occlusion',
                         'TSLN': 'tessellation', 'ERM': 'epiretinal membrane', 'LS': 'laser scar',
                         'MS': 'macular scar', 'CSR': 'central serous retinopathy', 'ODC': 'optic disc cupping',
                         'CRVO': 'central retinal vein occlusion', 'TV': 'tortuous vessels', 'AH': 'asteroid hyalosis',
                         'ODP': 'optic disc pallor', 'ODE': 'optic disc edema', 'ST': 'shunt',
                         'AION': 'anterior ischemic optic neuropathy', 'PT': 'parafoveal telangiectasia',
                         'RT': 'retinal traction', 'RS': 'retinitis', 'CRS': 'chorioretinitis', 'EDN': 'edudates',
                         'RPEC': 'retinal pigment epithelium changes', 'MHL': 'macular hole',
                         'RP': 'retinitis pigmentosa', 'CWS': 'cotton wool spots', 'CB': 'colobomas',
                         'ODPM': 'optic disc pit maculopathy', 'PRH': 'preretinal haemorrhage',
                         'MNF': 'myelinated nerve fibers', 'HR': 'haemorrhagic retinopathy',
                         'CRAO': 'central retinal artery occlusion', 'TD': 'tilted disc',
                         'CME': 'cystoid macular edema', 'PTCR': 'post traumatic choroidal rupture',
                         'CF': 'choroidal folds', 'VH': 'vitreous haemorrhage', 'MCA': 'macroaneurysm',
                         'VS': 'vasculitis', 'BRAO': 'branch retinal artery occlusion', 'PLQ': 'plaque',
                         'HPED': 'haemorrhagic pigment epithelial detachment', 'CL': 'collaterals'}

    path_dataset = "04_RFMiD/"
    partitions = ["Training", "Validation", "Testing"]
    letters = ["a", "b", "c"]
    data = []
    for iPartition in range(len(partitions)):
        subpath_images = "1. Original Images/" + letters[iPartition] + ". " + partitions[iPartition] + " Set/"
        subpath_dataframe = "2. Groundtruths/" + letters[iPartition] + ". RFMiD_" + partitions[
            iPartition] + "_Labels.csv"

        dataframe = pd.read_csv(PATH_DATASETS + path_dataset + subpath_dataframe)
        for iFile in range(dataframe.shape[0]):
            image_path = path_dataset + subpath_images + str(dataframe["ID"][iFile]) + ".png"
            categories, atributes = [], []

            if dataframe["Disease_Risk"][iFile] == 1:
                categories.append("a disease")
                ids = np.argwhere(np.array(dataframe)[iFile, 2:])

                for i in list(ids):
                    dis_abreviation = dataframe.columns.to_list()[i[0] + 2]
                    categories.append(template_diseases[dis_abreviation])
            else:
                categories.append("no disease")
                categories.append("healthy")

            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "04_RFMid.csv")
    print(f'04_RFMiD has {len(data)} images')


def adequate_06_DEN():
    path_dataset = "06_DEN/"
    data = []

    partitions = ["DeepEyeNet_train.json", "DeepEyeNet_test.json", "DeepEyeNet_valid.json"]
    for iPartition in partitions:
        f = open(PATH_DATASETS + path_dataset + iPartition)
        meta = json.load(f)

        for iSample in meta:
            image_path = path_dataset + list(iSample.keys())[0]
            categories, atributes = [], []

            info = iSample[list(iSample.keys())[0]]
            categories.extend(info["keywords"].split(", "))
            categories.extend(info["clinical-description"].split(". "))

            if "" in categories:
                categories.remove("")

            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "06_DEN.csv")
    print(f'06_DEN has {len(data)} images')


def adequate_07_lag():
    path_dataset = "07_LAG/"
    categories_paths = ["non_glaucoma", "suspicious_glaucoma"]
    categories = ["no glaucoma", "glaucoma"]

    data = []
    for i in range(len(categories_paths)):
        images = os.listdir(PATH_DATASETS + path_dataset + categories_paths[i] + "/image/")

        for iImage in images:
            data.append({"image": path_dataset + categories_paths[i] + "/image/" + iImage,
                         "atributes": [],
                         "categories": [categories[i]]})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "07_LAG.csv")
    print(f'07_LAG has {len(data)} images')


def adequate_08_odir5k():
    path_dataset = "08_ODIR-5K/"
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "full_df.csv")

    # Train and Incremental test division
    mask_cat = np.logical_and(dataframe["C"].values == 1,
                              dataframe[["D", "G", "C", "A", "H", "M", "O"]].values.sum(-1) == 1)
    mask_myo = np.logical_and(dataframe["M"].values == 1,
                              dataframe[["D", "G", "C", "A", "H", "M", "O"]].values.sum(-1) == 1)
    mask_normal = np.logical_and(dataframe["N"].values == 1,
                                 dataframe[["D", "G", "C", "A", "H", "M", "O"]].values.sum(-1) == 0)
    mask_normal[np.argwhere(mask_normal == 1)[200:]] = False
    mask_test = np.logical_or(mask_cat, mask_myo)
    mask_test = np.logical_or(mask_test, mask_normal)

    dataframe_train = dataframe[np.logical_not(mask_test)]
    dataframe_test = dataframe[mask_test]
    # Train subset
    data = []
    for iFile in range(dataframe_train.shape[0]):
        id = dataframe_train["ID"].values[iFile]
        for iEye in ["Right", "Left"]:
            image_path = path_dataset + "train_resized/" + str(id) + "_" + (iEye).lower() + ".jpg"
            categories = []
            description = dataframe_train[(iEye + "-Diagnostic Keywords")].values[iFile]
            if "myop" not in description and "cataract" not in description:
                categories.extend(description.split("，"))
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "08_ODIR.csv")
    print(f'08_ODIR-5K has {len(data)} images')
    # Test subset
    data = []
    counter_n, counter_m, counter_c = 1, 1, 1
    for iFile in range(dataframe_test.shape[0]):
        id = dataframe_test["ID"].values[iFile]

        for iEye in ["Right", "Left"]:
            image_path = path_dataset + "preprocessed_images/" + str(id) + "_" + (iEye).lower() + ".jpg"
            description = dataframe_test[(iEye + "-Diagnostic Keywords")].values[iFile]
            if "myop" in description and counter_m <= 200:
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": ["pathologic myopia"]})
                    counter_m += 1
            if "cataract" in description and counter_c <= 200:
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": ["cataract"]})
                    counter_c += 1
            if "normal" in description and counter_n <= 200:
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": ["normal"]})
                    counter_n += 1
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "08_ODIR200x3.csv")


def adequate_09_papila():
    path_dataset = "09_PAPILA/"
    subpath_images = "FundusImages/"
    dataframes = [pd.read_excel(PATH_DATASETS + path_dataset + "ClinicalData/patient_data_od.xlsx"),
                  pd.read_excel(PATH_DATASETS + path_dataset + "ClinicalData/patient_data_os.xlsx")]
    labels_glaucoma = {0: "normal", 1: "glaucoma", 2: "glaucoma"}
    data = []
    for iFile in range(dataframes[0].shape[0] - 2):
        id = dataframes[0]["Unnamed: 0"][iFile + 2][1:]

        image_path = path_dataset + subpath_images + "RET" + id + "OD" + ".jpg"
        if os.path.isfile(PATH_DATASETS + image_path):
            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [labels_glaucoma[dataframes[0]["Diagnosis"][iFile + 2]]]})

        image_path = path_dataset + subpath_images + "RET" + id + "OS" + ".jpg"
        if os.path.isfile(PATH_DATASETS + image_path):
            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [labels_glaucoma[dataframes[1]["Diagnosis"][iFile + 2]]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "09_PAPILA.csv")
    print(f'09_PAPILA has {len(data)} images')


def adequate_10_paraguay():
    path_dataset = "10_PARAGUAY/"
    data = []

    subpaths = ["1. No DR signs", "2. Mild (or early) NPDR", "3. Moderate NPDR",
                "4. Severe NPDR", "5. Very Severe NPDR", "6. PDR", "7. Advanced PDR"]
    categories = ["no diabetic retinopathy", "mild diabetic retinopathy",
                  "moderate diabetic retinopathy", "severe diabetic retinopathy",
                  "severe diabetic retinopathy", "proliferative diabetic retinopathy",
                  "proliferative diabetic retinopathy"
                  ]
    for iPath in range(len(subpaths)):
        images = os.listdir(PATH_DATASETS + path_dataset + subpaths[iPath] + "/")

        for iImage in images:
            data.append({"image": path_dataset + subpaths[iPath] + "/" + iImage,
                         "atributes": [],
                         "categories": [categories[iPath]]})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "10_PARAGUAY.csv")
    print(f'10_PARAGUAY has {len(data)} images')


def adequate_11_stare():
    path_dataset = "11_STARE/all-images/"
    data = []
    metadata = "all-mg-codes.txt"

    for line in open(PATH_DATASETS + path_dataset + metadata):
        categories, atributes = [], []
        columns = line.strip().split("\t")

        image_path = path_dataset + "documents/" + columns[0] + ".ppm"
        description = columns[-1].split("\n")[0].lower().split("        ")[-1].replace("\"", "")

        categories.extend(description.replace("\t", "").replace(" and ", " or ").replace("?", "").split(" or "))

        if os.path.isfile(PATH_DATASETS + image_path):
            data.append({"image": image_path,
                         "atributes": [],
                         "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "11_STARE.csv")
    print(f'11_STARE has {len(data)} images')


def adequate_12_aria():
    path_dataset = "12_ARIA/"
    sub_path = 'images/'
    categories = {'a': "age-related macular degeneration",
                  'c': "normal",
                  'd': "diabetic retinopathy"}
    data = []
    for iFile in os.listdir(PATH_DATASETS + path_dataset + sub_path):
        pattern = r"_([a-zA-Z]+)_"
        match = re.search(pattern, iFile)
        categorie = match.group(1)
        data.append({"image": path_dataset + sub_path + "/" + iFile,
                     "atributes": [],
                     "categories": [categories[categorie]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "12_ARIA.csv")
    print(f'12_ARIA has {len(data)} images')


def adequate_13_fives():
    path_dataset = "13_FIVES/"
    images_subpath = ["train/Original/", "test/Original/"]
    labels_dme = {"A": "age related macular degeneration",
                  "D": "diabetic retinopathy",
                  "G": "glaucoma",
                  "N": "normal"}
    data = []
    for iSubpath in images_subpath:
        files = os.listdir(PATH_DATASETS + path_dataset + iSubpath)
        for iFile in files:
            if iFile != "Thumbs.db":
                category__code = iFile.split(".")[0].split("_")[-1]
                data.append({"image": path_dataset + iSubpath + iFile,
                             "atributes": [],
                             "categories": [labels_dme[category__code]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "13_FIVES.csv")
    print(f'13_FIVES has {len(data)} images')


def adequate_14_agar300():
    path_dataset = "14_AGAR300/"
    finding = ["microaneurysms", "diabetic retinopathy"]

    data = []
    files = os.listdir(PATH_DATASETS + path_dataset)
    for iFile in files:
        data.append({"image": path_dataset + iFile,
                     "atributes": [],
                     "categories": finding})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "14_AGAR300.csv")
    print(f'14_AGAR300 has {len(data)} images')


def adequate_15_aptos():
    path_dataset = "15_APTOS/"
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy"}
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "train.csv")
    images_subpath = "train_images/"

    data = []
    for iFile in range(dataframe.shape[0]):
        image_path = path_dataset + images_subpath + dataframe["id_code"][iFile] + ".png"
        categories, atributes = [], []

        categories.append(labels_dr[dataframe["diagnosis"][iFile]])
        if os.path.isfile(PATH_DATASETS + image_path):
            data.append({"image": image_path,
                         "atributes": atributes,
                         "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "15_APTOS.csv")
    print(f'15_APTOS has {len(data)} images')


def adequate_16_fundoct():
    path_dataset = "16_FUND-OCT/"
    data = []
    dict_macula = {'acute CSR': 'acute central serous retinopathy', 'chronic CSR': 'chronic central serous retinopathy',
                   'ME': 'cystoid macular edema', 'geographic_AMD': 'geographical age-related macular degeneration',
                   'Healthy': 'normal', 'neovascular_AMD': 'neovascular age-related macular degeneration',
                   'AMD': 'age-related macular degeneration', 'CSR': 'central serous retinopathy'}
    # Glaucoma/NoGlaucoma
    subpath = "Dataset/OD/"

    data = []
    files = glob.glob(PATH_DATASETS + path_dataset + subpath + "*/*/*/*Color*")
    for iFile in files:
        data.append({"image": iFile.replace(PATH_DATASETS, ""),
                     "atributes": [],
                     "categories": [
                         iFile.replace(PATH_DATASETS, "").split("/")[3].lower().replace("healthy", "normal")]})

    # Macula-related
    subpath = "Dataset/Macula/"
    files = glob.glob(PATH_DATASETS + path_dataset + subpath + "*/*/*/*Color*")
    for iFile in files:
        data.append({"image": iFile.replace(PATH_DATASETS, ""),
                     "atributes": [],
                     "categories": [
                         dict_macula[iFile.replace(PATH_DATASETS, "").split("/")[3]]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "16_FUND-OCT.csv")
    print(f'16_FUND-OCT has {len(data)} images')


def adequate_17_diaretdb1():
    import xml.etree.ElementTree as ET

    path_dataset = "17_DiaRetDB1/ddb1_v02_01/"
    subpath_images = "images/"
    subpath_annotations = "groundtruth/"

    files = os.listdir(PATH_DATASETS + path_dataset + subpath_images)
    data = []
    for iFile in files:
        categories = []
        for annotator in ["_01.xml", "_02.xml", "_03.xml", "_04.xml"]:
            annotation_id = iFile.replace(".png", annotator)
            tree = ET.parse(PATH_DATASETS + path_dataset + subpath_annotations + annotation_id)
            root = tree.getroot()
            for item in root.findall('./markinglist/marking/'):
                if item.tag == 'markingtype':
                    categories.append(
                        item.text.lower().replace("_", " ").replace("irma", "intraretinal microvascular abnormalities"))
        categories = list(np.unique(categories))
        categories.remove("disc")
        if len(categories) == 0:
            continue
        if os.path.exists(PATH_DATASETS + path_dataset + subpath_images + iFile.replace(PATH_DATASETS, "")):
            data.append({"image": path_dataset + subpath_images + iFile.replace(PATH_DATASETS, ""),
                         "atributes": [],
                         "categories": categories})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "17_DiaRetDB1.csv")
    print(f'17_DiaRetDB1 has {len(data)} images')


def adequate_18_drions_db():
    path_dataset = "18_DRIONS-DB/"
    images_subpath = "images/"

    data = []
    files = os.listdir(PATH_DATASETS + path_dataset + images_subpath)
    for iFile in files:
        if iFile.endswith('jpg'):
            data.append({"image": path_dataset + images_subpath + iFile,
                         "atributes": [],
                         "categories": ["no cataract", "a disease"]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "18_DRIONS-DB.csv")
    print(f'18_DRIONS-DB has {len(data)} images')


def adequate_19_drishtigs1():
    path_dataset = "19_Drishti-GS1/"
    dataframe = pd.read_excel(PATH_DATASETS + path_dataset + "Drishti-GS1_diagnosis.xlsx", skiprows=3)[1:]
    subpath_images = ["Drishti-GS1_files/Training/Images/", "Drishti-GS1_files/Test/Images/"]
    data = []
    for iPartition in subpath_images:
        for iFile in range(dataframe.shape[0]):
            id = dataframe["Drishti-GS File"].values[iFile][:-1]
            finding = dataframe["Total"].values[iFile].lower()
            image_path = path_dataset + iPartition + id + ".png"

            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": [],
                             "categories": [finding.replace("glaucomatous", "glaucoma")]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "19_Drishti-GS1.csv")
    print(f'19_Drishti-GS1 has {len(data)} images')


def adequate_20_e_ophta():
    path_dataset = "20_E-ophta/"
    labels = {"EX": "exudates", "healthy": "healthy", "MA": "microaneurysms"}
    subpath_images = ["e_optha_EX/EX/", "e_optha_EX/healthy/", "e_optha_MA/MA/", "e_optha_MA/healthy/"]

    data = []
    for iSub in subpath_images:
        finding = labels[iSub.split("/")[-2]].replace("healthy", "normal")
        for root, dirs, files in os.walk(PATH_DATASETS + path_dataset + iSub):
            for filename in files:
                if filename != "Thumbs.db":
                    print(os.path.join(root, filename))

                    data.append({"image": os.path.join(root, filename).replace(PATH_DATASETS, ""),
                                 "atributes": [],
                                 "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "20_E-ophta.csv")


def adequate_21_g1020():
    path_dataset = "21_G1020/"
    image_subpath = "Images/"
    labels = {0: "normal", 1: "glaucoma"}
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "G1020.csv")

    data = []
    for iFile in range(dataframe.shape[0]):
        id = dataframe["imageID"].values[iFile]
        finding = labels[dataframe["binaryLabels"].values[iFile]]
        image_path = path_dataset + image_subpath + id

        if os.path.isfile(PATH_DATASETS + image_path):
            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "21_G1020.csv")


def adequate_23_hrf():
    path_dataset = "23_HRF/"
    data = []

    # Disease
    image_subpath = "images/"
    labels = {"dr": "diabetic retinopathy", "g": "glaucoma", "h": "normal"}
    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    for iFile in files:
        data.append({"image": path_dataset + image_subpath + iFile,
                     "atributes": [],
                     "categories": [labels[iFile.split("_")[-1].split(".")[0]]]})

    # Noise
    # image_subpath = "Noise/"
    # labels = {"bad": "noisy", "good": "clean"}
    # files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)
    #
    # for iFile in files:
    #     data.append({"image": path_dataset + image_subpath + iFile,
    #                  "atributes": [labels[iFile.split("_")[-1].split(".")[0]]],
    #                  "categories": []})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "23_HRF.csv")


def adequate_24_origa():
    path_dataset = "24_ORIGA/"
    image_subpath = "Images/"
    labels = {0: "no glaucoma", 1: "glaucoma"}
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "OrigaList.csv")

    data = []
    for iFile in range(dataframe.shape[0]):
        id = dataframe["Filename"].values[iFile]
        finding = labels[dataframe["Glaucoma"].values[iFile]]
        image_path = path_dataset + image_subpath + id

        if os.path.isfile(PATH_DATASETS + image_path):
            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "24_ORIGA.csv")


def adequate_25_refuge():
    path_dataset = "25_REFUGE/"
    data = []

    # Disease
    image_subpath = "REFUGE-Training400/"                               # We only have labels for train subset
    labels = {"g": "glaucoma", "n": "no glaucoma"}
    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    for iFile in files:
        data.append({"image": path_dataset + image_subpath + iFile,
                     "atributes": [],
                     "categories": [labels[iFile[0]]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "25_REFUGE.csv")


def adequate_26_roc():
    path_dataset = "26_ROC/"

    files = glob.glob(PATH_DATASETS + path_dataset + "*/*.jpg")
    data = []
    for iFile in files:
        data.append({"image": iFile.replace(PATH_DATASETS, ""),
                     "atributes": [],
                     "categories": ["microaneurysms"]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "26_ROC.csv")


def adequate_27_brset():
    path_dataset = "27_BRSET/"
    image_subpath = "fundus_photos/"
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "labels.csv")

    anatomical_dict = {"1": "normal", "2": "abnormal", 'bv': ""}
    dr_dict = {"0": "no diabetic retinopathy",
               "1": "mild diabetic retinopathy",
               "2": "moderate diabetic retinopathy",
               "3": "severe diabetic retinopathy.",
               "4": "proliferative diabetic retinopathy"}
    findings = ["macular_edema", "scar", "nevus", "amd", "vascular_occlusion", "hypertensive_retinopathy",
                "drusens", "hemorrhage", "retinal_detachment", "myopic_fundus", "increased_cup_disc", "other"]
    find_names = ["macular edema", "scar", "nevus", "age-related macular degeneration", "vascular occlusion",
                  "hypertensive retinopathy", "drusens", "hemorrhage", "retina detachment",
                  "myopia", "increased cup disc", "a disease"]

    data = []
    for iFile in range(dataframe.shape[0]):
        categories, atributes = [], []
        id = dataframe["image_id"].values[iFile] + ".jpg"

        # optic_disc
        categories.append(anatomical_dict[dataframe["optic_disc"].values[iFile]] + " optic disc")
        # vessels
        categories.append(anatomical_dict[str(dataframe["vessels"].values[iFile])] + " vessels")
        # macula
        categories.append(anatomical_dict[str(dataframe["macula"].values[iFile])] + " macula")
        # DR_ICDR
        categories.append(dr_dict[str(dataframe["DR_ICDR"].values[iFile])])
        # Noise
        if dataframe["focus"].values[iFile] == 2 or dataframe["iluminaton"].values[iFile] == 2 \
                or dataframe["image_field"].values[iFile] == 2 or dataframe["image_field"].values[iFile] == 2:
            atributes.append("Noisy")
        # findings
        for i in range(len(findings)):
            if dataframe[findings[i]].values[iFile] == 1:
                categories.append(find_names[i])

        image_path = path_dataset + image_subpath + id
        if os.path.isfile(PATH_DATASETS + image_path):
            data.append({"image": image_path,
                         "atributes": [],
                         "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "27_BRSET.csv")


def adequate_28_OIA():
    path_dataset = "28_OIA-DDR/"
    data = []
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy", 5: ""}

    subpath_grading = "DR_grading/"
    subpath_segmentation = "lesion_segmentation/"
    lesions_path = ["EX/", "HE/", "MA/", "SE/"]
    lesions = ["hard exudates", "haemorrhages", "microaneurysms", "soft exudates"]
    partitions = ["train", "test", "valid"]

    for iPartition in partitions:
        dataframe = pd.read_table(PATH_DATASETS + path_dataset + subpath_grading + iPartition + ".txt", delimiter=" ",
                                  header=None)
        files = list(dataframe[0].values)

        for iFile in range(len(files)):
            categories, atributes = [], []

            image_path = path_dataset + subpath_grading + iPartition + "/" + files[iFile]
            categories.append(labels_dr[dataframe[1].values[iFile]])

            if dataframe[1].values[iFile] == 5:
                atributes.append("noisy")

            for i in range(len(lesions_path)):
                for iiPartition in partitions:
                    if os.path.isfile(
                            PATH_DATASETS + path_dataset + subpath_segmentation + iiPartition + "/" + "label/" +
                            lesions_path[i] + files[iFile].replace(".jpg", ".tif")):
                        categories.append(lesions[i])

            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "28_OIA-DDR.csv")


def adequate_29_airogs():
    path_dataset = "29_AIROGS/"
    labels = {"RG": "glaucoma", "NRG": "no glaucoma"}
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "train_labels.csv")

    data = []
    for i in range(6):
        print(i)
        files = os.listdir(PATH_DATASETS + path_dataset + str(i) + '/')

        for iFile in files:
            id= ""
            if iFile.split(".")[1] != 'db':
                id = dataframe["challenge_id"] == iFile.split(".")[0]

            finding = labels[dataframe[id]["class"].values[0]]
            image_path = path_dataset + str(i) + '/' + iFile

            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": [],
                             "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "29_AIROGS.csv")


def adequate_30_sustech():
    path_dataset = "30_SUSTech-SYSU/"
    image_subpath = "originalImages/"
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy", 5: ""}

    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + image_subpath + "drLabels.csv")
    data_c5 = pd.read_csv(PATH_DATASETS + path_dataset + image_subpath + "c5_DR_reclassified.csv")

    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    data = []
    for iFile in files:
        id = dataframe["Fundus_images"] == iFile
        id_c5 = data_c5["Fundus_images"] == iFile

        if np.argwhere(id.values).__len__() > 0:
            if dataframe[id]["DR_grade(American_Academy_of_Ophthalmology)"].values[0]==5:
                finding = labels_dr[data_c5[id_c5]["DR_grade(American_Academy_of_Ophthalmology)"].values[0]]
            else:
                finding = labels_dr[dataframe[id]["DR_grade(American_Academy_of_Ophthalmology)"].values[0]]

            image_path = path_dataset + image_subpath + iFile

            findings = [finding]
            if os.path.isfile(PATH_DATASETS + image_path):
                if os.path.isfile(PATH_DATASETS + path_dataset + "exudatesLabels/" + iFile.split(".")[0] + ".xml"):
                    findings.append("exudates")

                data.append({"image": image_path,
                             "atributes": [],
                             "categories": findings})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "30_SUSTech-SYSU.csv")


def adequate_31_jichi():
    path_dataset = "31_JICHI/"
    image_subpath = "dmr/"
    labels_dr = {"ndr": ["no diabetic retinopathy"],
                 "sdr": ["microaneurysm", "retinal hemorrhage", "hard exudate", "retinal edema",
                         "more than three small soft exudates"],
                 "ppdr": ["soft exudate", "varicose veins", "intraretinal microvascular abnormality",
                          "non-perfusion area over one disc area"],
                 "pdr": ["neovascularization", "preretinal haemorrhage", "fibrovascular proliferativemembrane",
                         "tractionalretinaldetachment"],
                 }

    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + image_subpath + "list.csv")

    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    data = []
    for iFile in files:
        id = dataframe["Image"] == iFile

        if np.argwhere(id.values).__len__() > 0:

            finding = labels_dr[dataframe[id]["Davis_grading_of_one_figure"].values[0]]
            image_path = path_dataset + image_subpath + iFile

            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": [],
                             "categories": finding})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "31_JICHI.csv")


def adequate_32_chaksu():
    path_dataset = "32_CHAKSU/"
    subpaths = ["Train/", "Test/"]
    data = []
    scanners = ["Bosch", "Forus", "Remidio"]
    dataframe_id = "Glaucoma_Decision_Comparison_[SCAN]_majority.csv"
    path_images = "1.0_Original_Fundus_Images/"
    labels = {"NORMAL": "no glaucoma", "GLAUCOMA SUSPECT": "glaucoma"}
    formats = {'Bosch': '.JPG', 'Forus': '.png', 'Remidio': '.JPG'}

    for iSubpath in subpaths:
        for iScanner in scanners:

            dataframe = pd.read_csv(PATH_DATASETS + path_dataset + iSubpath + '6.0_Glaucoma_Decision/' + dataframe_id.replace("[SCAN]", iScanner))
            files = dataframe["Images"].values.tolist()

            for iFile in files:
                image_path = path_dataset + iSubpath + path_images + iScanner + "/" + iFile.split(".")[0] + formats[
                    iScanner]

                if os.path.isfile(PATH_DATASETS + image_path):
                    if "Majority Decision" in list(dataframe.keys()):
                        ky = "Majority Decision"
                    else:
                        ky = "Glaucoma Decision"

                    finding = labels[dataframe[dataframe["Images"] == iFile][ky].values[0]]

                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "32_CHAKSU.csv")


def adequate_33_dr():
    path_dataset = "33_DR1-2/"
    subpaths = ["Cotton-wool Spots", "Deep Hemorrhages", "Drusen", "Hard Exudates", "Normal Images", "Red Lesions",
                "Superficial Hemorrhages"]
    transfer = {"Cotton-wool Spots": "cotton wool spots", "Deep Hemorrhages": "deep haemorrhages",
                "Drusen": "drusens", "Hard Exudates": "hard exudates", "Normal Images": "normal",
                "Red Lesions": "red small dots", "Superficial Hemorrhages": "superficial haemorrhages"}
    data = []
    for iPath in subpaths:
        files = os.listdir(PATH_DATASETS + path_dataset + 'DR1-images-by-lesions/' + iPath + "/")

        for iFile in files:
            image_path = path_dataset + 'DR1-images-by-lesions/' + iPath + "/" + iFile

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [transfer[iPath]]})

        if iPath in ["Cotton-wool Spots", "Drusen", "Hard Exudates", "Normal Images", "Red Lesions"]:
            files = os.listdir(PATH_DATASETS + path_dataset + 'DR2-images-by-lesions/' + iPath + "/")

            for iFile in files:
                image_path = path_dataset + 'DR2-images-by-lesions/' + iPath + "/" + iFile

                data.append({"image": image_path,
                             "atributes": [],
                             "categories": [transfer[iPath]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "33_DR1-2.csv")


def adequate_34_cataract():
    path_dataset = "34_Cataract/"
    subpaths = ["1_normal", "2_cataract", "2_glaucoma", "3_retina_disease"]
    transfer = {"1_normal": "normal", "2_cataract": "cataract",
                "2_glaucoma": "glaucoma", "3_retina_disease": "retinitis"}
    data = []
    for iPath in subpaths:
        files = os.listdir(PATH_DATASETS + path_dataset + 'dataset/' + iPath + "/")

        for iFile in files:
            image_path = path_dataset + 'dataset/' + iPath + "/" + iFile

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [transfer[iPath]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "34_Cataract.csv")


def adequate_35_scardat():
    path_dataset = "35_ScarDat/"
    subpaths = ["train/", "val/", "test/"]
    subsubsubpaths = ["positive/", "negative/"]

    transfer = {"positive/": "laser scar", "negative/": "no laser scar"}
    data = []
    for iPath in subpaths:
        for iiPath in subsubsubpaths:

            files = os.listdir(PATH_DATASETS + path_dataset + 'laser_scar_dataset_448/' + iPath + iiPath)

            for iFile in files:
                image_path = path_dataset + 'laser_scar_dataset_448/' + iPath + iiPath + iFile

                data.append({"image": image_path,
                             "atributes": [],
                             "categories": [transfer[iiPath]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "35_ScarDat.csv")


def MM_Retinal_dataset():
    path_dataset = "39_MM_Retinal_dataset/"
    subpath_images = ["CFP"]

    data = []
    for iSub in subpath_images:
        dataframe = pd.read_csv(PATH_DATASETS + path_dataset + iSub + "_translated_v1.csv")
        for iFile in range(dataframe.shape[0]):
            id = dataframe["Image_ID"].values[iFile]
            caption = dataframe["en_caption"].values[iFile]
            image_path = path_dataset + iSub + "/" + id      

            flag=False
            if os.path.isfile(PATH_DATASETS + image_path + ".png"):
                image_path += ".png"
                flag=True
            elif os.path.isfile(PATH_DATASETS + image_path + ".jpg"):
                image_path += ".jpg"
                flag = True
            else:
                print(image_path)

            if flag==True:
                data.append({"image": image_path,
                         "caption": caption})

        df_out = pd.DataFrame(data)
        df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "39_MM_Retinal_dataset.csv")


def MIMICCXR():
    # Only the picture where the first one of each check is AP/PA is extracted
    path_dataset = "mimic-cxr/"

    data_train, data_val, data_test = [], [], []
    data = []
    count_ok = 0
    count_fail = 0
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "mimic-cxr-2.0.0-metadata.csv")
    split_dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "mimic-cxr-2.0.0-split.csv")

    # text_module = []
    # for iFile in os.listdir(PATH_DATASETS + path_dataset + "files/"):
    #     for iiFile in os.listdir(PATH_DATASETS + path_dataset + "files/" + iFile + '/'):
    #         for iiiFile in os.listdir(PATH_DATASETS + path_dataset + "files/" + iFile + '/' + iiFile + '/'):
    #             report_path = PATH_DATASETS + path_dataset + "files/" + iFile + '/' + iiFile + '/' + iiiFile
    #
    #             with open(report_path, "r", encoding="utf-8") as f:
    #                 text = f.read()
    #             module = re.findall(r'(?m)^\s*([A-Z ]+):', text)
    #
    #             for m in module:
    #                 if m.strip() not in text_module:
    #                     text_module.append(m.strip())
    text_module = ['HISTORY', 'FINDINGS', 'EXAMINATION', 'INDICATION', 'TECHNIQUE', 'COMPARISON', 'IMPRESSION',
                   'NOTIFICATION', 'CLINICAL HISTORY', 'EXAM', 'CLINICAL INFORMATION', 'PORTABLE AP CHEST RADIOGRAPH',
                   'WET READ', 'CLINICAL INDICATION', 'COMPARISONS', 'TYPE OF EXAMINATION', 'TWO VIEWS OF THE CHEST',
                   'CHEST TWO VIEWS', 'PA AND LATERAL CHEST RADIOGRAPH', 'FRONTAL AND LATERAL CHEST', 'PATIENT HISTORY',
                   'REASON FOR EXAMINATION', 'FRONTAL AND LATERAL CHEST RADIOGRAPH', 'CONCLUSION',
                   'PA AND LATERAL VIEWS OF THE CHEST', 'REASON FOR EXAM', 'PORTABLE CHEST', 'STUDY', 'REPORT',
                   'PORTABLE FRONTAL CHEST RADIOGRAPH', 'PORTABLE UPRIGHT CHEST RADIOGRAPH', 'SINGLE PORTABLE AP VIEW OF THE CHEST',
                   'ADDENDUM', 'UPRIGHT AP AND LATERAL VIEWS OF THE CHEST', 'SINGLE PORTABLE VIEW OF THE CHEST',
                   'FRONTAL AND LATERAL VIEWS OF THE CHEST', 'FRONTAL AND LATERAL CHEST RADIOGRAPHS', 'FINDINGS AND IMPRESSION',
                   'H', 'FRONTAL CHEST RADIOGRAPH', 'REFERENCE EXAM', 'AP AND LATERAL CHEST RADIOGRAPH', 'AP AND LATERAL CHEST RADIOGRAPHS',
                   'CHEST', 'COMPARISON FILM', 'PA AND LATERAL CHEST RADIOGRAPHS', 'ONE VIEW OF THE CHEST',
                   'SUPINE AP VIEW OF THE CHEST', 'FINSINGS', 'FRONTAL VIEWS OF THE CHEST', 'PA AND LATERAL VIES OF THE CHEST',
                   'RECOMMENDATION', 'SUPINE PORTABLE CHEST RADIOGRAPH', 'COMPARISON EXAM', 'AP AND LATERAL VIEWS OF THE CHEST',
                   'DATE', 'IMORESSION', 'SINGLE PORTABLE FRONTAL VIEW OF THE CHEST', 'NOTE', 'CC', 'PRELIMINARY REPORT',
                   'COMPARSION', 'CHEST AP', 'PORTABLE FRONTAL VIEW OF THE CHEST', 'AP CHEST', 'UPRIGHT AP VIEW OF THE CHEST',
                   'PORTABLE SUPINE RADIOGRAPH OF THE CHEST', 'PORTABLE SUPINE FRONTAL VIEW OF THE CHEST', 'THREE VIEWS OF THE CHEST',
                   'RECOMMENDATIONS', 'COMMENT', 'CHEST RADIOGRAPH', 'SINGLE FRONTAL VIEW OF THE CHEST', 'SINGLE FRONTAL VIEW',
                   'PORTABLE UPRIGHT RADIOGRAPH OF THE CHEST', 'REFERENCE EXAMINATION', 'PORTABLE CHEST RADIOGRAPH',
                   'PORTABLE UPRIGHT AP CHEST RADIOGRAPH', 'PORTABLE AP RADIOGRAPH OF THE CHEST', 'CHEST PORTABLE',
                   'SINGLE FRONTAL CHEST RADIOGRAPH', 'TWO PORTABLE AP VIEWS OF THE CHEST', 'UPRIGHT PA AND LATERAL VIEWS OF THE CHEST',
                   'TWO VIEWS OF THE CHEST WITH CONE DOWNED VIEWS OF THE LEFT RIBS', 'FRONTAL AND LATERAL VIEWS THE CHEST',
                   'FRONTAL UPRIGHT PORTABLE CHEST', 'PA AND LATERAL CHEST', 'PROCEDURE', 'COMMENTS', 'SINGLE AP VIEW OF THE CHEST',
                   'PORTABLE AP CHEST', 'PA AND LATERLCHEST RADIOGRAPH', 'PFI', 'SINGLE PORTABLE CHEST RADIOGRAPH',
                   'SUPINE PORTABLE FRONTAL CHEST RADIOGRAPH', 'TYPE OF EXAM', 'FINDING', 'PORTABLE AP VIEW OF THE CHEST',
                   'AP SUPINE PORTABLE CHEST RADIOGRAPH', 'IMPRRESSION', 'UPRIGHT PORTABLE AP CHEST RADIOGRAPH',
                   'CHEST SINGLE VIEW', 'PRIOR STUDY', 'AP VIEW OF THE CHEST', 'AP PORTABLE UPRIGHT CHEST',
                   'PORTABLE RADIOGRAPH OF THE CHEST', 'FOLLOWS', 'FRONTAL PORTABLE CHEST', 'INTERVAL HISTORY',
                   'PORTABLE SUPINE CHEST RADIOGRAPH', 'FRONTAL VIEW OF THE CHEST', 'WETREAD', 'PORTABLE AP UPRIGHT CHEST RADIOGRAPH',
                   'PORTABLE UPRIGHT AP VIEW OF THE CHEST', 'AP UPRIGHT CHEST RADIOGRAPH', 'PORTABLE UPRIGHT FRONTAL CHEST RADIOGRAPH',
                   'ADDENDUM  IMPRESSION IN', 'CLINIC INDICATION', 'AP AND LATERAL RADIOGRAPHS OF THE CHEST',
                   'PORTABLE UPRIGHT FRONTAL VIEW OF THE CHEST', 'COMPARISON STUDIES', 'UPRIGHT PORTABLE CHEST', 'INDICATIONS',
                   'NOTIFICATIONS', 'TWO VIEWS OF THE LEFT CHEST', 'PORTABLE SUPINE AP VIEW OF THE CHEST',
                   'BEDSIDE UPRIGHT FRONTAL CHEST RADIOGRAPH', 'SUPINE PORTABLE RADIOGRAPH OF THE CHEST',
                   'AP PORTABLE SUPINE CHEST RADIOGRAPH', 'AP PORTABLE CHEST', 'PORTABLE UPRIGHT AP VIEW OF THE ABDOMEN',
                   'PORTABLE FRONTAL CHEST', 'ACCESSION NUMBER', 'SINGLE PORTABLE RADIOGRAPH', 'PA AND LATERAL RADIOGRAPHS OF THE CHEST',
                   'RIGHT RIBS', 'OPINION', 'ADDITIONAL CLINICAL HISTORY PROVIDED', 'SINGLE AP PORTABLE VIEW OF THE CHEST',
                   'CHEST CT TWO VIEWS', '', 'BEDSIDE AP RADIOGRAPH OF THE CHEST', 'SINGLE AP ERECT PORTABLE VIEW OF THE CHEST',
                   'AP UPRIGHT VIEW OF THE CHEST', 'PORTABLE AP FRONTAL VIEW OF THE CHEST', 'PA AND LAT CHEST RADIOGRAPH',
                   'FINIDNGS', 'CHEST AND UPPER ABDOMEN', 'FRONTAL AND LATERAL VIEWS OF CHEST', 'PA AND LATERAL FILMS OF THE CHEST',
                   'SUPINE PORTABLE CHEST RADIOGRAPHS', 'SINGLE FRONTAL PORTABLE VIEW OF THE CHEST', 'IMPRESION',
                   'AP UPRIGHT AND LATERAL CHEST RADIOGRAPHS', 'PORTABLE AP UPRIGHT RADIOGRAPH OF THE CHEST',
                   'SUPINE AP PORTABLE CHEST RADIOGRAPH', 'SUPINE PORTABLE CHEST', 'SINGLE PORTABLE UPRIGHT CHEST RADIOGRAPH',
                   'SINGLE UPRIGHT PORTABLE CHEST RADIOGRAPH', 'SUPINE PORTABLE AP CHEST RADIOGRAPH', 'PA AND LATERAL VIEWS CHEST',
                   'PORTABLE PA CHEST RADIOGRAPH', 'AP PORTABLE CHEST RADIOGRAPH', 'PORTABLE SUPINE FRONTAL CHEST RADIOGRAPH',
                   'FRONTAL CHEST RADIOGRAPHS', 'UPRIGHT FRONTAL AND LATERAL VIEWS OF THE CHEST', 'UPRIGHT AP AND LATERAL CHEST RADIOGRAPH',
                   'PORTABLE SUPINE AP CHEST RADIOGRAPH', 'PORTABLE AP UPRIGHT VIEW OF THE CHEST', 'CHEST RADIOGRAPHS',
                   'CHEST COMPARISON FILM', 'UPRIGHT AP AND LATERAL VIEWS OF CHEST', 'FRONTAL LATERAL CHEST RADIOGRAPH',
                   'FRONTAL SUPINE PORTABLE CHEST', 'FRONTAL PORTABLE UPRIGHT CHEST', 'RIBS', 'PORTABLE AP FRONTAL CHEST RADIOGRAPH',
                   'PORTABLE AP AND LATERAL CHEST RADIOGRAPHS', 'PORTABLE FRONTAL CHEST RADIOGRAPHS', 'TWO PORTABLE FRONTAL RADIOGRAPHS',
                   'CXR', 'TWO AP VIEWS OF THE CHEST', 'MAIN REPORT', 'AP UPRIGHT PORTABLE CHEST RADIOGRAPH',
                   'PA AND LATERAL VIEW OF THE CHEST', 'ONE VIEW OF THE CHEST AND ABDOMEN', 'SUPINE PORTABLE RADIOGRAPH',
                   'IMPRESSON', 'SINGLE AP PORTABLE CHEST RADIOGRAPH', 'FINDGINGS', 'CHEST PA AND LATERAL RADIOGRAPH',
                   'IMPERSSION', 'AP PORTABLE FRONTAL CHEST RADIOGRAPH', 'UPRIGHT FRONTAL CHEST RADIOGRAPH', 'AP CHEST RADIOGRAPH',
                   'TWO VIES OF THE CHEST', 'BONE WINDOWS', 'COMPARISIONS', 'FRONTAL PORTABLE CHEST RADIOGRAPH', 'IMPESSION',
                   'SINGLE SUPINE PORTABLE VIEW OF THE CHEST', 'SUPINE FRONTAL CHEST RADIOGRAPH', 'REASON FOR THE EXAM',
                   'FRONTAL AND LATERAL FRONTAL CHEST RADIOGRAPH', 'FRONTAL RADIOGRAPH OF THE CHEST', 'SINGLE PORTABLE AP CHEST RADIOGRAPH',
                   'RIGHT', 'PLEASE NOTE', 'PORTABLE AP SUPINE CHEST RADIOGRAPH', 'SEMI ERECT PORTABLE CHEST RADIOGRAPH',
                   'COMPARISION', 'COMPARISON STUDY', 'AP VIEWS OF THE CHEST', 'SINGLE AP UPRIGHT CHEST RADIOGRAPH',
                   'PA AND LATERAL RADIOGRAPH OF THE CHEST', 'IMPRESSIONS', 'PORTABLE ERECT FRONTAL CHEST RADIOGRAPH',
                   'COMPARE', 'OF DOSE', 'CHESTAP', 'UPRIGHT AP CHEST RADIOGRAPH', 'UPRIGHT FRONTAL AND LATERAL CHEST RADIOGRAPHS',
                   'CLINCAL HISTORY', 'CT', 'CONLCUSION', 'FRONTAL AND LATERAL VIEWS OF THE CHEST IN EXPIRATION',
                   'OMPARISON', 'TECHNIQUE AND FINDINGS', 'ERROR', 'SUPINE FRONTAL VIEW OF THE CHEST', 'FIDINGS',
                   'CORRECTED IMPRESSION', 'THE IMPRESSION SHOULD READ', 'SINGLE UPRIGHT PORTABLE VIEW OF THE CHEST',
                   'FINDINGS AND IMRPESSION', 'CHEST AP SUPINE', 'ADDENDUM  INDICATION', 'CHEST PA', 'PORTALBLE AP CHEST RADIOGRAPH',
                   'FRONTAL SEMI UPRIGHT PORTABLE CHEST', 'DEDICATED VIEWS OF THE RIGHT RIBS', 'FRONAL AND LATERAL VIEWS OF THE CHEST',
                   'PORTABLE AP UPRIGHT CHEST RADIOGRAPHS', 'AP UPRIGHT AND LATERAL VIEWS OF THE CHEST', 'PORTABLE UPRIGHT CHEST',
                   'SINGLE  PORTABLE FRONTAL VIEW OF THE CHEST', 'FRONTAL AND LATERAL VIEWS CHEST', 'DATE OF EXAM',
                   'REFERENCE FINDINGS', 'FINDIGNS', 'IMPRSSION', 'PA AND AP CHEST RADIOGRAPH', 'UPRIGHT FRONTAL VIEW OF THE CHEST',
                   'AP VIEW AND LATERAL VIEW OF THE CHEST', 'SINGLE AP UPRIGHT PORTABLE CHEST RADIOGRAPH', 'RESIDENT WET READ',
                   'IMPRESSOIN', 'FINDINDGS', 'PORTABLE AP CHEST RADIOGRAPHS', 'ANESTHESIA', 'MEDICATIONS', 'CONTRAST',
                   'FLUOROSCOPY TIME AND DOSE', 'PROCEDURE DETAILS', 'SEMIERECT PORTABLE RADIOGRAPH OF THE CHEST',
                   'PA AND LATERAL VIEWS OF CHEST', 'SEMIERECT AP VIEW OF THE CHEST', 'FRONTAL AND LATERAL UPRIGHT CHEST RADIOGRAPH',
                   'UPRIGHT PORTABLE RADIOGRAPH OF THE CHEST', 'PRIOR EXAM', 'COMPARISON FILMS', 'PA AND LATERAL UPRIGHT CHEST RADIOGRAPHS',
                   'FINDINS', 'RIGHTAND LEFT FRONTAL OBLIQUE VIEWS OF THE CHEST', 'AP SUPINE CHEST RADIOGRAPH',
                   'TWO PORTABLE ERECT VIEWS OF THE CHEST', 'FRONTAL SUPINE PORTABLE VIEW OF THE CHEST', 'COMPARRISON',
                   'BEDSIDE AP UPRIGHT RADIOGRAPH OF THE CHEST', 'SUPINE PORTABLE FRONTAL VIEW OF THE CHEST',
                   'AP UPRIGHT VIEWS OF THE CHEST DURING INSPIRATION AND EXPIRATION', 'SUPINE CHEST', 'PORTABLE SUPINE CHEST',
                   'COMPARISON CHEST', 'CHEST AND PELVIS FILMS', 'FRONTAL AP AND LATERAL CHEST', 'PA AND LAT', 'COMPARISON EXAMS',
                   'PA AND LATERAL CHEST FILMS', 'BEDSIDE FRONTAL CHEST RADIOGRAPH', 'COCLUSION', 'ADDENDUM  CORRECTION',
                   'FRONTAL PORTABLE SUPINE CHEST', 'SINGLE FRONTAL PORTABLE SUPINE VIEW OF THE CHEST', 'TIME', 'AP UPRIGHTPORTABLE CHEST',
                   'CHEST HISTORY', 'RECOMMEDATIONS', 'SINGLE AP CHEST RADIOGRAPH', 'COMPARISON TO PRIOR STUDY',
                   'SEMIUPRIGHT PORTABLE RADIOGRAPH OF THE CHEST', 'CHEST AP PORTABLE', 'PA AND LATERAL RADIOGRAPH',
                   'TECHIQUE', 'SINGLE APRADIOGRAPHS', 'ADDENDUM  HISTORY', 'UPRIGHT PA AND LATERAL VIEWS THE CHEST',
                   'CHEST PORTABLE VIEW', 'RECOMMMENDATIONS', 'PELVIS', 'FINDOINGS', 'AP RADIOGRAPH OF THE CHEST',
                   'FRONTAL LATERAL VIEWS CHEST', 'SINGLE SUPINE AP PORTABLE CHEST RADIOGRAPH', 'NOTFICATIONS',
                   'COMAPRISON', 'SINGLE PA VIEW OF THE CHEST', 'ADDENDUM  COMPARISON', 'REFERENCE  EXAM', 'COR',
                   'OSSEOUS STRUCTURES', 'AP FILM', 'UPRIGHT PORTABLE CHEST RADIOGRAPH', 'FINGDINGS', 'AP FRONTAL AND LATERAL CHEST RADIOGRAPHS',
                   'FRONTAL CHEST RADIOGRAPH WITH THE PATIENT IN SUPINE AND UPRIGHT POSITIONS', 'NCHCT',
                   'TECHNIQUE PA AND LATERAL VIEWS OF THE CHEST', 'SINGLE PORTABLE UPRIGHT VIEW OF THE CHEST',
                   'AP PORTABLE UPRIGHT CHEST RADIOGRAPH', 'PORTABLE UPRIGHT RADIOGRAPH CHEST', 'FRONTAL PORTABLE UPRIGHT RADIOGRAPH',
                   'NDICATION', 'REASON  FOR EXAMINATION', 'MPRESSION', 'TWO VIEWS OF THE THORACIC SPINE', 'FRONTAL AND LATERAL RADIOGRAPHS',
                   'CHEST PA AND LAT RADIOGRAPH', 'REASON FORE EXAM', 'ADDENDUM  IMPRESSION', 'AMENDMENT', 'PORTABLE ERECT RADIOGRAPH',
                   'INDCATION', 'FIMPRESSION', 'SUPINE CHEST RADIOGRAPH', 'SINGLE FRONTAL CHEST RADIOGRAPHS', 'DOUBLE CHEST RADIOGRAPH',
                   'TYPE OF THE EXAMINATION', 'IDICATION', 'AP FRONTAL CHEST RADIOGRAPH', 'PRELIMINARY RESIDENT WET READ',
                   'PORTABLE RADIOGRAPH', 'PORTABLE AP AND LATERAL CHEST RADIOGRAPH', 'FINDNINGS', 'REASON FOR INDICATION',
                   'REASON OF EXAM', 'CHEST SUPINE', 'FRONTALSUPINE CHEST RADIOGRAPH', 'TWO PORTABLE VIEWS OF THE CHEST',
                   'IIMPRESSION', 'SINGLE AP UPRIGHT VIEW OF THE CHEST', 'ABDOMEN', 'CHEST PA AND LATERAL', 'PA LATERAL VIEWS OF THE CHEST', 'FINDNGS']

    target_IMPRESSION = ['IMPRESSION', 'IMPRRESSION', 'IMPRESION', 'IMPRESSON', 'IMPERSSION', 'IMPESSION', 'IMPRESSIONS',
                         'IMPRESSIONS', 'IMPRESSOIN', 'FIMPRESSION','IMORESSION']
    target_FINDINGS = ['FINDINGS', 'FINSINGS', 'FINDING', 'FINIDNGS', 'FINDGINGS', 'FINDIGNS', 'FINDINDGS', 'FINDINS',
                       'FINDOINGS', 'FINGDINGS', 'FINDNINGS', 'FINDNGS', 'FINDINGS AND IMPRESSION', 'FINDINGS AND IMRPESSION']

    for iFile in os.listdir(PATH_DATASETS + path_dataset + "files/"):
        for iiFile in os.listdir(PATH_DATASETS + path_dataset + "files/" + iFile + '/'):
            for iiiFile in os.listdir(PATH_DATASETS + path_dataset + "files/" + iFile + '/' + iiFile + '/'):
                report_path = PATH_DATASETS + path_dataset + "files/" + iFile + '/' + iiFile + '/' + iiiFile
                study_name = iiiFile.split('.')[0]
                with open(report_path, "r", encoding="utf-8") as f:
                    text = f.read()

                findings_match = None
                impression_match = None
                for i in target_FINDINGS:
                    pattern = r"(?si)" + i + r"\s*:\s*(.*?)(?=\n\s*(?:" + "|".join(text_module) + r")\s*:|$)"
                    if re.search(pattern, text) != None:
                        findings_match = re.search(pattern, text)
                        break
                for i in target_IMPRESSION:
                    pattern = r"(?si)" + i + r"\s*:\s*(.*?)(?=\n\s*(?:" + "|".join(text_module) + r")\s*:|$)"
                    if re.search(pattern, text) != None:
                        impression_match = re.search(pattern, text)
                        break

                if findings_match == None and impression_match == None:
                    count_fail += 1
                    continue

                findings = findings_match.group(1).strip().replace("\n", "") if findings_match else ""
                impression = impression_match.group(1).strip().replace("\n", "") if impression_match else ""
                report = impression + " " + findings

                for image in os.listdir(PATH_DATASETS + path_dataset + "image/" + iFile + '/' + iiFile + '/' + study_name + '/'):
                    image_path = path_dataset + "image/" + iFile + '/' + iiFile + '/' + study_name + '/' + image
                    image_name = image.split('.')[0]

                    ViewPosition = dataframe.loc[dataframe["dicom_id"]==image_name, "ViewPosition"].values[0]
                    split = split_dataframe.loc[dataframe["dicom_id"]==image_name, "split"].values[0]
                    if ViewPosition in ["AP", "PA"]:
                        if split == "train":
                            data.append({"image": image_path,
                                         "caption": report,
                                         "Split": "train"})
                        elif split == "validate":
                            data.append({"image": image_path,
                                         "caption": report,
                                         "Split": "val"})
                        elif split == "test":
                            data.append({"image": image_path,
                                         "caption": report,
                                         "Split": "test"})
                        count_ok += 1
                        break

    df_out = pd.DataFrame(data_train)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "mimic-cxr.csv")
    print(len(data_train))
    df_out = pd.DataFrame(data_test)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "mimic-cxr_test.csv")
    print('OK：' + str(count_ok))
    print('fail:' + str(count_fail))


def Chexpert():
    path_dataset = "CheXpert-v1.0/"
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "train.csv")
    labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema',
              'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
              'Fracture', 'Support Devices']

    # Chexpert 5*200
    data_test =[]
    df_test = pd.read_csv(PATH_DATASETS + path_dataset + "chexpert_5x200.csv")
    for i in range(df_test.shape[0]):
        row = df_test.iloc[i]
        image_path = row["Path"]

        cls = ''
        for col in labels:
            if row[col] == 1:
                cls = col
                break
        data_test.append({"image": image_path,
                     "atributes": [],
                     "categories": [cls]})
    df_out = pd.DataFrame(data_test)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Chexpert5*200.csv")
    print(len(data_test))

    data_train =[]
    dataframe["patient_id"] = dataframe["Path"].apply(lambda x: x.split("/")[-3])
    df_test["patient_id"] = df_test["Path"].apply(lambda x: x.split("/")[-3])
    dataframe = dataframe[~dataframe["patient_id"].isin(df_test["patient_id"].tolist())]
    OK = 0
    fail = 0
    for i in range(dataframe.shape[0]):
        row = dataframe.iloc[i]
        if row["Frontal/Lateral"] == "Lateral":
            fail += 1
            continue

        image_path = row["Path"]
        pos = []
        neg = []
        unc = []

        for col in labels:
            if row[col] == 1:
                pos.append(col)
            elif row[col] == 0:
                neg.append(col)
            elif row[col] == -1:
                unc.append(col)

        data_train.append({"image": image_path,
                     "atributes": [],
                     "categories": [pos, neg, unc]})
        OK += 1

    df_out = pd.DataFrame(data_train)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "CheXpert-v1.0.csv")
    print(len(data_train))
    print('OK' + str(OK))
    print('Fail' + str(fail))

    data_valid = []
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "valid.csv")
    for i in range(dataframe.shape[0]):
        row = dataframe.iloc[i]
        if row["Frontal/Lateral"] == "Lateral":
            continue

        image_path = row["Path"]
        pos = []
        neg = []
        unc = []

        for col in labels:
            if row[col] == 1:
                pos.append(col)
            elif row[col] == 0:
                neg.append(col)
            elif row[col] == -1:
                unc.append(col)

        data_valid.append({"image": image_path,
                     "atributes": [],
                     "categories": [pos, neg, unc]})

    df_out = pd.DataFrame(data_valid)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Chexpert_valid.csv")
    print(len(data_valid))


def RSNA_Pneumonia():
    path_dataset = "RSNA_Pneumonia/"
    image_path = "Training/Images/"
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "stage2_train_metadata.csv")
    data = []

    for i in range(dataframe.shape[0]):
        row = dataframe.iloc[i]
        categories = ""
        if row["Target"] == 0:
            categories = "No Finding"
        else:
            categories = "Pneumonia"
        data.append({"image": path_dataset + image_path + row["patientId"] + ".png",
                     "atributes": [],
                     "categories": [categories]})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "RSNA_Pneumonia.csv")
    print(len(data))


def siim_pneumothorax():
    path_dataset = "siim_pneumothorax/train/"
    image_path = "images/256/dicom/"
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "train-rle.csv")
    data = []

    for i in range(dataframe.shape[0]):
        row = dataframe.iloc[i]
        categories = ""
        if row[" EncodedPixels"] == " -1":
            categories = "No Finding"
        else:
            categories = "Pneumothorax"
        data.append({"image": path_dataset + image_path + row["ImageId"] + ".png",
                     "atributes": [],
                     "categories": [categories]})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "siim_pneumothorax.csv")
    print(len(data))


def COVIDx():
    path_dataset = "COVIDx/"
    mapping = {'pneumonia' : "Pneumonia", 'COVID-19' : 'COVID-19', 'normal' : "No Finding"}

    data = []
    with open(PATH_DATASETS + path_dataset + "train_split_v3.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            image_name = parts[1]
            label = parts[2]

        data.append({"image": path_dataset + 'train/' + image_name,
                     "atributes": [],
                     "categories": [mapping[label]]})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "COVIDx_train.csv")
    print(len(data))

    data = []
    with open(PATH_DATASETS + path_dataset + "test_split_v3.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            image_name = parts[1]
            label = parts[2]

        data.append({"image": path_dataset + 'test/' + image_name,
                     "atributes": [],
                     "categories": [mapping[label]]})
    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "COVIDx_test.csv")
    print(len(data))


def openi():
    path_dataset = "openi/"
    data = []
    test_set = []

    for iFile in os.listdir(PATH_DATASETS + path_dataset + "report/"):
        tree = ET.parse(PATH_DATASETS + path_dataset + "report/" + iFile)
        root = tree.getroot()

        findings = ""
        impression = ""

        for abstract_text in root.findall('.//AbstractText'):
            label = abstract_text.get('Label')
            text = abstract_text.text

            if label == 'FINDINGS' and text!=None:
                findings = text
            elif label == 'IMPRESSION' and text!=None:
                impression = text

        report = impression + " " + findings

        parent_image_ids = []
        for parent_image in root.findall('.//parentImage'):
            image_id = parent_image.get('id')
            if image_id:
                parent_image_ids.append(image_id)

        if len(report)>0 and len(parent_image_ids)>0:
            data.append({"image" : path_dataset + "image/" + parent_image_ids[0] + ".png",
                         "caption": report,
                         "Split": ""})

    random.seed(42)
    random.shuffle(data)
    for i in range(0, int(0.15*len(data))):
        data[i]["Split"] = "test"
        test_set.append(data[i])
    for i in range(int(0.15*len(data)), 2*int(0.15*len(data))):
        data[i]["Split"] = "val"
    for i in range(2*int(0.15*len(data)), len(data)):
        data[i]["Split"] = "train"

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "openi.csv")
    print(len(data))
    df_out = pd.DataFrame(test_set)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "openi_test.csv")


def OCT_01_RetinalOCT_C8():
    path_dataset = "OCT01_RetinalOCT_C8/RetinalOCT_Dataset/"
    sub_paths = ["train/", "val/", "test/"]
    labels_dr = {'AMD': "age related macular degeneration", 'CNV': "choroidal neovascularization",
                 "CSR": "central serous retinopathy", "DME": "diabetic macular edema", "DR": "diabetic retinopathy",
                 "DRUSEN":"drusens", "MH":"macular hole", "NORMAL":"normal"}

    data = []
    for sub_path in sub_paths:
        categories = [d for d in os.listdir(PATH_DATASETS + path_dataset + sub_path)]
        for cls in categories:
            images = [f for f in os.listdir(PATH_DATASETS + path_dataset + sub_path + cls + '/')]
            for image in images:
                image_path = path_dataset + sub_path + cls + '/' + image
                categories, atributes = [], []
                categories.append(labels_dr[cls])
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": atributes,
                                 "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT01_RetinalOCT_C8.csv")
    print(f'01_RetinalOCT_C8 has {len(data)} images')


def OCT03_Large_Dataset_of_Labeled_OCT():
    path_dataset = "OCT03_Large_Dataset_of_Labeled_OCT/OCT/"
    sub_paths = ["train/", "test/"]
    labels_dr = {'CNV': "choroidal neovascularization", "DME": "diabetic macular edema", "DRUSEN":"drusens", "NORMAL":"normal"}

    data = []
    for sub_path in sub_paths:
        categories = [d for d in os.listdir(PATH_DATASETS + path_dataset + sub_path)]
        for cls in categories:
            images = [f for f in os.listdir(PATH_DATASETS + path_dataset + sub_path + cls + '/')]
            for image in images:
                image_path = path_dataset + sub_path + cls + '/' + image
                categories, atributes = [], []
                categories.append(labels_dr[cls])
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": atributes,
                                 "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT03_Large_Dataset_of_Labeled_OCT.csv")
    print(f'OCT03_Large_Dataset_of_Labeled_OCT has {len(data)} images')


def OCT04_GAMMA1():
    path_dataset = "OCT04_GAMMA1/Glaucoma_grading/training/multi-modality_images/"
    dataframe = pd.read_excel(PATH_DATASETS + "OCT04_GAMMA1/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx")
    labels_dr = {'non': "normal", "early": "glaucoma", "mid_advanced":"glaucoma"}

    data = []
    for i, item in dataframe.iterrows():
        categories, atributes = [], []
        if item['non'] == 1:
            categories.append(labels_dr['non'])
        elif item['early'] == 1:
            categories.append(labels_dr['early'])
        elif item['mid_advanced'] == 1:
            categories.append(labels_dr['mid_advanced'])

        if item['data'] < 10:
            ImgName = '000' + str(item['data'])
        elif item['data'] < 100:
            ImgName = '00' + str(item['data'])
        elif item['data'] >= 100:
            ImgName = '0' + str(item['data'])

        for f in os.listdir(PATH_DATASETS + path_dataset + ImgName + '/' + ImgName + '/'):
            if int(f.split('_')[0]) < 115 or int(f.split('_')[0]) > 145:
                continue
            image_path = path_dataset + ImgName + '/' + ImgName + '/' + f
            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT04_GAMMA1.csv")
    print(f'OCT04_GAMMA1 has {len(data)} images')


def OCT05_OCTID():
    path_dataset = "OCT05_OCTID/"
    labels_dr = {'Age-related-Macular-Degeneration-Retinal-OCT-images': "age related macular degeneration",
                 "Central-serous-retinopathy-retinal-OCT-images": "central serous retinopathy",
                 "Diabetic-Retinopathy-Retinal-OCT-Images":"diabetic retinopathy",
                 "Macular-Hole-Retinal-OCT-images":"macular hole", "Normal-Retinal-OCT-images":"normal"}

    data = []
    categories = [d for d in os.listdir(PATH_DATASETS + path_dataset) if d in labels_dr.keys()]   # 所有类别
    for cls in categories:
        images = [f for f in os.listdir(PATH_DATASETS + path_dataset + cls + '/') if f.endswith('jpeg')]
        for image in images:
            image_path = path_dataset + cls + '/' + image
            categories, atributes = [], []
            categories.append(labels_dr[cls])
            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "OCT05_OCTID.csv")
    print(f'OCT05_OCTID has {len(data)} images')


def OCT06_STAGE1():
    path_dataset = "OCT06_STAGE1/"
    sub_paths = ["STAGE_training/", "STAGE_validation/"]
    sub_image_paths = ["training_images/", "validation_images/"]
    dataframe_paths = ["data_info_training.xlsx", "data_info_validation.xlsx"]
    labels_dr = {'normal': "normal", "early": "glaucoma", "advanced":"glaucoma", "intermediate":"glaucoma"}

    data = []
    for sub_path, sub_image_path, dataframe_path in zip(sub_paths, sub_image_paths, dataframe_paths):
        dataframe = pd.read_excel(PATH_DATASETS + path_dataset + sub_path + dataframe_path)
        for i, item in dataframe.iterrows():
            if item['ID'] < 10:
                ImgName = '000' + str(item['ID'])
            elif item['ID'] < 100:
                ImgName = '00' + str(item['ID'])
            elif item['ID'] >= 100:
                ImgName = '0' + str(item['ID'])
            categories, atributes = [], []
            categories.append(labels_dr[item['Glaucoma_stage']])

            for f in os.listdir(PATH_DATASETS + path_dataset + sub_path + sub_image_path +  ImgName + '/'):
                image_path = path_dataset + sub_path + sub_image_path +  ImgName + '/' + f
                if os.path.isfile(PATH_DATASETS + image_path) and f.endswith('.jpg'):
                    if int(f.split('_')[0]) < 115 or int(f.split('_')[0]) > 145:
                        continue
                    data.append({"image": image_path,
                                 "atributes": atributes,
                                 "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT06_STAGE1.csv")
    print(f'OCT06_STAGE1 has {len(data)} images')


def OCT07_STAGE2():
    path_dataset = "OCT07_STAGE2/"
    sub_paths = ["train/", "valid/"]
    sub_image_paths = ["training_images/", "validation_images/"]
    dataframe_paths = ["data_info_training.xlsx", "data_info_val.xlsx"]
    labels_dr = {'normal': "normal", "early": "glaucoma", "advanced":"glaucoma", "intermediate":"glaucoma"}

    data = []
    for sub_path, sub_image_path, dataframe_path in zip(sub_paths, sub_image_paths, dataframe_paths):
        dataframe = pd.read_excel(PATH_DATASETS + path_dataset + sub_path + dataframe_path)
        for i, item in dataframe.iterrows():
            categories, atributes = [], []
            categories.append(labels_dr[item['Glaucoma_stage']])
            if item['ID'] < 10:
                ImgName = '000' + str(item['ID'])
            elif item['ID'] < 100:
                ImgName = '00' + str(item['ID'])
            elif item['ID'] >= 100:
                ImgName = '0' + str(item['ID'])

            for f in os.listdir(PATH_DATASETS + path_dataset + sub_path + sub_image_path +  ImgName + '/'):
                image_path = path_dataset + sub_path + sub_image_path +  ImgName + '/' + f

                if os.path.isfile(PATH_DATASETS + image_path) and '_' in f:
                    if int(f.split('_')[0]) < 115 or int(f.split('_')[0]) > 145:
                        continue
                    data.append({"image": image_path,
                                 "atributes": atributes,
                                 "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT07_STAGE2.csv")
    print(f'OCT07_STAGE2 has {len(data)} images')


def OCT08_glaucoma_detection():
    path_dataset = "OCT08_glaucoma_detection/"
    labels_dr = {'Normal': "normal", "POAG": "glaucoma"}

    data = []
    for f in os.listdir(PATH_DATASETS + path_dataset):
        image_path = path_dataset + f
        categories, atributes = [], []
        if f.startswith('Normal'):
            categories.append(labels_dr["Normal"])
        elif f.startswith('POAG'):
            categories.append(labels_dr["POAG"])

        os.makedirs(PATH_DATASETS + image_path.replace('.npy', ''))
        data_img = np.load(PATH_DATASETS + image_path)
        for i in range(27,38):
            plt.imsave(PATH_DATASETS + image_path.replace('.npy', '') + '/' + str(i+1) + '.png', data_img[i], cmap = 'gray')

            if os.path.isfile(PATH_DATASETS + image_path.replace('.npy', '') + '/' + str(i+1) + '.png'):
                data.append({"image": image_path.replace('.npy', '') + '/' + str(i+1) + '.png',
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT08_glaucoma_detection.csv")
    print(f'OCT08_glaucoma_detection has {len(data)} images')


def OCT09_GOALS():
    path_dataset = "OCT09_GOALS/GOALS2022-Train/Train/Image/"
    labels_dr = {'normal': "normal", "GC": "Glaucoma"}

    data = []
    dataframe = pd.read_excel(PATH_DATASETS + 'OCT09_GOALS/GOALS2022-Train/Train/Train_GC_GT.xlsx')
    for i, item in dataframe.iterrows():
        if item['ImgName'] < 10:
            ImgName = '000' + str(item['ImgName'])
        elif item['ImgName'] < 100:
            ImgName = '00' + str(item['ImgName'])
        elif item['ImgName'] >= 100:
            ImgName = '0' + str(item['ImgName'])
        image_path = path_dataset + ImgName + '.png'
        categories, atributes = [], []
        if item['GC_Label'] == 1:
            categories.append(labels_dr['GC'])
        elif item['GC_Label'] == 0:
            categories.append(labels_dr['normal'])
        if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT09_GOALS.csv")
    print(f'OCT09_GOALS has {len(data)} images')


def OCT10_OCTDL():
    path_dataset = "OCT10_OCTDL/OCTDL/"
    labels_dr = {'AMD': "age related macular degeneration", "DME": "diabetic macular edema", "ERM": "epiretinal membrane",
                 "NO":"normal", "RAO":"retinal artery occlusion", "RVO":"retinal vein occlusion",
                 "VID":"Vitreomacular Interface Disease"}

    data = []
    categories = [d for d in os.listdir(PATH_DATASETS + path_dataset)]
    for cls in categories:
        images = [f for f in os.listdir(PATH_DATASETS + path_dataset + cls + '/')]
        for image in images:
            image_path = path_dataset + cls + '/' + image
            categories, atributes = [], []
            categories.append(labels_dr[cls])
            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "OCT10_OCTDL.csv")
    print(f'OCT10_OCTDL has {len(data)} images')


def OCT11_OIMHS():
    path_dataset = "OCT11_OIMHS/Images/"
    labels_dr = {'1': "macular hole stage1", "2" : "macular hole stage2", "3" : "macular hole stage3", "4" : "macular hole stage4"}

    data = []
    dataframe = pd.read_excel(PATH_DATASETS + 'OCT11_OIMHS/Demographics of the participants.xlsx')
    for i, item in dataframe.iterrows():
        ID = item["Eye ID"]
        categories, atributes = [], []
        categories.append(labels_dr[str(item["Stage"])])

        image_list = [f for f in os.listdir(PATH_DATASETS + path_dataset + str(ID) + '/') if not "crop" in f]
        num_of_img = len(image_list)
        for j in range(num_of_img//2-5, num_of_img//2+6):
            image_path = PATH_DATASETS + path_dataset + str(ID) + '/' + str(j+1) + '.png'
            image = Image.open(image_path)
            cropped_img = image.crop((0, 0, 512, 512))
            image_cropped_path = image_path.replace(str(j+1) + '.', str(j+1) + '_crop.')
            cropped_img.save(PATH_DATASETS + image_cropped_path)

            if os.path.isfile(PATH_DATASETS + image_cropped_path):
                    data.append({"image": image_cropped_path,
                                 "atributes": atributes,
                                 "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT11_OIMHS.csv")
    print(f'OCT11_OIMHS has {len(data)} images')


def OCT12_OCTA_500():
    path_dataset = "OCT12_OCTA_500/"
    sub_paths = ["OCTA_3mm/", "OCTA_6mm/"]
    labels_dr = {'NORMAL': "normal", 'AMD': "age related macular degeneration", 'CNV': "choroidal neovascularization",
                 "DR": "diabetic retinopathy", "CSC" : "central serous retinopathy", "RVO" : "retinal vein occlusion", }

    data = []
    for sub_path in sub_paths:
        dataframe = pd.read_excel(PATH_DATASETS + path_dataset + sub_path + "Text labels.xlsx")
        for i, item in dataframe.iterrows():
            categories, atributes = [], []
            if item['Disease'] == 'OTHERS':
                continue
            categories.append(labels_dr[item['Disease']])

            for f in os.listdir(PATH_DATASETS + path_dataset + sub_path + 'OCT/' + str(item['ID']) + '/'):
                image_path = path_dataset + sub_path + 'OCT/' + str(item['ID']) + '/' + f
                if (sub_path == "OCTA_3mm/") and (int(f.split('.')[0]) < 130 or int(f.split('.')[0]) > 160):
                    continue
                if (sub_path == "OCTA_6mm/") and (int(f.split('.')[0]) < 180 or int(f.split('.')[0]) > 220):
                    continue
                if os.path.isfile(PATH_DATASETS + image_path):
                        data.append({"image": image_path,
                                     "atributes": atributes,
                                     "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT12_OCTA_500.csv")
    print(f'OCT12_OCTA_500 has {len(data)} images')


def OCT14_DUKE_DME():
    path_dataset = "OCT14_DUKE_DME/"
    labels_dr = {'AMD': "age related macular degeneration", "DME": "diabetic macular edema", "NORMAL":"normal"}

    data = []
    categories = [d for d in os.listdir(PATH_DATASETS + path_dataset)]
    for cls in categories:
        images = [f for f in os.listdir(PATH_DATASETS + path_dataset + cls + '/TIFFs/8bitTIFFs/')]
        for image in images:
            image_path = path_dataset + cls + '/TIFFs/8bitTIFFs/' + image
            if int(image.split('.')[0]) < 20 or int(image.split('.')[0]) > 30:
                continue

            categories, atributes = [], []
            if cls.startswith('AMD'):
                categories.append(labels_dr["AMD"])
            elif cls.startswith('DME'):
                categories.append(labels_dr["DME"])
            elif cls.startswith("NORMAL"):
                categories.append(labels_dr["NORMAL"])
            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT14_DUKE_DME.csv")
    print(f'OCT14_DUKE_DME has {len(data)} images')


def OCT16_BIOMISA_Retinal_Image_Database_for_Macular_Disorders():
    path_dataset = "OCT16_BIOMISA_Retinal_Image_Database_for_Macular_Disorders/Macula/"
    labels_dr = {'AMD': "age related macular degeneration", "CSR": "central serous retinopathy"}

    data = []
    categories = [d for d in os.listdir(PATH_DATASETS + path_dataset)]
    for cls in categories:
        patients = [p for p in os.listdir(PATH_DATASETS + path_dataset + cls + '/')]
        for patient in patients:
            categories, atributes = [], []
            categories.append(labels_dr[cls])
            if cls == 'CSR' and os.path.isdir(PATH_DATASETS + path_dataset + cls + '/' + patient + '/Left Eye/'):
                files = [f for f in os.listdir(PATH_DATASETS + path_dataset + cls + '/' + patient + '/Left Eye/') if
                         'B-scan' in f]
                image_path1 = path_dataset + cls + '/' + patient + '/Left Eye/' + files[0]
            else:
                image_path1 = path_dataset + cls + '/' + patient + '/Left Eye/original.jpg'
            if os.path.isfile(PATH_DATASETS + image_path1):
                data.append({"image": image_path1,
                             "atributes": atributes,
                             "categories": categories})

            if cls == 'CSR' and os.path.isdir(PATH_DATASETS + path_dataset + cls + '/' + patient + '/Right Eye/'):
                files = [f for f in os.listdir(PATH_DATASETS + path_dataset + cls + '/' + patient + '/Right Eye/') if 'B-scan' in f]
                image_path2 = path_dataset + cls + '/' + patient + '/Right Eye/' + files[0]
            else:
                image_path2 = path_dataset + cls + '/' + patient + '/Right Eye/original.jpg'
            if os.path.isfile(PATH_DATASETS + image_path2):
                data.append({"image": image_path2,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "OCT16_BIOMISA_Retinal_Image_Database_for_Macular_Disorders.csv")
    print(f'OCT16_BIOMISA_Retinal_Image_Database_for_Macular_Disorders has {len(data)} images')


MIMICCXR()