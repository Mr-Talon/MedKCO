import os.path

import pandas as pd
import numpy as np
from PIL import Image
import torch
import random
import pickle

from medkco.modeling.model import MedKCOModel
from medkco.modeling.misc import set_seeds
from medkco.modeling.dictionary import definitions, definitions_OCT, definitions_CXR

modality = 'CFP'    # TODO CFP OCT CXR
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MM_OUT = True       # Calculate multimodal data separately

if modality=="CFP":
    dataset_list = ["01_EYEPACS", "03_IDRID", "04_RFMid", "06_DEN", "07_LAG", "08_ODIR", "09_PAPILA", "10_PARAGUAY",
                    "11_STARE", "12_ARIA","14_AGAR300", "15_APTOS", "16_FUND-OCT", "17_DiaRetDB1", "18_DRIONS-DB",
                    "19_Drishti-GS1", "20_E-ophta", "21_G1020", "23_HRF", "24_ORIGA", "26_ROC", "27_BRSET", "28_OIA-DDR",
                    "29_AIROGS", "30_SUSTech-SYSU", "31_JICHI", "32_CHAKSU", "33_DR1-2", "34_Cataract", "35_ScarDat",
                    "39_MM_Retinal_dataset"]

    label_list = ["hard exudates", "soft exudates", "microaneurysms", "haemorrhages", "media haze", "drusens", "tessellation",
                  "laser scar", "optic disc cupping", "tortuous vessels", "asteroid hyalosis", "optic disc pallor", "exudates",
                  "cotton wool spots", "colobomas", "preretinal haemorrhage", "myelinated nerve fibers", "tilted disc",
                  "vitreous haemorrhage", "large optic cup", "optic atrophy", "fibrosis", "silicon oil", "scar", "nevus",
                  "red small dots", "no diabetic retinopathy", "mild diabetic retinopathy", "moderate diabetic retinopathy",
                  "severe diabetic retinopathy", "proliferative diabetic retinopathy", "age-related macular degeneration",
                  "pathologic myopia", "branch retinal vein occlusion", "epiretinal membrane", "macular scar",
                  "central retinal vein occlusion", "optic disc edema", "shunt", "retinal traction", "retinitis",
                  "retinal pigment epithelium changes", "retinitis pigmentosa", "haemorrhagic retinopathy",
                  "central retinal artery occlusion", "post traumatic choroidal rupture", "choroidal folds", "vasculitis",
                  "branch retinal artery occlusion", "plaque", "collaterals", "maculopathy", "severe hypertensive retinopathy",
                  "disc swelling and elevation", "dragged disk", "congenital disk abnormality",
                  "peripheral retinal degeneration and break", "yellow-white spots flecks",
                  "no proliferative diabetic retinopathy", "hypertensive retinopathy", "geographical age-related macular degeneration",
                  "abnormal optic disc", "abnormal vessels", "abnormal macula", "macular edema", "increased cup disc",
                  "a disease" "intraretinal microvascular abnormalities", "retina detachment", "normal", "diabetic macular edema",
                  "no referable diabetic macular edema", "non clinically significant diabetic macular edema", "central serous retinopathy",
                  "anterior ischemic optic neuropathy", "parafoveal telangiectasia", "chorioretinitis", "macular hole",
                  "optic disc pit maculopathy", "haemorrhagic pigment epithelial detachment", "Vogt-Koyanagi syndrome",
                  "glaucoma", "Bietti crystalline dystrophy", "neoplasm", "no glaucoma", "neovascular age-related macular degeneration",
                  "cataract", "no cataract", "macroaneurysm", "cystoid macular edema", "acute central serous retinopathy",
                  "chronic central serous retinopathy", "neovascularisation"]

    data_root_path = "../Datasets/FUNDUS/"
    dataframes_path = "./local_data/dataframes/pretrainingCFP/"
    model_path = ""     # model for extracting features
    output_path = "./local_data/dataframes/xxx/"
    feature_path = './local_data/dataframes/CFP.pkl'

elif modality=="OCT":
    dataset_list = ["OCT01_RetinalOCT_C8", "OCT03_Large_Dataset_of_Labeled_OCT", "OCT04_GAMMA1", "OCT06_STAGE1", "OCT07_STAGE2",
                    "OCT08_glaucoma_detection", "OCT09_GOALS", "OCT11_OIMHS", "OCT12_OCTA_500", "OCT14_DUKE_DME",
                    "OCT16_BIOMISA_Retinal_Image_Database_for_Macular_Disorders", "OCT17_MM_Retinal_OCT"]

    label_list = ["age related macular degeneration", "drusens", "choroidal neovascularization", "central serous retinopathy",
                  "diabetic retinopathy", "diabetic macular edema", "retinal artery occlusion", "retinal vein occlusion",
                  "vitreomacular Interface Disease", "macular hole", "epiretinal membrane", "glaucoma", "normal"]

    data_root_path = "../Datasets/FUNDUS/"
    dataframes_path = "./local_data/dataframes/pretrainingOCT/"
    model_path = ""
    output_path = "./local_data/dataframes/xxx/"
    feature_path = './local_data/dataframes/OCT.pkl'

elif modality=="CXR":
    dataset_list = ['CheXpert-v1.0', 'mimic-cxr']

    label_list = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity", "Edema",
                  "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
                  "Support Devices", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
                  "healthy"]

    data_root_path = "../Datasets/CXR/"
    dataframes_path = "./local_data/dataframes/pretrainingCXR/"
    model_path = ""
    output_path = "./local_data/dataframes/xxx/"
    feature_path = './local_data/dataframes/CXR.pkl'


def main():
    seed = 42
    set_seeds(seed, use_cuda=True)

    caption = ''
    if modality=='CFP':
        caption = 'A fundus photograph of [CLS]'
    elif modality == 'OCT':
        caption = 'An Optical coherence tomography(OCT) photograph of [CLS]'
    elif modality == 'CXR':
        caption = 'A chest x-ray photograph of [CLS]'
    model = MedKCOModel(from_checkpoint=True, weights_path=model_path, caption=caption, bert_type='./Bio_ClinicalBERT')

    # Extract text features from all categories
    text_input_ids, text_attention_mask = model.preprocess_text(label_list)
    with torch.no_grad():
        label_text_embeds = model.text_model(text_input_ids, text_attention_mask)

    cls_feature_list = {}       # (image_path, img_feature, text_feature)
    cls_feature_center = {}     # (img_feature_avg, text_feature_avg)
    MM_cls_feature_center = {}
    cls_feature_distance = {}   # (image_path, img_distance, text_distance)
    MM_cls_feature_distance = {}
    cls_feature_r = {}
    MM_cls_feature_r = {}

    if os.path.exists(feature_path):
        with open(feature_path, 'rb') as f:
            cls_feature_list = pickle.load(f)
    else:
        print("Step 1: compute feature...")
        for iDataset in dataset_list:
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
                data_i["image_path"] = data_root_path + data_i["image"]

                # image
                img = np.array(Image.open(data_i["image_path"]), dtype=float)
                if np.max(img) > 1:
                    img /= 255
                if len(img.shape) > 2:
                    img = np.transpose(img, (2, 0, 1))
                else:
                    img = np.expand_dims(img, 0)
                img = np.expand_dims(img, 0)
                img = torch.tensor(img).to(torch.float32).to(device)

                # text
                atr_sample = random.sample(data_i['atributes'], 1)[0] if len(data_i['atributes']) > 0 else ""
                cat_sample = None
                if "CheXpert-v1.0" in data_i["image"]:
                    cat_sample = data_i['categories']
                    data_i["sel_category"] = str(cat_sample)
                else:
                    cat_sample = random.sample(data_i['categories'], 1)[0] if len(data_i['categories']) > 0 else ""
                    data_i["sel_category"] = cat_sample

                if "CheXpert-v1.0" in data_i["image"]:
                    report = ""
                    positive, negative, uncertain = cat_sample
                    for pos in positive:
                        cand = definitions_CXR[pos]["pos"]
                        sentence = random.choice(cand)
                        if len(sentence) > 0:
                            report += " " + sentence
                    for neg in negative:
                        cand = definitions_CXR[neg]["neg"]
                        sentence = random.choice(cand)
                        if len(sentence) > 0:
                            report += " " + sentence
                    for unc in uncertain:
                        cand = definitions_CXR[unc]["unc"]
                        sentence = random.choice(cand)
                        if len(sentence) > 0:
                            report += " " + sentence
                    data_i["report"] = [report]
                elif 'OCT17_MM_Retinal_OCT' in data_i["image"] or '39_MM_Retinal_dataset' in data_i["image"] or 'mimic-cxr' in data_i["image"] or 'openi' in data_i["image"]:
                    data_i["report"] = [cat_sample]
                else:
                    caption = ''
                    if modality == 'CFP':
                        caption = "A [ATR] fundus photograph of [CLS]"
                    elif modality == 'OCT':
                        caption = "A [ATR] Optical coherence tomography(OCT) photograph of [CLS]"
                    data_i["report"] = [caption.replace("[ATR]", atr_sample).replace("[CLS]", cat_sample).replace("  "," ")]

                if data_i["image"].split("/")[0] not in ["06_EYENET", "11_STARE", "08_ODIR-5K", "31_JICHI", "39_MM_Retinal_dataset",
                                                              "OCT17_MM_Retinal_OCT", "CheXpert-v1.0", "mimic-cxr", "openi"]:
                    if modality == 'OCT' and data_i["sel_category"] in list(definitions_OCT.keys()):
                        prompts = [data_i["sel_category"]] + definitions_OCT[data_i["sel_category"]]
                        new_cat = random.sample(prompts, 1)[0]
                        data_i["report"][0] = data_i["report"][0].replace(data_i["sel_category"], new_cat)
                    elif modality == 'CFP' and data_i["sel_category"] in list(definitions.keys()):
                        prompts = [data_i["sel_category"]] + definitions[data_i["sel_category"]]
                        new_cat = random.sample(prompts, 1)[0]
                        data_i["report"][0] = data_i["report"][0].replace(data_i["sel_category"], new_cat)
                report = data_i["report"]

                # feature extraction
                text_input_ids, text_attention_mask = model.preprocess_report(report)
                with torch.no_grad():
                    img_embeds = model.vision_model(img)
                    text_embeds = model.text_model(text_input_ids, text_attention_mask)

                # cluster caption
                if data_i["sel_category"] not in label_list:
                    logits = model.compute_logits(text_embeds, label_text_embeds).t()
                    data_i["sel_category"] = label_list[torch.argmax(logits)]

                if data_i["sel_category"] not in cls_feature_list:
                    cls_feature_list[data_i["sel_category"]] = [(data_i["image"], img_embeds.to('cpu').numpy(), text_embeds.to('cpu').numpy())]
                else:
                    cls_feature_list[data_i["sel_category"]].append((data_i["image"], img_embeds.to('cpu').numpy(), text_embeds.to('cpu').numpy()))
        with open(feature_path, 'wb') as f:
            print("save all feature...")
            pickle.dump(cls_feature_list, f)


    print("Step 2: compute feature center...")
    for key in cls_feature_list:
        img_sum, text_sum, count = 0, 0, 0
        MM_img_sum, MM_text_sum, MM_count = 0, 0, 0
        for sample in cls_feature_list[key]:
            if (sample[0].startswith("39_") or sample[0].startswith("OCT17") or sample[0].startswith("mimic") or sample[0].startswith("openi"))  and MM_OUT:
                MM_img_sum = MM_img_sum + sample[1]
                MM_text_sum = MM_text_sum + sample[2]
                MM_count += 1
            else:
                img_sum = img_sum + sample[1]
                text_sum = text_sum + sample[2]
                count += 1
        if count>0:
            cls_feature_center[key] = (img_sum / count, text_sum / count)
        if MM_OUT and MM_count>0:
            MM_cls_feature_center[key] = (MM_img_sum / MM_count, MM_text_sum / MM_count)


    print("Step 3: compute feature distance...")
    def distance(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    for key in cls_feature_list:
        if key in cls_feature_center.keys():
            img_center = cls_feature_center[key][0]
            text_center = cls_feature_center[key][1]
        if MM_OUT and key in MM_cls_feature_center.keys():
            MM_img_center = MM_cls_feature_center[key][0]
            MM_text_center = MM_cls_feature_center[key][1]

        for sample in cls_feature_list[key]:
            if (sample[0].startswith("39_") or sample[0].startswith("OCT17") or sample[0].startswith("mimic") or sample[0].startswith("openi")) and MM_OUT:
                if key not in MM_cls_feature_distance:
                    MM_cls_feature_distance[key] = [(sample[0], distance(MM_img_center, sample[1]), distance(MM_text_center, sample[2]))]
                else:
                    MM_cls_feature_distance[key].append((sample[0], distance(MM_img_center, sample[1]), distance(MM_text_center, sample[2])))
            else:
                if key not in cls_feature_distance:
                    cls_feature_distance[key] = [(sample[0], distance(img_center, sample[1]), distance(text_center, sample[2]))]
                else:
                    cls_feature_distance[key].append((sample[0], distance(img_center, sample[1]), distance(text_center, sample[2])))


    print("Step 4: compute feature r...")
    for key in cls_feature_distance:
        img_r, text_r = 0, 0
        for sample in cls_feature_distance[key]:
            if sample[1] > img_r:
                img_r = sample[1]
            if sample[2] > text_r:
                text_r = sample[2]
        cls_feature_r[key] = (img_r, text_r)

    if MM_OUT:
        for key in MM_cls_feature_distance:
            MM_img_r, MM_text_r = 0, 0
            for sample in MM_cls_feature_distance[key]:
                if sample[1] > MM_img_r:
                    MM_img_r = sample[1]
                if sample[2] > MM_text_r:
                    MM_text_r = sample[2]
            MM_cls_feature_r[key] = (MM_img_r, MM_text_r)
        print(MM_cls_feature_r)


    print("Step 5: save result...")
    num_of_stage = 1
    num_of_stage_MM = 2                 # num of stages in curriculum 2
    dataframe_list = {}
    for iDataset in dataset_list:
        dataframe = pd.read_csv(dataframes_path + iDataset + ".csv")
        dataframe['stage'] = None
        dataframe['img_norm_r'] = None
        dataframe_list[iDataset] = dataframe

    for key in cls_feature_distance:
        print("Step 5: Current cls: " + key)
        for i, sample in enumerate(cls_feature_distance[key]):
            dataset = sample[0].split('/')[0].lower()
            dataset_name = ""
            for j, d in enumerate(dataset_list):
                if d.lower() in dataset:
                    dataset_name = dataset_list[j]
                    break
            for s in range(num_of_stage):
                if (sample[1] > cls_feature_r[key][0] / num_of_stage * s and
                    sample[1] <= cls_feature_r[key][0] / num_of_stage * (s+1)):
                    dataframe_list[dataset_name].loc[dataframe_list[dataset_name]['image'] == sample[0], 'stage'] = num_of_stage - s
                    break
            if cls_feature_r[key][0] != 0:
                dataframe_list[dataset_name].loc[dataframe_list[dataset_name]['image'] == sample[0], 'img_norm_r'] = sample[1] / cls_feature_r[key][0]
            else:
                dataframe_list[dataset_name].loc[dataframe_list[dataset_name]['image'] == sample[0], 'img_norm_r'] = 0

    if MM_OUT:
        for key in MM_cls_feature_distance:
            print("Step 5: Current cls: " + key)
            for i, sample in enumerate(MM_cls_feature_distance[key]):
                dataset = sample[0].split('/')[0].lower()
                dataset_name = ""
                for j, d in enumerate(dataset_list):
                    if d.lower() in dataset:
                        dataset_name = dataset_list[j]
                        break

                # Divide the stages according to the distance of image feature from far to near
                for s in range(num_of_stage_MM):
                    if (sample[1] > MM_cls_feature_r[key][0] / num_of_stage_MM * s and
                        sample[1] <= MM_cls_feature_r[key][0] / num_of_stage_MM * (s+1)):
                        dataframe_list[dataset_name].loc[dataframe_list[dataset_name]['image'] == sample[0], 'stage'] = num_of_stage_MM - s
                        break

                if MM_cls_feature_r[key][0] != 0:
                    dataframe_list[dataset_name].loc[dataframe_list[dataset_name]['image'] == sample[0], 'img_norm_r'] = sample[1] / MM_cls_feature_r[key][0]
                else:
                    dataframe_list[dataset_name].loc[dataframe_list[dataset_name]['image'] == sample[0], 'img_norm_r'] = 0
    for key in dataframe_list:
        dataframe_list[key].to_csv(output_path + key + '.csv')


if __name__ == "__main__":
    main()