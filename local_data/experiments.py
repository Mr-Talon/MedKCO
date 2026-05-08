from local_data.constants import *

def get_experiment_setting(experiment):
    if experiment == "25_REFUGE":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "25_REFUGE.csv",
                   "task": "classification",
                   "targets": {"no glaucoma": 0, "glaucoma": 1}}

    elif experiment == "13_FIVES":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "13_FIVES.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "age related macular degeneration": 1, "diabetic retinopathy": 2,
                               "glaucoma": 3}}

    elif experiment == "08_ODIR200x3":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "08_ODIR200x3.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "pathologic myopia": 1, "cataract": 2}}

    elif experiment == 'OCT05_OCTID':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "OCT05_OCTID.csv",
                   "task": "classification",
                   "targets": {"central serous retinopathy": 0, "age related macular degeneration": 1, "macular hole": 2,
                               "diabetic retinopathy": 3, "normal": 4}}

    elif experiment == 'OCT10_OCTDL':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "OCT10_OCTDL.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "age related macular degeneration": 1, "diabetic macular edema": 2,
                               "retinal artery occlusion": 3, "Vitreomacular Interface Disease": 4,
                               "epiretinal membrane": 5, "retinal vein occlusion": 6 }}

    elif experiment == 'Chexpert5*200':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "Chexpert5*200.csv",
                   "task": "classification",
                   "targets": {"Atelectasis": 0, "Cardiomegaly": 1, "Consolidation": 2,
                               "Edema": 3, "Pleural Effusion": 4}}

    elif experiment == 'RSNA_Pneumonia':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "RSNA_Pneumonia.csv",
                   "task": "classification",
                   "targets": {"Pneumonia": 0, "No Finding": 1}}

    elif experiment == 'siim_pneumothorax':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "siim_pneumothorax.csv",
                   "task": "classification",
                   "targets": {"Pneumothorax": 0, "No Finding": 1}}

    elif experiment == 'COVIDx':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "COVIDx_train.csv",
                   "task": "classification",
                   "targets": {"Pneumonia": 0, "No Finding": 1, "COVID-19": 2}}

    elif experiment == 'openi':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "openi.csv",
                   "task": "Retrieval",
                   "targets": {}}

    elif experiment == 'mimic-cxr':
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "mimic-cxr_test.csv",
                   "task": "Retrieval",
                   "targets": {}}

    else:
        setting = None
        print("Experiment not prepared...")

    return setting