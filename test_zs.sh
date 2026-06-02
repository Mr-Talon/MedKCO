export CUDA_VISIBLE_DEVICES=0

# CFP
CFP_path="./results/pretraining/MedKCO_CFP.pth"
python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment 08_ODIR200x3 --method zero_shot --weights_path $CFP_path --project_features True --modality CFP
python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment 13_FIVES --method zero_shot --weights_path $CFP_path --project_features True --modality CFP
python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment 25_REFUGE --method zero_shot --weights_path $CFP_path --project_features True --modality CFP

# OCT
OCT_path="./results/pretraining/MedKCO_OCT.pth"
python main_transferability.py --data_root_path $Data_Root_Path --shots_train 0 --shots_test 100% --folds 1 --experiment OCT05_OCTID --method zero_shot --weights_path $OCT_path --project_features True --modality OCT
python main_transferability.py --data_root_path $Data_Root_Path --shots_train 0 --shots_test 100% --folds 1 --experiment OCT10_OCTDL --method zero_shot --weights_path $OCT_path --project_features True --modality OCT

# CXR
CXR_path="./results/pretraining/MedKCO_CXR.pth"
domain_knowledge=False
python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment Chexpert5*200 --method zero_shot --weights_path $CXR_path --project_features True --norm_features False --domain_knowledge $domain_knowledge --modality CXR
python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment RSNA_Pneumonia --method zero_shot --weights_path $CXR_path --project_features True --norm_features False --domain_knowledge $domain_knowledge --modality CXR
python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment siim_pneumothorax --method zero_shot --weights_path $CXR_path --project_features True --norm_features False --domain_knowledge $domain_knowledge --modality CXR
python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment COVIDx --method zero_shot --weights_path $CXR_path --project_features True --norm_features False --domain_knowledge $domain_knowledge --modality CXR
