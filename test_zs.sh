export CUDA_VISIBLE_DEVICES=4
#model_path="/mnt/data/zcr/SPCL_found/results/pretraining/CL_smoothtext_perstage_b24_g1/resnet_v2_epoch"   # 3090
model_path="/data/crzhang/SPCL_found/results/pretraining/cvprrebuttal/resnet_v2_epoch"   # A6000
knowledge_dict=False
domain_knowledge=False
Data_Root_Path="/data/rqwu/OCT_dataset/transferability/"    # OCT使用

##########################################################################
##########################################################################
# 记得改model.py里的caption和dictionary！！！！！！！！！！！！！！！！！！！！！
# 记得改main函数的size ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
##########################################################################
##########################################################################

for i in {10..25}; do
  path="$model_path${i}.pth"
  echo "Current path: $path"

  # CFP
#  python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment 08_ODIR200x3 --method zero_shot --weights_path $path --project_features True --knowledge_dict $knowledge_dict
#  python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment 13_FIVES --method zero_shot --weights_path $path --project_features True --knowledge_dict $knowledge_dict
#  python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment 25_REFUGE --method zero_shot --weights_path $path --project_features True --knowledge_dict $knowledge_dict
#  python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment TAOP --method zero_shot --weights_path $path --project_features True --knowledge_dict $knowledge_dict
#  python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment AMD --method zero_shot --weights_path $path --project_features True --knowledge_dict $knowledge_dict

  # OCT
  python main_transferability.py --data_root_path $Data_Root_Path --shots_train 0 --shots_test 100% --folds 1 --experiment OCT05_OCTID --method zero_shot --weights_path $path --project_features True --knowledge_dict $knowledge_dict
  python main_transferability.py --data_root_path $Data_Root_Path --shots_train 0 --shots_test 100% --folds 1 --experiment OCT10_OCTDL --method zero_shot --weights_path $path --project_features True --knowledge_dict $knowledge_dict

  # CXR
#  python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment Chexpert5*200 --method zero_shot --weights_path $path --project_features True --norm_features False --knowledge_dict $knowledge_dict --domain_knowledge $domain_knowledge
#  python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment RSNA_Pneumonia --method zero_shot --weights_path $path --project_features True --norm_features False --knowledge_dict $knowledge_dict --domain_knowledge $domain_knowledge
#  python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment siim_pneumothorax --method zero_shot --weights_path $path --project_features True --norm_features False --knowledge_dict $knowledge_dict --domain_knowledge $domain_knowledge
#  python main_transferability.py --shots_train 0 --shots_test 100% --folds 1 --experiment COVIDx --method zero_shot --weights_path $path --project_features True --norm_features False --knowledge_dict $knowledge_dict --domain_knowledge $domain_knowledge
done