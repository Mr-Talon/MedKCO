import argparse
import torch
import os

from medkco.pretraining.data.transforms import augmentations_pretraining_CFPOCT, augmentations_pretraining_CXR
from medkco.pretraining.data.dataloader import get_loader
from medkco.modeling.model import MedKCOModel
from medkco.modeling.misc import set_seeds

from local_data.constants import *


def process(args):
    torch.cuda.set_device(args.local_rank)
    if args.test_code:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
        torch.distributed.init_process_group(backend='nccl')
    seed = 42
    set_seeds(seed, use_cuda=True)

    image_size, caption, norm_features, augmentations_pretraining = None, None, None, None
    if args.modality == "CXR":
        image_size = 224
        norm_features = False
        caption = ""
        augmentations_pretraining = augmentations_pretraining_CXR
    elif args.modality == "OCT":
        image_size = 512
        caption = "A [ATR] Optical coherence tomography(OCT) photograph of [CLS]"
        norm_features = True
        augmentations_pretraining = augmentations_pretraining_CFPOCT
    elif args.modality == "CFP":
        image_size = 512
        caption = "A [ATR] fundus photograph of [CLS]"
        norm_features = True
        augmentations_pretraining = augmentations_pretraining_CFPOCT

    # Create dataloader
    dataloaders = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                             datasets=args.datasets, balance=args.balance, batch_size=args.batch_size,
                             num_workers=args.num_workers, banned_categories=args.banned_categories,
                             caption=caption, augment_description=args.augment_description,
                             test_code=args.test_code, SPCL=args.SPCL, modality=args.modality)

    # define model
    model = MedKCOModel(vision_type=args.architecture, out_path=args.out_path, from_checkpoint=args.load_weights, vision_pretrained=True,
                         weights_path=args.weights_path, test_code=args.test_code, image_size=image_size, norm_features=norm_features,
                         caption = caption, modality=args.modality, bert_type=args.bert)
    model.to(device)
    if not args.test_code:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                          find_unused_parameters=True)
        model = model.module

    model.fit(dataloaders, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
              warmup_epoch=args.warmup_epoch, store_num=args.store_num, transforms=augmentations_pretraining,
              local_rank=args.local_rank, test_code=args.test_code, SPCL=args.SPCL, args=args)


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--dataframes_path', default=PATH_DATAFRAME_PRETRAIN)
    parser.add_argument('--modality', default="CFP")        # TODO
    # CFP
    parser.add_argument('--datasets', default=["01_EYEPACS", "03_IDRID", "04_RFMid",
                                               "06_DEN", "07_LAG", "08_ODIR", "09_PAPILA", "10_PARAGUAY",
                                               "11_STARE", "12_ARIA", "14_AGAR300", "15_APTOS", "16_FUND-OCT",
                                               "17_DiaRetDB1", "18_DRIONS-DB", "19_Drishti-GS1",
                                               "20_E-ophta", "21_G1020", "23_HRF", "24_ORIGA", "26_ROC",
                                               "27_BRSET", "28_OIA-DDR", "29_AIROGS", "30_SUSTech-SYSU", "31_JICHI",
                                               "32_CHAKSU", "33_DR1-2", "34_Cataract", "35_ScarDat", "39_MM_Retinal_dataset"])
    # OCT
    # parser.add_argument('--datasets', default=["OCT01_RetinalOCT_C8", "OCT03_Large_Dataset_of_Labeled_OCT",
    #                                                         "OCT04_GAMMA1", "OCT06_STAGE1", "OCT07_STAGE2", "OCT08_glaucoma_detection",
    #                                                         "OCT09_GOALS", "OCT11_OIMHS", "OCT12_OCTA_500", "OCT14_DUKE_DME",
    #                                                         "OCT16_BIOMISA_Retinal_Image_Database_for_Macular_Disorders",
    #                                                         "OCT17_MM_Retinal_OCT"])
    # CXR
    # parser.add_argument('--datasets', default=['CheXpert-v1.0', 'mimic-cxr'])
    parser.add_argument('--banned_categories', default=['myopia', 'cataract', 'retinitis pigmentosa',
                                                        "myopic", "myope", "myop", "retinitis", "macular hole"], help="oct without: macular hole")  # TODO oct without: macular hole
    parser.add_argument('--out_path', default=PATH_RESULTS_PRETRAIN+"MedKCO_CFP/", help='output path') # TODO
    parser.add_argument('--augment_description', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))

    # model architecture
    parser.add_argument('--architecture', default='resnet_v2', help='')
    parser.add_argument('--bert', default='./Bio_ClinicalBERT', help='./Bio_ClinicalBERT')                               # TODO download

    # Training hyperparameter
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=48, type=int)                                          # TODO CXR 128    CFP/OCT 48
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')                        # TODO CXR 5e-5   CFP/OCT 1e-4
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')                           # TODO CXR 1e-4   CFP/OCT 1e-5
    parser.add_argument('--scheduler', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--weights_path', default='')
    parser.add_argument('--load_weights', default=False, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--store_num', default=1, type=int)
    parser.add_argument('--num_workers', default=16, type=int, help='workers number for DataLoader')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--test_code', default=True, type=lambda x: (str(x).lower() == 'true'), help="1 gpu/test")

    # SPCL curriculum learning
    parser.add_argument('--SPCL', default=True, type=lambda x: (str(x).lower() == 'true'))             # TODO

    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()