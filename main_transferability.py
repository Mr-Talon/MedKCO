import argparse
import torch
import torch.multiprocessing as mp

from medkco.modeling.model import MedKCOModel
from medkco.transferability.data.dataloader import get_dataloader_splits
from medkco.utils.metrics import evaluate, average_folds_results, save_results, retrieval_image_text
from medkco.modeling.misc import set_seeds
from medkco.transferability.modeling.adapters import ZeroShot, Retrieval

from local_data.constants import *
from local_data.experiments import get_experiment_setting

import warnings
warnings.filterwarnings("ignore")
mp.set_sharing_strategy('file_system')

set_seeds(42, use_cuda=torch.cuda.is_available())


def init_adapter(model, args):
    adapter = None
    # zero shot
    if args.method == "zero_shot":
        print("Zero-shot classification...", end="\n")
        adapter = ZeroShot(model, args.setting["targets"], domain_knowledge=args.domain_knowledge, modality=args.modality)
    # Retrieval
    elif args.method == 'Retrieval':
        print("Retrieval...", end="\n")
        adapter = Retrieval(model, args.setting["targets"], domain_knowledge=args.domain_knowledge)
    return adapter


def generate_experiment_id(args):
    id = args.experiment + '_method_' + args.method +\
         '_domain knowledge_' + str(args.domain_knowledge) + \
         '_proj_' + str(args.project_features)
    return id


def process(args):
    args.metrics_test, args.weights = [], []
    experiment_id = generate_experiment_id(args)
    print(experiment_id)

    for iFold in range(args.folds):
        print("\nTransferability (fold : " + str(iFold + 1) + ")", end="\n")
        args.iFold = iFold

        if args.modality == "CXR":
            image_size = (224, 224)
            caption = "A chest x-ray photograph of [CLS]"
        elif args.modality == "OCT":
            image_size = (512, 512)
            caption = "An Optical coherence tomography(OCT) photograph of [CLS]"
        elif args.modality == "CFP":
            image_size = (512, 512)
            caption = "A fundus photograph of [CLS]"

        args.setting = get_experiment_setting(args.experiment)
        args.loaders = get_dataloader_splits(args.setting["dataframe"], args.data_root_path, args.setting["targets"],
                                             shots_train=args.shots_train, shots_val=args.shots_val,
                                             shots_test=args.shots_test, balance=args.balance,
                                             batch_size=args.batch_size, num_workers=args.num_workers, seed=iFold,
                                             task=args.setting["task"], size=image_size,
                                             batch_size_test=args.batch_size_test, dataset=args.experiment)

        model = MedKCOModel(from_checkpoint=args.load_weights, weights_path=args.weights_path,
                      projection=args.project_features, norm_features=args.norm_features,
                      vision_pretrained=args.init_imagenet, caption=caption, bert_type="./Bio_ClinicalBERT")
        adapter = init_adapter(model, args)

        if args.loaders["test"] is not None:
            if args.setting["task"] == "classification":
                refs, preds = adapter.predict(args.loaders["test"], dataset=args.experiment)
                metrics_fold = evaluate(refs, preds, args.setting["task"])
            elif args.setting["task"] == "Retrieval":
                img_embeds, text_embeds, text = adapter.predict(args.loaders["test"], dataset=args.experiment)
                metrics_fold = retrieval_image_text(img_embeds, text_embeds, text)
            args.metrics_test.append(metrics_fold)
        args.weights.append(adapter.model.state_dict())

    if args.loaders["test"] is not None:
        print("\nTransferability (cross-validation)", end="\n")
        args.metrics = average_folds_results(args.metrics_test, args.setting["task"])
    else:
        args.metrics = None

    save_results(args.metrics, args.out_path, id_experiment=generate_experiment_id(args),
                 id_metrics="metrics", save_model=args.save_model, weights=args.weights)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_RESULTS_TRANSFERABILITY, help='output path')
    parser.add_argument('--save_model', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--shots_train', default="0%", type=lambda x: (str(x)))
    parser.add_argument('--shots_val', default="0%", type=lambda x: (str(x)))
    parser.add_argument('--shots_test', default="100%", type=lambda x: (str(x)))
    parser.add_argument('--balance', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--folds', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--batch_size_test', default=8, type=int)
    parser.add_argument('--size', default=(512, 512), help="CFP/OCT(512, 512), CXR(224, 224)")

    parser.add_argument('--modality', default='CFP')
    parser.add_argument('--experiment', default='13_FIVES')
    parser.add_argument('--method', default='Retrieval',
                        help='zero_shot - Retrieval')
    parser.add_argument('--num_workers', default=16, type=int, help='workers number for DataLoader')

    parser.add_argument('--weights_path', default='')
    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--init_imagenet', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--project_features', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--norm_features', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--domain_knowledge', default=True, type=lambda x: (str(x).lower() == 'true'))
    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    main()
