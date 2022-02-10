import os
from argparse import ArgumentParser
from itertools import product
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import yaml
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataManagement.domain_datasets import NAME2DATASET
from bert_extensions import BertForFineTuning
from ignite_utils import create_transformers_evaluator, output_for_metrics, \
    transformers_prep_batch
from legacy_code.adversarial import adversarial_estimation
from legacy_code.instance_reweighting import estimate_da_performance_loss
from task import finetune_model
from utils import get_tokenizer, set_random_seed


def pair_to_colname(pair):
    return f'{pair[0]}_{pair[1]}'


if __name__ == '__main__':
    set_random_seed()

    parser = ArgumentParser()
    parser.add_argument('config_path', type=str,
                        help='Path to the .yml file containing the '
                             'configurations for this run.')
    parser.add_argument('--copy_config', type=str, default=None,
                        help='Path to save a copy of the config file, '
                             'if desired.')
    yaml_args = parser.parse_args()
    yaml_path = yaml_args.config_path

    with open(yaml_path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    if yaml_args.copy_config is not None:
        save_path = yaml_args.copy_config
        save_dir = os.path.dirname(save_path)
        if save_dir not in ['', '.'] and not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'w+') as config_copy:
            yaml.dump(args, config_copy)

    name2estimator = {'uri': estimate_da_performance_loss,
                      'adversarial': adversarial_estimation}

    # my_device = torch.device('cpu')
    my_device = torch.device(torch.cuda.current_device()
                             if torch.cuda.is_available() else 'cpu')

    # TODO: Manage datasets with one domain.
    dataset_domains = [(ds_type['type'], domain)
                       for ds_type in args['dataset']
                       for domain in ds_type['domains']]
    dataset_domains = sorted(dataset_domains, key=itemgetter(0, 1))

    tokenizer = get_tokenizer(args['model_class'])

    estimators_dict = args['predict']
    result_types = ['ground truth'] + list(estimators_dict.keys())
    multi_index = pd.MultiIndex.from_product([result_types, dataset_domains])
    results = pd.DataFrame(
        data=np.zeros((len(multi_index),
                       len(dataset_domains)),
                      dtype=float),
        index=multi_index,
        columns=list(map(pair_to_colname, dataset_domains))
    )

    ft_params = args['ft']
    eval_params = args['eval']

    dev_datasets = {pair_to_colname((ds_class, domain_name)):
                        NAME2DATASET[ds_class](domain_name, 'dev', tokenizer)
                    for ds_class, domain_name in dataset_domains}
    dev_loaders = {name: DataLoader(
        dataset,
        batch_size=eval_params['batch_size'],
        num_workers=eval_params['workers'],
        shuffle=False,
        pin_memory=True
    )
        for name, dataset in dev_datasets.items()}

    for source_type, source_domain in dataset_domains:
        print(f"Source domain: {source_domain}")
        target_domains = list(filter(lambda x: x != (source_type, source_domain),
                                     dataset_domains))

        finetuned, source_metrics = finetune_model(
            BertForFineTuning,
            source_type,
            source_domain,
            args,
            tokenizer,
            my_device
        )
        results.loc[
            ('ground truth', (source_type, source_domain)),
            pair_to_colname((source_type, source_domain))
        ] = (source_metrics['cross_entropy'])

        eval_engine = create_transformers_evaluator(
            finetuned,
            metrics={'score': Loss(
                torch.nn.CrossEntropyLoss(),
                output_transform=output_for_metrics
            )},
            device=my_device,
            non_blocking=True,
            prepare_batch=transformers_prep_batch
        )
        ProgressBar(persist=False, desc='Evaluating').attach(eval_engine)

        print(f"Evaluation ground truth for domains {target_domains}")
        for target_pair in tqdm(
                target_domains,
                desc='Domains',
                bar_format='',
                total=len(target_domains)
        ):
            target_type, target_domain = target_pair
            target_loader = dev_loaders[pair_to_colname(target_pair)]
            eval_engine.run(target_loader)
            score = eval_engine.state.metrics['score']
            results.loc[('ground truth', (source_type, source_domain)),
                        pair_to_colname((target_type, target_domain))] = score
            del target_type, target_domain

        if len(estimators_dict) > 0:
            print(f"Running estimators {list(estimators_dict.keys())} "
                  f"over domains {target_domains}")

        for estimator, (target_type, target_domain) in product(
                estimators_dict,
                target_domains
        ):
            estimate = name2estimator[estimator](
                source_type,
                source_domain,
                target_type,
                target_domain,
                BertForFineTuning,
                tokenizer,
                args,
                'train',
                estimators_dict[estimator],
                device=my_device
            )
            if isinstance(estimate, torch.Tensor):
                estimate = estimate.item()
            results.loc[
                (estimator, (source_type, source_domain)),
                pair_to_colname((target_type, target_domain))
            ] = estimate

        del target_type, target_domain, estimator
        print(f"Done with {source_domain}!")

    print(results)
    try:
        save_path = args['administrative']['save_results']
        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if not os.path.isdir(os.path.dirname(save_dir)):
                os.makedirs(os.path.dirname(save_dir))
            results.to_csv(
                save_path,
                float_format="%.4f"
            )
    except KeyError:
        pass
