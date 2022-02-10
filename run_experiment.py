import json
import logging
import os
from argparse import Namespace, ArgumentParser
from pathlib import Path

import pandas as pd
import torch
import yaml
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from torch.utils.data import DataLoader
from transformers import BertPreTrainedModel, BertTokenizer

from DataManagement.causalm_format import get_pivots_concept_csv
from DataManagement.domain_datasets import BaseDomainSentimentDataset
from bert_extensions import BertForFineTuning
from bert_mlm_finetune import BertForMLMPreTraining
from bert_pivots_finetune import BertForPivotTreatControlPreTraining, BertForPivotTreatPreTraining
from causalm_utils import RANDOM_SEED, init_logger
from constants import *
from constants import MAX_SENTIMENT_SEQ_LENGTH, MLM_PROB, MAX_PRED_PER_SEQ
# import utils
from filename_formats import get_experiment_title, get_additional_pretrain_title, get_domain_title, \
    get_details_file_name, ConceptType
from ignite_utils import create_transformers_evaluator, transformers_prep_batch
from paths import EXPERIMENTS_ROOT
from pregenerate_training_data import generate_data_for_domain
from task import finetune_model
from utils import set_random_seed, get_tokenizer


def pregenerate_source_data(config):
    source = config['source']
    target = config['target']
    n_pivots = config['n_pivots']
    concept_type = ConceptType.str2concept(config.get('concept_type', 'uni'))
    n_clusters = config.get('n_clusters', None)
    tc_index = config.get('tc_index', None)
    cc_index = config.get('cc_index', None)
    shap = config.get('shap', False)

    tokenizer = get_tokenizer(config['model_class'])
    source_path = BaseDomainSentimentDataset.get_dataset_root_path(source['dataset'], source['domain'])
    target_path = BaseDomainSentimentDataset.get_dataset_root_path(target['dataset'], target['domain'])

    domain_paths = {'source': source_path, 'target': target_path}

    output_dir = Path(source_path) / get_experiment_title(config)
    if output_dir.is_dir() and any(output_dir.iterdir()) \
            and not config.get('force', False):
        LOGGER.warning(f'Directory {output_dir} already exists, skipping pregeneration. '
                       f'If you want to run pregeneration anew despite having an equivalent past run, '
                       f'add the \'force = True\' entry in your config.')
        return

    # Creating the pivot csv files so that they exist later on.
    concept_csv_kwargs = {'n_pivots': n_pivots, 'concept_type': concept_type, 'shap': shap}
    if n_clusters is not None:
        concept_csv_kwargs['n_clusters'] = n_clusters
    train_csv = get_pivots_concept_csv(domain_paths['source'], 'train', tokenizer,
                                       **concept_csv_kwargs)
    dev_csv = get_pivots_concept_csv(domain_paths['source'], 'dev', tokenizer,
                                     **concept_csv_kwargs)

    pregen_args = Namespace(
        pivots_path=source_path,
        output_dir=output_dir,
        bert_model=config['model_class'],
        do_whole_word_mask=PRETRAIN_DATA_WHOLE_WORD_MASK,
        reduce_memory=PRETRAIN_DATA_REDUCE_MEMORY,
        num_workers=PRETRAIN_DATA_NUM_WORKERS,
        epochs_to_generate=config['max_epochs'],
        max_seq_len=MAX_SENTIMENT_SEQ_LENGTH,
        short_seq_prob=PRETRAIN_DATA_SHORT_SEQ_PROB,
        masked_lm_prob=MLM_PROB,
        max_predictions_per_seq=MAX_PRED_PER_SEQ,
        domain=get_domain_title(config),
        treated_index=tc_index,
        control_index=cc_index,
        n_pivots=config['n_pivots'],
        concept_type=concept_type,
        shap=config.get('shap', False)
    )
    generate_data_for_domain(pregen_args, get_domain_title(config))


def additional_pretraining(config):
    source = config['source']
    target = config['target']
    task = config['task']
    source_path = BaseDomainSentimentDataset.get_dataset_root_path(source['dataset'], source['domain'])
    target_path = BaseDomainSentimentDataset.get_dataset_root_path(target['dataset'], target['domain'])

    domain_paths = {'source': source_path, 'target': target_path}
    source_title = get_domain_title(config)
    pretrain_output_title = get_additional_pretrain_title(config)
    exps_path = Path(EXPERIMENTS_ROOT)
    experiment_dir = exps_path / 'pretraining' / source_title / pretrain_output_title
    if experiment_dir.is_dir() and (experiment_dir / 'model' / 'config.json').is_file() \
            and not config.get('force', False):
        LOGGER.warning(f'Directory {experiment_dir} already exists, skipping pretrain. '
                       f'If you want to run pretraining anew despite having an equivalent past run, '
                       f'add the \'force = True\' entry in your config.')
        return

    if task in ['MLM', 'CAUSALM']:
        pretrain_args = Namespace(
            pregenerated_data=Path(domain_paths['source']) / get_experiment_title(config),
            output_dir=experiment_dir / 'model',
            bert_model=config['model_class'],
            reduce_memory=PRETRAIN_REDUCE_MEMORY,
            epochs=config['max_epochs'],
            local_rank=-1,
            no_cuda=config['no_cuda'],
            gradient_accumulation_steps=config['pretrain']['gradient_accumulation_steps'],
            train_batch_size=config['pretrain']['batch_size'],
            fp16=PRETRAIN_FP16,
            loss_scale=PRETRAIN_LOSS_SCALE,
            warmup_steps=config['pretrain']['warmup_steps'],
            adam_epsilon=config['pretrain']['adam_epsilon'],
            learning_rate=config['pretrain']['learning_rate'],
            seed=RANDOM_SEED,
            domain=get_domain_title(config),
            control=config['pretrain']['use_control'],
        )
        if task == 'MLM':
            from mlm_finetune_on_pregenerated import pretrain_on_domain as pretrain_func
        else:
            from pivots_finetune_on_pregenerated import pretrain_on_domain as pretrain_func

        LOGGER.info(f"\nPretraining for domain: {pretrain_args.domain}")
        pretrain_func(pretrain_args)


def finetune_pretrained(config):
    """
        - Load model from the pretraining function.
        - Decide whether to use Lightning or ignite.
        - Finetune on source.
    """
    source = config['source']
    task = config['task']

    source_title = get_domain_title(config)
    exps_path = Path(EXPERIMENTS_ROOT)

    pretrained_path = exps_path / 'pretraining' / source_title / get_additional_pretrain_title(config) / 'model'

    # Freeze all encoder layers, in case the config was transferred from an older version.
    config['ft']['unfreeze_layers'] = 0
    if task == 'NO_EXTRA':
        pretrained_model = BertPreTrainedModel.from_pretrained(config['model_class'])
        tokenizer = get_tokenizer(config['model_class'])
    else:
        if task == 'MLM':
            pretrained_model_class = BertForMLMPreTraining
        elif 'cc_index' not in config or config['cc_index'] == 'None':
            pretrained_model_class = BertForPivotTreatPreTraining
        else:
            pretrained_model_class = BertForPivotTreatControlPreTraining
        wrapper_model = pretrained_model_class.from_pretrained(str(pretrained_path))
        tokenizer = BertTokenizer.from_pretrained(str(pretrained_path))
        pretrained_model = wrapper_model.bert

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() and not config['no_cuda'] else 'cpu')
    modified_save_dir = exps_path / 'finetuned' / get_domain_title(config) / get_details_file_name(config)
    if modified_save_dir.is_dir() and (modified_save_dir / 'config.json').is_file() \
            and not config.get('force', False):
        LOGGER.warning(f'Directory {modified_save_dir} already exists, loading modified finetuned model and metrics.'
                       f'If you want to run finetuning anew despite having an equivalent past run, '
                       f'add the \'force = True\' entry in your config.')
    else:
        modified_finetuned_model, source_metrics = finetune_model(BertForFineTuning,
                                                                  source['dataset'], source['domain'],
                                                                  config, tokenizer,
                                                                  device=device,
                                                                  pretrained_bert_state_dict=pretrained_model.state_dict(),
                                                                  save_dir=modified_save_dir)
        with open(modified_save_dir / 'metrics.json', 'w+') as f:
            json.dump(source_metrics, f)

    gold_save_dir = exps_path / 'finetuned' / get_domain_title(config) / get_details_file_name(config, True)
    if gold_save_dir.is_dir() and (gold_save_dir / 'config.json').is_file() \
            and not config.get('force', False):
        LOGGER.warning(f'Directory {gold_save_dir} already exists, loading original finetuned model and metrics.'
                       f'If you want to run finetuning anew despite having an equivalent past run, '
                       f'add the \'force = True\' entry in your config.')
    else:
        gold_model, gold_metrics = finetune_model(BertForFineTuning, source['dataset'], source['domain'], config,
                                                  tokenizer,
                                                  device=device, save_dir=gold_save_dir)
        with open(gold_save_dir / 'metrics.json', 'w+') as f:
            json.dump(gold_metrics, f)


def per_sample_evaluation(config):
    source = config['source']
    target = config['target']
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() and not config['no_cuda'] else 'cpu')

    exps_path = Path(EXPERIMENTS_ROOT)

    modified_save_dir = exps_path / 'finetuned' / get_domain_title(config) / get_details_file_name(config)
    gold_save_dir = exps_path / 'finetuned' / get_domain_title(config) / get_details_file_name(config, True)

    modified_finetuned_model = BertForFineTuning.from_pretrained(str(modified_save_dir))
    tokenizer = BertTokenizer.from_pretrained(str(modified_save_dir))

    pair_run_name = f'{get_domain_title(config)}_{get_domain_title(config, is_target=True)}'

    details_dir = exps_path / 'per_sample' / pair_run_name
    details_dir.mkdir(parents=True, exist_ok=True)

    domains = {'source': source, 'target': target}
    datasets = {name: BaseDomainSentimentDataset.get_dataset(value['dataset'], value['domain'], 'dev', tokenizer,
                                                             with_ids=True) for name, value in domains.items()}
    loaders = {name: DataLoader(dataset, config['ft']['batch_size'],
                                num_workers=config['ft']['workers'],
                                shuffle=False,
                                pin_memory=True)
               for name, dataset in datasets.items()}

    def _log_individual_predictions(engine, output_file, is_target=False):
        batch = engine.state.batch
        sample_ids = batch[0][0].cpu()
        predictions = engine.state.output['y_pred']['sentiment'].cpu().tolist()

        tc_column = torch.empty(*sample_ids.shape)
        tc_column.fill_(config.get('tc_index', None))

        cc_column = torch.empty(*sample_ids.shape)
        cc_column.fill_(config.get('cc_index', None))

        source_target_column = torch.ones(*sample_ids.shape) if is_target else torch.zeros(*sample_ids.shape)

        batch_ids_and_predictions = pd.DataFrame.from_dict({'id': sample_ids,
                                                            'tc': tc_column,
                                                            'cc': cc_column,
                                                            'is_target': source_target_column,
                                                            'predictions': predictions})
        batch_ids_and_predictions.to_csv(output_file, index=False, mode='a+',
                                         header=not os.path.isfile(output_file))

    def evaluate_for_model(model, output_file, desc=None):
        if desc is None:
            desc = 'Predictions per sample'
        per_sample_evaluator = create_transformers_evaluator(model,
                                                             metrics=None,
                                                             device=device,
                                                             non_blocking=True,
                                                             prepare_batch=transformers_prep_batch)

        for name, loader in loaders.items():
            pbar = ProgressBar(persist=False, desc=desc + name)
            pbar.attach(per_sample_evaluator)
            logging_handle = per_sample_evaluator.add_event_handler(Events.ITERATION_COMPLETED,
                                                                    _log_individual_predictions,
                                                                    output_file,
                                                                    name == 'target')
            per_sample_evaluator.run(loader)
            logging_handle.remove()
            pbar.close()

    modified_model_predictions_filename = get_details_file_name(config, is_gold=False)
    modified_model_predictions_filename += '.csv'
    output_file = details_dir / modified_model_predictions_filename
    if (not output_file.is_file()) or config.get('force', False):
        evaluate_for_model(modified_finetuned_model, output_file,
                           'Modified predictions/sample, ')
    else:
        logging.info(f"File {output_file} already exists, skipping"
                     f"logging of modified predictions per sample. "
                     f"If you want to override the current results, "
                     f"use 'force = True' in the config file.")

    original_model_predictions_filename = get_details_file_name(config, is_gold=True) + '.csv'
    if os.path.isfile(details_dir / original_model_predictions_filename) and not config.get('force', False):
        logging.info(f"File {details_dir / original_model_predictions_filename} already exists, skipping"
                     f"logging of original predictions per sample. If you want to override the current "
                     f"results, use 'force = True' in the config file.")
        return

    gold_model = BertForFineTuning.from_pretrained(str(gold_save_dir))
    evaluate_for_model(gold_model, details_dir / original_model_predictions_filename,
                       'Original predictions/sample, ')


# def compute_aggregate_measures(config):
#     target = config['target']
#     device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() and not config['no_cuda'] else 'cpu')
#
#     exps_path = Path(EXPERIMENTS_ROOT)
#
#     modified_save_dir = exps_path / 'finetuned' / get_domain_title(config) / get_details_file_name(config)
#     gold_save_dir = exps_path / 'finetuned' / get_domain_title(config) / get_details_file_name(config, True)
#
#     tokenizer = BertTokenizer.from_pretrained(str(modified_save_dir))
#     gold_model = BertForFineTuning.from_pretrained(str(gold_save_dir))
#
#     with open(modified_save_dir / 'metrics.json', 'r') as f:
#         source_metrics = json.load(f)
#
#     with open(gold_save_dir / 'metrics.json', 'r') as f:
#         gold_metrics = json.load(f)
#
#     result_metrics = {'source': get_domain_title(config),
#                       'target': get_domain_title(config, is_target=True),
#                       'task': config['task'],
#                       'tc': config['tc_index'],
#                       'cc': config.get('cc_index', None)}
#
#     pretraining_hyperparams = config['pretrain'].copy()
#     finetuning_hyperparams = config['ft'].copy()
#     del pretraining_hyperparams['local_rank']
#     del pretraining_hyperparams['use_control']
#     del finetuning_hyperparams['workers']
#     del finetuning_hyperparams['num_classes']
#     del finetuning_hyperparams['log_frequency']
#     del finetuning_hyperparams['max_epochs']
#     result_metrics.update(**{f'pretrain_{key}': value for key, value in pretraining_hyperparams.items()},
#                           **{f'finetune_{key}': value for key, value in finetuning_hyperparams.items()})
#
#     pairs_outputs_path = exps_path / 'pairs'
#     pairs_outputs_path.mkdir(parents=True, exist_ok=True)
#     pair_run_name = f'{get_domain_title(config)}_{get_domain_title(config, is_target=True)}'
#     pair_summary_filename = f'{pair_run_name}.csv'
#     existing_lines_path = pairs_outputs_path / pair_summary_filename
#     if existing_lines_path.is_file():
#         existing_lines = pd.read_csv(pairs_outputs_path / pair_summary_filename)
#     else:
#         existing_lines = None
#     hyperparams_df = pd.DataFrame.from_dict({k: [v] for k, v in result_metrics.items()},
#                                             orient='columns')
#     if existing_lines is not None \
#             and (len(hyperparams_df.merge(existing_lines).index) > 0) \
#             and not config.get('force', False):
#         LOGGER.warning(f'Aggregated metrics file already exists, skipping this stage.'
#                        f'If you want to add this line anew despite having an equivalent past run, '
#                        f'add the \'force = True\' entry in your config.')
#         return
#
#     result_metrics.update({'modified_source_f1': source_metrics['f1'],
#                            'modified_source_acc': source_metrics['accuracy'],
#                            'source_f1': gold_metrics['f1'],
#                            'source_acc': gold_metrics['accuracy']})
#
#     eval_engine = create_transformers_evaluator(
#         gold_model,
#         metrics={'accuracy': Accuracy(output_transform=output_for_metrics),
#                  'f1': Fbeta(0.5, output_transform=output_for_metrics)},
#         device=device,
#         non_blocking=True,
#         prepare_batch=transformers_prep_batch
#     )
#     ProgressBar(persist=False, desc='Evaluating').attach(eval_engine)
#     dev_ds = BaseDomainSentimentDataset.get_dataset(target['dataset'], target['domain'], 'dev', tokenizer,
#                                                     with_ids=True)
#     dev_loader = DataLoader(dev_ds, config['ft']['batch_size'],
#                             num_workers=config['ft']['workers'],
#                             shuffle=False,
#                             pin_memory=True)
#     eval_engine.run(dev_loader)
#     result_metrics['target_f1'] = eval_engine.state.metrics['f1']
#     result_metrics['target_acc'] = eval_engine.state.metrics['accuracy']
#
#     for key, value in result_metrics.items():
#         result_metrics[key] = [value]
#
#     results_df = pd.DataFrame.from_dict(result_metrics, orient='columns')
#     all_results = existing_lines.append(results_df, ignore_index=True, sort=True) \
#         if existing_lines is not None else results_df
#     all_results.to_csv(pairs_outputs_path / pair_summary_filename, mode='w+', index=False, header=True)


def run_experiment(config):
    """
    Input:
        dictionary with the following entries
        - random_seed  # (optional)
        - source:
            - dataset
            - domain
        - target:
            - dataset
            - domain
        - task # (Can be {MLM, CAUSALM, NO_EXTRA})
        - force  # (flag, whether to generate/pretrain if result directories already exist.
        - generate  # (flag, optional)
        - extra_pretrain # (flag, optional)
        - finetune # (flag, optional)
        - n_pivots
        - n_grams # (one of [UNI | BI | KMEANS])
        - n_clusters # optional
        - tc_index
        - cc_index
        - no_cuda
        - pretrain:
            - local_rank
            - gradient_accumulation_steps
            - batch_size
            - warmup_steps
            - adam_epsilon
            - learning_rate
            - use_control
        - model_class: bert-[base|large]-[cased|uncased]
        - max_epochs
        - ft:
            batch_size
            workers
            num_classes
            init_lr
            weight_decay
            lr_gamma
            log_frequency
            max_epochs
        - per_sample
        - aggregated_measures
        - shap # Boolean, whether or not to sort pivots by SHAP values

    Output:
        - source
        - target
        - f1 per finetuning epoch
        - MLM loss per extra pretraining epoch (optional)
        - TC loss per extra pretraining epoch (optional)
        - CC loss per extra pretraining epoch (optional)
    """
    set_random_seed(config.get('random_seed', RANDOM_SEED))

    global LOGGER
    LOGGER = init_logger(LOGGER_NAME, Path(EXPERIMENTS_ROOT) / 'logs' / get_additional_pretrain_title(config))
    LOGGER.info('Beginning run for source %s and target %s',
                get_domain_title(config),
                get_domain_title(config, is_target=True))

    if 'per_sample' not in config:
        config['per_sample'] = config.get('finetune', False)

    if 'aggregated_measures' not in config:
        config['aggregated_measures'] = config.get('finetune', False)

    if 'concept_type' not in config:
        config['concept_type'] = config.get('n_grams', 'UNI')

    if 'generate' in config and config['generate']:
        pregenerate_source_data(config)

    if 'extra_pretrain' in config and config['extra_pretrain']:
        additional_pretraining(config)

    if 'finetune' in config and config['finetune']:
        finetune_pretrained(config)

    if config.get('per_sample', False):
        per_sample_evaluation(config)

    # Removed these, they're redundant once we have the per-sample predictions.
    # if config.get('aggregated_measures', False):
    #     compute_aggregate_measures(config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_path', type=str,
                        help='Path to the .yml file containing the '
                             'configurations for this run.')
    parser.add_argument('--copy_config', type=str, default=None,
                        help='Path to save a copy of the config file, '
                             'if desired.')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Automatically supplied when using torch.distributed.launch')
    yaml_args = parser.parse_args()
    yaml_path = yaml_args.config_path

    with open(yaml_path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    args['pretrain']['local_rank'] = yaml_args.local_rank
    if yaml_args.copy_config is not None:
        save_path = yaml_args.copy_config
        save_dir = os.path.dirname(save_path)
        if save_dir not in ['', '.'] and not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'w+') as config_copy:
            yaml.dump(args, config_copy)

    run_experiment(args)
