from functools import partial
from typing import Dict, Optional

from ignite.contrib.engines import common
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from ignite.metrics import Loss
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from DataManagement.domain_datasets import BaseDomainSentimentDataset, \
    DomainRecognitionDataset, MultiTaskDataset
from bert_extensions import BertForFineTuning, FineTuningHead
from ignite_utils import create_transformers_trainer, \
    create_transformers_evaluator, transformers_prep_batch, \
    output_for_metrics, FbetaIgnore
from utils import get_log_results


def adversarial_estimation(source_type: str, source_domain: Optional[str],
                           target_type: str, target_domain: Optional[str],
                           model_class: type, tokenizer,
                           config, partition: str = 'train',
                           estimator_args: Dict = None, device=None):
    assert issubclass(model_class, PreTrainedModel)
    # dataset_class = NAME2DATASET[config['dataset']['type']]
    # assert issubclass(dataset_class, BaseDomainSentimentDataset)
    ignore_index = -1

    sentiment_ft_head = FineTuningHead(None, config['num_classes'])
    adversarial_ft_head = FineTuningHead(None, 2, do_grl=True)
    ft_tasks = {
        'sentiment': sentiment_ft_head,
        'domain': adversarial_ft_head
    }
    ft_params = estimator_args['ft']
    model = model_class.from_pretrained(
        config['model_class'],
        finetuning_tasks=ft_tasks,
        unfreeze_layers=ft_params['unfreeze_layers']
    )
    partitions = ['train', 'dev']
    sentiment_datasets = {
        partition: BaseDomainSentimentDataset.get_dataset(
            source_type,
            source_domain,
            'train',
            tokenizer
        )
        for partition in partitions
    }

    adversarial_datasets = {
        partition: DomainRecognitionDataset(
            source_type,
            source_domain,
            target_type,
            target_domain,
            partition,
            tokenizer
        )
        for partition in partitions
    }

    joint_datasets = {
        partition: MultiTaskDataset(ignore_index=ignore_index,
                                    sentiment=sentiment_datasets[partition],
                                    domain=adversarial_datasets[partition])
        for partition in partitions
    }

    batch_size = ft_params['batch_size']
    num_workers = ft_params['workers']

    dataloaders = {partition: DataLoader(dataset, batch_size, num_workers,
                                         pin_memory=True)
                   for partition, dataset in joint_datasets.items()}

    params_to_train = list(filter(
        lambda x: x.requires_grad,
        model.parameters()
    ))
    optimizer = optim.AdamW(
        params_to_train,
        lr=ft_params['init_lr'],
        weight_decay=ft_params['weight_decay']
    )
    loss_func = nn.CrossEntropyLoss(ignore_index=ignore_index)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        len(dataloaders['train']),
        gamma=ft_params['lr_gamma']
    )

    train_engine = create_transformers_trainer(
        model,
        optimizer,
        device=device,
        non_blocking=True,
        prepare_batch=transformers_prep_batch,
    )

    train_metrics = {}
    eval_metrics = {}
    for task in ft_tasks.keys():
        train_metrics[f'{task}_loss'] = Loss(
            loss_func,
            output_transform=partial(output_for_metrics, task_name=task),
            device=device
        )
        train_metrics[f'{task}_f1'] = FbetaIgnore(
            0.5,
            output_transform=partial(output_for_metrics,
                                     task_name=task),
            device=device,
            ignore_index=ignore_index
        )
        eval_metrics[f'{task}_loss'] = Loss(
            loss_func,
            output_transform=partial(output_for_metrics, task_name=task),
            device=device
        )
        eval_metrics[f'{task}_f1'] = FbetaIgnore(
            0.5,
            output_transform=partial(output_for_metrics,
                                     task_name=task),
            device=device,
            ignore_index=ignore_index
        )

    for name, metric in train_metrics.items():
        metric.attach(train_engine, name)

    eval_engine = create_transformers_evaluator(
        model,
        metrics=eval_metrics,
        device=device,
        non_blocking=True,
        prepare_batch=transformers_prep_batch,
    )

    common.setup_common_training_handlers(
        train_engine,
        lr_scheduler=lr_scheduler,
        log_every_iters=ft_params['log_frequency'],
        device=device
    )

    common.add_early_stopping_by_val_score(3, eval_engine, train_engine,
                                           'domain_f1')

    @train_engine.on(Events.EPOCH_COMPLETED)
    def print_training_metrics(engine):
        print(f'Epoch {engine.state.epoch} training metrics:')
        print(", ".join([f'{name}: {value}'
                         for name, value in engine.state.metrics.items()]))

    train_engine.add_event_handler(
        Events.EPOCH_COMPLETED,
        get_log_results(
            eval_engine,
            dataloaders['dev'],
            'Validation Results'
        )
    )

    eval_pbar = ProgressBar(persist=False, desc="Evaluation")
    eval_pbar.attach(eval_engine)

    train_engine.run(dataloaders['train'], max_epochs=ft_params['max_epochs'])

    return eval_engine.state.metrics['sentiment_loss']


if __name__ == '__main__':
    import torch
    import yaml
    from argparse import ArgumentParser
    from transformers import BertTokenizer

    parser = ArgumentParser()
    parser.add_argument('config_path', type=str,
                        help='Path to the .yml file containing the '
                             'configurations for this run.')

    yaml_args = parser.parse_args()
    yaml_path = yaml_args.config_path

    with open(yaml_path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    source_ds = 'amazon'
    source = 'beauty'
    target_ds = 'amazon'
    target = 'book'
    model_class = BertForFineTuning
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    partition = 'train'
    # device = torch.device('cpu')
    device = torch.device(torch.cuda.current_device())

    adversarial_estimation(source_ds, source, target_ds, target, model_class, tokenizer,
                           args, partition, args['predict']['adversarial'],
                           device=device)
