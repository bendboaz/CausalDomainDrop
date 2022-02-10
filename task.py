import logging
from pathlib import Path
from typing import Dict

import torch
import transformers
from ignite.contrib.engines.common import setup_common_training_handlers
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events
from ignite.metrics import Loss, Accuracy
from ignite.metrics.fbeta import Fbeta
from torch import optim
from torch.utils.data import DataLoader

from DataManagement.domain_datasets import BaseDomainSentimentDataset
from bert_extensions import FineTuningHead
from ignite_utils import create_transformers_evaluator, \
    create_transformers_trainer, output_for_metrics, transformers_prep_batch
from utils import get_log_results


def finetune_model(model_class: type, dataset_type: str, domain_name: str,
                   config: Dict, tokenizer: transformers.PreTrainedTokenizer,
                   device=None, pretrained_bert_state_dict=None, save_dir=None):
    """
    Fine tunes a given model on the desired partition.
    model_class should describe a module that works like a
    transformer (gets input_ids and attention_mask, returns output
    in a tuple with the prediction scores at the beginning).
    """
    assert issubclass(model_class, transformers.BertPreTrainedModel)
    ft_params = config['ft']

    device = torch.device('cpu') if device is None else device

    batch_size = ft_params['batch_size']
    num_workers = ft_params['workers']

    datasets = {partition: BaseDomainSentimentDataset.get_dataset(
        dataset_type,
        domain_name,
        partition,
        tokenizer
    )
        for partition in 'train dev'.split()}
    dataloaders = {partition: DataLoader(
        dataset,
        batch_size,
        shuffle=(partition == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
        for partition, dataset in datasets.items()}

    main_task = FineTuningHead(None, num_labels=ft_params['num_classes'])

    model = model_class.from_pretrained(
        config['model_class'],
        finetuning_tasks={'sentiment': main_task},
        unfreeze_layers=ft_params['unfreeze_layers']
    )
    if pretrained_bert_state_dict is not None:
        model.bert.load_state_dict(pretrained_bert_state_dict)

    params_to_train = list(filter(
        lambda x: x.requires_grad,
        model.parameters()
    ))
    optimizer = optim.AdamW(
        params_to_train,
        lr=ft_params['init_lr'],
        weight_decay=ft_params['weight_decay']
    )
    loss_func = torch.nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        len(dataloaders['train']),
        gamma=ft_params['lr_gamma']
    )

    n_gpu = torch.cuda.device_count()
    with_dataparallel = n_gpu > 1 and not config.get('no_cuda', False)
    if with_dataparallel:
        model = torch.nn.DataParallel(model, output_device=device)

    train_engine = create_transformers_trainer(
        model,
        optimizer,
        device=device,
        non_blocking=True,
        prepare_batch=transformers_prep_batch,
        average_outputs=with_dataparallel
    )
    eval_engine = create_transformers_evaluator(
        model,
        metrics=dict(cross_entropy=Loss(
            loss_func,
            output_transform=output_for_metrics,
            device=device
        ),
            accuracy=Accuracy(output_transform=output_for_metrics,
                              device=device),
            f1=Fbeta(0.5, output_transform=output_for_metrics,
                     device=device)
        ),
        device=device,
        non_blocking=True,
        prepare_batch=transformers_prep_batch,
        average_outputs=with_dataparallel
    )

    setup_common_training_handlers(
        train_engine,
        lr_scheduler=lr_scheduler,
        log_every_iters=ft_params['log_frequency'],
        device=device
    )

    train_metrics = {'ce': Loss(loss_func,
                                output_transform=output_for_metrics,
                                device=device),
                     'acc': Accuracy(output_transform=output_for_metrics,
                                     device=device),
                     'f1': Fbeta(0.5, output_transform=output_for_metrics,
                                 device=device)}

    for name, metric in train_metrics.items():
        metric.attach(train_engine, name)

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

    bare_model = model.module if with_dataparallel else model
    del model

    if save_dir is not None:
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        bare_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    return bare_model, eval_engine.state.metrics
