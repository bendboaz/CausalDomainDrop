from typing import Dict

import torch
from ignite.contrib.engines import common
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss, Fbeta
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, PreTrainedModel

from DataManagement.domain_datasets import BaseDomainSentimentDataset, \
    DomainRecognitionDataset, NAME2DATASET
from ignite_utils import create_transformers_evaluator, \
    create_transformers_trainer
from paths import GLOVE_RAW_DIR, GLOVE_PROC_DIR
from task import finetune_model
from utils import prepare_glove_vectors


class DomainClassifier(nn.Module):
    def __init__(self, embed_dim, tokenizer, n_labels=2,
                 hidden_dims=None, load_embeds=True, loss_func=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.vocab = tokenizer.vocab
        self.num_labels = n_labels
        self.loss_func = loss_func

        if self.loss_func is None:
            self.loss_func = nn.CrossEntropyLoss()

        if load_embeds:
            pretrained = prepare_glove_vectors(
                tokenizer,
                GLOVE_RAW_DIR,
                GLOVE_PROC_DIR,
                embed_dim
            )
            self.embeds = nn.Embedding.from_pretrained(pretrained)
        else:
            self.embeds = nn.Embedding(tokenizer.vocab_size(), embed_dim)

        if hidden_dims is None:
            hidden_dims = []

        hidden_dims = [embed_dim] + hidden_dims + [n_labels]
        linear_layers = [nn.Linear(inp, out)
                         for inp, out in zip(
                hidden_dims[:-1],
                hidden_dims[1:]
            )]
        all_layers = []
        for layer in linear_layers[:-1]:
            all_layers.extend([layer, nn.ReLU()])
        all_layers.append(linear_layers[-1])
        self.classifier = nn.Sequential(*all_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels=None):
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"DomainClassifier.forward: batch and attention "
                             f"masks must have the same shape, "
                             f"got {input_ids.shape} and {attention_mask.shape}.")
        embeds = self.embeds(input_ids)
        masked_embeds = embeds * attention_mask.view(*embeds.shape[:-1], 1)
        mean_embeds = (masked_embeds.sum(1)
                       / attention_mask.sum(-1).unsqueeze(-1))
        results = self.classifier(mean_embeds)
        outputs = (results,)

        if labels is not None:
            loss = self.loss_func(results, labels)
            outputs = (loss,) + outputs

        return outputs


def get_domain_classifier(source_type: str, source_domain: str,
                          target_type: str, target_domain: str,
                          tokenizer: BertTokenizer, device=None,
                          discriminator_config: Dict = None):
    datasets = {partition: DomainRecognitionDataset(
        source_type,
        source_domain,
        target_type,
        target_domain,
        partition,
        tokenizer
    )
        for partition in ['train', 'dev']}
    dataloaders = {partition: DataLoader(
        dataset,
        shuffle=(partition == 'train'),
        batch_size=discriminator_config['batch_size'],
        num_workers=discriminator_config['workers'],
        pin_memory=True
    )
        for partition, dataset in datasets.items()}

    domain_counts = datasets['train'].domain_counts()
    total_counts = sum(domain_counts.values())
    domain_classifier_loss_func = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            [domain_counts[domain] / total_counts
             for domain in sorted(domain_counts)],
            device=device)
    )

    domain_classifier = DomainClassifier(
        discriminator_config['embed_dim'],
        tokenizer,
        2,
        discriminator_config['hidden_layers'],
        loss_func=domain_classifier_loss_func
    )

    domain_classifier.to(torch.device('cpu') if device is None else device,
                         non_blocking=True)

    optimizer = optim.AdamW(
        filter(
            lambda x: x.requires_grad,
            domain_classifier.parameters()
        ),
        lr=discriminator_config['init_lr'],
        weight_decay=discriminator_config['weight_decay']
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        len(dataloaders['train']),
        discriminator_config['lr_gamma']
    )

    train_engine = create_transformers_trainer(
        domain_classifier,
        optimizer,
        device=device,
        non_blocking=True,
    )

    def shallow_output_transform(output):
        return output['y_pred'], output['y']

    eval_engine = create_transformers_evaluator(
        domain_classifier,
        metrics=dict(cross_entropy=Loss(
            domain_classifier_loss_func,
            output_transform=shallow_output_transform
        ),
            accuracy=Accuracy(
                output_transform=shallow_output_transform
            ),
            f1=Fbeta(
                0.5,
                output_transform=shallow_output_transform
            )
        ),
        device=device,
        non_blocking=True
    )

    @train_engine.on(Events.EPOCH_COMPLETED)
    def evaluate_domain_discriminator(engine):
        eval_engine.run(dataloaders['dev'])

    common.add_early_stopping_by_val_score(
        discriminator_config['patience'],
        eval_engine,
        train_engine,
        'cross_entropy'
    )
    common.setup_common_training_handlers(
        train_engine,
        lr_scheduler=lr_scheduler,
        log_every_iters=1,
        device=device
    )
    train_engine.run(
        dataloaders['train'],
        max_epochs=discriminator_config['max_epochs']
    )
    print(f'Domain discriminator validation scores '
          f'(training stopped at epoch {train_engine.state.epoch}):')
    print(', '.join(f'{name}: {value:.4f}'
                    for name, value in eval_engine.state.metrics.items()))
    return domain_classifier


def estimate_da_performance_loss(source_ds: str, source_domain: str,
                                 target_ds: str, target_domain: str,
                                 model_class: type, tokenizer,
                                 config, partition: str = 'train',
                                 estimator_args: Dict = None, device=None):
    assert issubclass(model_class, PreTrainedModel)
    finetuned_model, _ = finetune_model(
        model_class,
        source_ds,
        source_domain,
        config,
        tokenizer,
        device
    )

    dataset_class = NAME2DATASET[source_ds]
    assert issubclass(dataset_class, BaseDomainSentimentDataset)

    domain_classifier = get_domain_classifier(
        source_ds,
        source_domain,
        target_ds,
        target_domain,
        tokenizer,
        device,
        estimator_args['discriminator']
    )

    source_dataset = BaseDomainSentimentDataset.get_dataset(
        source_ds,
        source_domain,
        partition,
        tokenizer
    )

    if device is not None:
        finetuned_model.to(device, non_blocking=True)
    estimator_specific = estimator_args['estimator']
    source_loader = DataLoader(
        source_dataset,
        shuffle=False,
        batch_size=estimator_specific['batch_size'],
        num_workers=estimator_specific['workers'],
        pin_memory=True
    )

    weighted_sum = 0.0
    n_samples = 0
    loss_func = nn.CrossEntropyLoss(reduction='none')

    for (inputs, masks, token_type_ids), labels in tqdm(
            source_loader,
            desc=f'Weighted scores',
            total=len(source_loader),
            unit='batch'
    ):
        if device is not None:
            inputs = inputs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels['sentiment'] = \
                labels['sentiment'].to(device, non_blocking=True)

        domain_likelihoods = nn.functional.softmax(
            domain_classifier(inputs, masks)[0],
            dim=-1
        )
        with torch.no_grad():
            predictions = finetuned_model(inputs, attention_mask=masks)[0]
        losses = loss_func(predictions['sentiment'], labels['sentiment'])
        weighted_sum += (
            ((domain_likelihoods[:, 0] / domain_likelihoods[:, 1]) * losses)
                .sum()
                .item()
        )
        n_samples += labels['sentiment'].shape[0]

    return weighted_sum / n_samples

# if __name__ == "__main__":
# RANDOM_SEED = 1337
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)
#
# parser = ArgumentParser()
# parser.add_argument('dataset_type', type=str, choices=['amazon', 'he_small', 'he_large'])
# parser.add_argument('--domains', type=list, default=None)
# parser.add_argument('--n_epochs', type=int, default=5)
# parser.add_argument('--batch_size', type=int, default=2)
# parser.add_argument('--lr', type=float, default=1e-5)
# parser.add_argument('--lr_gamma', type=float, default=1.0)
# parser.add_argument('--weight_decay', type=float, default=0.0)
#
# args = parser.parse_args()
#
# my_device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
# name2dataset = {'amazon': AmazonReviewsDomainDataset, 'he_small': HeReviewSmall, 'he_large': HeReviewLarge}
# dataset_class = name2dataset[args.dataset_type]
# allowed_domains = sorted(dataset_class.get_allowed_domains())
# if args.domains is None:
#     domains = allowed_domains
# else:
#     if not set(args.domains).issubset(set(allowed_domains)):
#         raise ArgumentError()
#     domains = sorted(args.domains)
#
# NUM_LABELS = dataset_class.num_classes()
#
# bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# for source_domain in domains:
#     target_domains = list(filter(lambda x: x != source_domain, domains))
#     print(f"Source domain: {source_domain}")
#     bert_model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased',
#                                                                             num_labels=NUM_LABELS)
#     bert_model.num_labels = NUM_LABELS
#     bert_model.to(my_device)
#
#     estimator_loss_func = nn.CrossEntropyLoss(reduction='none')
#
#     def get_finetuning_optimizer(params):
#         return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
#
#     def get_exponential_lr_scheduler(optimizer, step_size: int):
#         return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=args.lr_gamma)
#
#     finetuned, _ = finetune_model(bert_model, dataset_class, source_domain, args.n_epochs, bert_tokenizer,
#                                get_finetuning_optimizer, get_exponential_lr_scheduler, my_device,
#                                batch_size=args.batch_size)
#
#     datasets = {domain: dataset_class(domain=domain, partition='dev', tokenizer=bert_tokenizer)
#                 for domain in domains}
#
#     dataloaders = {domain: DataLoader(dataset, shuffle=False, num_workers=4, batch_size=args.batch_size,
#                                       pin_memory=True)
#                     for domain, dataset in datasets.items()}
#
#     target_evaluator = create_transformers_evaluator(finetuned,
#                                                          metrics=dict(cross_entropy=Loss(nn.CrossEntropyLoss(),
#                                                                                          output_transform=output_for_metrics)),
#                                                          device=my_device, non_blocking=True)
#     print("Computing source loss...")
#     target_evaluator.run(dataloaders[source_domain])
#     source_loss = target_evaluator.state.metrics['cross_entropy']
#     print(f"Actual source loss: {source_loss}")
#
#     for target in target_domains:
#         print(f"Source: {source_domain}, target: {target}")
#         predicted_loss = estimate_da_performance_loss(dataset_class, source_domain, target, finetuned,
#                                                       estimator_loss_func, bert_tokenizer, device=my_device,
#                                                       num_workers=4, batch_size=args.batch_size)
#         print(f"Predicted loss is: {predicted_loss}")
#
#         print("Computing actual target loss...")
#         target_evaluator.run(dataloaders[target])
#         target_loss = target_evaluator.state.metrics['cross_entropy']
#         print(f"Actual target loss: {target_loss}")
