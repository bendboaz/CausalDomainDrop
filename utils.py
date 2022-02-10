import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, PreTrainedTokenizer, PreTrainedModel

from paths import DATA_ROOT

class MaskedAvgPooler(torch.nn.Module):
    def forward(self, sequence, sequence_masks):
        return masked_avg_pooler(sequence, sequence_masks)


def masked_avg_pooler(sequence, sequence_masks) -> torch.Tensor:
    if sequence_masks is None:
        return sequence.mean(dim=1)
    broadcast_masks = (sequence_masks
                       .float()
                       .unsqueeze(dim=-1)
                       .expand_as(sequence))
    masked_sequences = sequence * broadcast_masks
    sequence_lengths = (sequence_masks
                        .sum(dim=-1)
                        .view(-1, 1, 1)
                        .expand_as(sequence))
    return torch.sum(masked_sequences / sequence_lengths, dim=1)


def get_log_results(eval_engine, loader, title=None):
    title = 'Results' if title is None else title

    def log_training_results(trainer):
        eval_engine.run(loader)
        metrics = eval_engine.state.metrics
        print(f'{title}, Epoch {trainer.state.epoch}: '
              f'{", ".join(map(lambda item: f"{item[0]}: {item[1]:.2f}", metrics.items()))}')

    return log_training_results


def get_tokenizer(description) -> PreTrainedTokenizer:
    is_cased = description.split('-')[-1] == 'uncased'
    tokenizer = BertTokenizer.from_pretrained(description,
                                              do_lower_case=is_cased)
    return tokenizer


def get_model_from_class(model_class: type,
                         pretrained_model_name_or_path: Optional[str],
                         **kwargs):
    if issubclass(model_class, PreTrainedModel):
        return model_class.from_pretrained(pretrained_model_name_or_path,
                                           **kwargs)

    return model_class(**kwargs)


def change_module_grad(module: torch.nn.Module, new_grad=False):
    """
    Either freeze or unfreeze a module.
    :param module: Module to freeze or unfreeze.
    :param new_grad: New requires_grad value to set.
    """
    for param in module.parameters():
        param.requires_grad = new_grad


def set_random_seed(seed=None):
    RANDOM_SEED = 1337 if seed is None else seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def domains_to_pair_name(source_dataset, source_domain, target_dataset, target_domain):
    return f'{source_dataset}_{source_domain}---{target_dataset}_{target_domain}'


def prepare_glove_vectors(tokenizer: BertTokenizer, raw_path,
                          proc_path, embed_dim=50):
    if not os.path.isdir(proc_path):
        os.mkdir(proc_path)

    proc_file_path = os.path.join(proc_path, f'glove.6B.{embed_dim}.pt')
    if os.path.isfile(proc_file_path):
        print("GloVe embedding vectors already exist :)")
        return torch.load(proc_file_path)

    embed_weights = torch.zeros(tokenizer.vocab_size, embed_dim)
    with open(os.path.join(raw_path, f'glove.6B.{embed_dim}d.txt'), 'rb') as f:
        for line in f:
            line = line.decode().split()
            word, vector = line[0], line[1:]
            word_id = tokenizer.convert_tokens_to_ids(word)
            if word_id != tokenizer.unk_token_id:
                embed_weights[word_id] = torch.tensor(list(map(float, vector)))

    torch.save(embed_weights, proc_file_path)
    return embed_weights


def get_probs(preds_column):
    preds_column = preds_column.apply(lambda x: x.replace("[", "").replace("]", "").split(','))
    preds_column = preds_column.apply(lambda x: [float(i) for i in x])
    preds_column = preds_column.apply(lambda x: np.exp(x) / np.sum(np.exp(x), axis=0))
    preds_column = preds_column.apply(lambda x: x[1])
    return preds_column


def get_domain_data_dir(domain_str):
    data_dir = Path(DATA_ROOT)
    if "amazon" in domain_str:
        is_balanced = domain_str.split('_amazon_')[0] == 'balanced'
        balanced_insert = 'balanced_' if is_balanced else ''
        domain_split = domain_str.split('amazon_')[1]
        target_data_path = data_dir / "amazon_reviews" / f"processed_{balanced_insert}amazon_reviews" / domain_split
    else:
        target_data_path = data_dir / "skytrax" / "airline"
    return target_data_path


def split_source_target_str(domain_pair_str):
    domain_pair_str_list = domain_pair_str.split('_')
    if domain_pair_str_list[0] == 'airline':
        source = 'airline_airline'
        target = '_'.join(domain_pair_str_list[2:])
    else:
        for i in range(2, len(domain_pair_str_list)):
            if domain_pair_str_list[i] == 'airline':
                target = 'airline_airline'
                source = '_'.join(domain_pair_str_list[:i])
                break
            elif domain_pair_str_list[i] == 'balanced' or domain_pair_str_list[i] == 'amazon':
                source = '_'.join(domain_pair_str_list[:i])
                target = '_'.join(domain_pair_str_list[i:])
                break
    return source, target


def order_df_by_another_df(target_df: pd.DataFrame, reference_df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    target_df = target_df.set_index(column_name)
    target_df = target_df.reindex(reference_df[column_name])
    target_df = target_df.reset_index()
    return target_df


def identity(arg):
    return arg
