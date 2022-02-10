import os
from collections import OrderedDict, Counter
from itertools import accumulate
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BertTokenizer

from paths import DATA_ROOT


class BaseDomainSentimentDataset(Dataset):
    """
    Every data tuple is of the form (input_ids, attention_masks, label).
    """

    def __init__(self, dataset_root: str, domain_name: str, partition: str,
                 tokenizer: PreTrainedTokenizer, with_ids=False) -> None:
        super().__init__()
        if domain_name not in os.listdir(dataset_root):
            raise ValueError(f"No such domain '{domain_name}'.")

        domain_root = os.path.join(dataset_root, domain_name)

        if f"{partition}.csv" not in os.listdir(domain_root):
            raise ValueError(f"{partition} is not a valid partition "
                             f"(only train/dev/test allowed).")

        self.with_ids = with_ids
        df_path = os.path.join(domain_root, f"{partition}.csv")
        self.reviews = pd.read_csv(df_path,
                                   dtype={'reviewText': str, 'sentiment': int},
                                   na_filter=False)
        self.reviews.columns = ['sample_id', 'reviewText', 'sentiment']
        self.tokenizer = tokenizer
        possible_labels = sorted(self.reviews["sentiment"].unique())
        self.tag2ix = {label: index
                       for index, label in
                       enumerate(possible_labels)}

        data = self.preprocess(self.reviews, 'reviewText', 'sentiment')
        self.input_ids = data['input_ids']
        self.token_type_ids = data['token_type_ids']
        self.attention_mask = data['attention_mask']
        self.labels = torch.tensor(self.reviews['sentiment'].to_list())
        self.sample_ids = torch.tensor(self.reviews['sample_id'].to_list())

    def preprocess(self, dataset: pd.DataFrame,
                   text_column: str, label_column: str) -> pd.DataFrame:
        new_dataset = dataset
        # Index the different possible labels:
        new_dataset[label_column] = (new_dataset[label_column]
                                     .apply(self.tag2ix.__getitem__))

        data = self.tokenizer.batch_encode_plus(
            new_dataset[f'{text_column}'],
            add_special_tokens=True,
            max_length=self.tokenizer.max_len,
            pad_to_max_length=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        return data

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, item):
        sample_ids = (self.sample_ids[item],) if self.with_ids else tuple()
        x, y = ((self.input_ids[item],
                 self.attention_mask[item],
                 self.token_type_ids[item]),
                OrderedDict(sentiment=self.labels[item]))
        return (sample_ids, x, y) if self.with_ids else (x, y)

    @staticmethod
    def get_dataset(ds_type: str, domain: str, partition: str,
                    tokenizer: PreTrainedTokenizer, with_ids=False):
        ds_class = NAME2DATASET[ds_type]
        datasets_with_domain = {'amazon', 'balanced_amazon', 'he_small', 'he_large', 'airline'}
        if ds_type in datasets_with_domain:
            return ds_class(domain, partition, tokenizer, with_ids=with_ids)
        else:
            return ds_class(partition, tokenizer, with_ids=with_ids)

    @staticmethod
    def get_dataset_root_path(domain_type, domain_name):
        ds_class = NAME2DATASET[domain_type]
        return os.path.join(ds_class.get_dataset_root(), domain_name)

    @classmethod
    def get_dataset_root(cls):
        raise NotImplementedError()

    @classmethod
    def get_allowed_domains(cls):
        dataset_root = cls.get_dataset_root()
        return list(filter(
            lambda dom: os.path.isdir(os.path.join(dataset_root, dom)),
            os.listdir(dataset_root)
        ))


class AmazonReviewsDomainDataset(BaseDomainSentimentDataset):
    @classmethod
    def get_dataset_root(cls):
        return os.path.join(DATA_ROOT, 'amazon_reviews',
                            'processed_amazon_reviews')

    def __init__(self, domain: str, partition: str,
                 tokenizer: PreTrainedTokenizer, **kwargs):
        amazon_proc_dir = self.get_dataset_root()
        super(AmazonReviewsDomainDataset, self).__init__(
            amazon_proc_dir,
            domain,
            partition,
            tokenizer,
            **kwargs
        )


class BalancedAmazonReviewsDomainDataset(BaseDomainSentimentDataset):
    @classmethod
    def get_dataset_root(cls):
        return os.path.join(DATA_ROOT, 'amazon_reviews',
                            'processed_balanced_amazon_reviews')

    def __init__(self, domain: str, partition: str,
                 tokenizer: PreTrainedTokenizer, **kwargs):
        amazon_proc_dir = self.get_dataset_root()
        super(BalancedAmazonReviewsDomainDataset, self).__init__(
            amazon_proc_dir,
            domain,
            partition,
            tokenizer,
            **kwargs
        )


class HeReviewDomainDataset(BaseDomainSentimentDataset):
    @staticmethod
    def get_metaroot():
        return os.path.join(DATA_ROOT, "Amazon_Sentiment_Analysis_Small_Large")

    @classmethod
    def get_dataset_root(cls):
        Warning("Using HeReviewDomainDataset.get_dataset_root is deprecated, "
                "use HeReviewSmall or HeReviewLarge")
        raise NotImplementedError()

    def __init__(self, dataset_root: str, domain: str, size: str,
                 partition: str, tokenizer: PreTrainedTokenizer, **kwargs):
        super(HeReviewDomainDataset, self).__init__(
            dataset_root,
            domain,
            partition,
            tokenizer,
            **kwargs
        )
        self.size = size


class HeReviewSmall(HeReviewDomainDataset):
    @classmethod
    def get_dataset_root(cls):
        return os.path.join(cls.get_metaroot(), 'small')

    def __init__(self, domain: str, partition: str,
                 tokenizer: PreTrainedTokenizer, **kwargs):
        super(HeReviewSmall, self).__init__(
            self.get_dataset_root(),
            domain,
            'small',
            partition,
            tokenizer,
            **kwargs
        )


class HeReviewLarge(HeReviewDomainDataset):
    @classmethod
    def get_dataset_root(cls):
        return os.path.join(cls.get_metaroot(), 'large')

    def __init__(self, domain: str, partition: str,
                 tokenizer: PreTrainedTokenizer, **kwargs):
        super(HeReviewLarge, self).__init__(
            self.get_dataset_root(),
            domain,
            'large',
            partition,
            tokenizer,
            **kwargs)


class AirlineReviewsDataset(BaseDomainSentimentDataset):
    @classmethod
    def get_dataset_root(cls):
        return os.path.join(DATA_ROOT, 'skytrax')

    def __init__(self, domain, partition, tokenizer, **kwargs):
        super(AirlineReviewsDataset, self).__init__(
            self.get_dataset_root(),
            domain,
            partition,
            tokenizer,
            **kwargs
        )


class DomainRecognitionDataset(Dataset):
    """
    Every data tuple is of the form (input_ids, attention_masks, label).
    """

    def __init__(self, source_type: str, source_domain: Optional[str],
                 target_type: str, target_domain: Optional[str],
                 partition: str, tokenizer: PreTrainedTokenizer = None) -> None:
        super(DomainRecognitionDataset, self).__init__()
        self.domain_names = [(source_type, source_domain),
                             (target_type, target_domain)]

        for d_type, domain in self.domain_names:
            d_class = NAME2DATASET[d_type]
            d_root = d_class.get_dataset_root()
            if domain is not None:
                if domain not in os.listdir(d_root):
                    raise ValueError(f"No such domain '{domain}'.")
                domain_root = os.path.join(d_root, domain)
                if f"{partition}.csv" not in os.listdir(domain_root):
                    raise ValueError(f"{partition} is not a valid partition.")

        self.tokenizer = tokenizer

        all_domains = [pd.read_csv(
            os.path.join(
                NAME2DATASET[d_type].get_dataset_root(),
                d_domain if d_domain is not None else '',
                f"{partition}.csv"
            ),
            usecols=['reviewText'],
            dtype={'reviewText': str},
            na_filter=False
        )
            for d_type, d_domain in self.domain_names]

        for ind, domain in enumerate(all_domains):
            domain['domain'] = np.ones(len(domain.index)) * ind

        reviews = pd.concat(all_domains, axis=0, ignore_index=True)

        self.tag2ix = {label: index
                       for index, label in
                       enumerate(sorted(reviews["domain"].unique()))}

        tokenized_data = self.preprocess(reviews, 'reviewText', 'domain')

        self.reviews_tensor = tokenized_data['input_ids']
        self.masks_tensor = tokenized_data['attention_mask']
        self.domain_tensor = torch.tensor(reviews['domain'].to_list())

    def preprocess(self, dataset: pd.DataFrame, text_column: str,
                   label_column: str) -> pd.DataFrame:
        new_dataset = dataset
        # Index the different possible labels:
        new_dataset[label_column] = (new_dataset[label_column]
                                     .apply(self.tag2ix.__getitem__))

        # Tokenize the reviews if a tokenizer was provided:
        tokenized_data = self.tokenizer.batch_encode_plus(
            new_dataset[text_column],
            add_special_tokens=True,
            max_length=self.tokenizer.max_len,
            pad_to_max_length=True,
            return_tensors='pt',
            return_attention_mask=True,
        )

        return tokenized_data

    def __len__(self):
        return self.reviews_tensor.shape[0]

    def __getitem__(self, item):
        review = self.reviews_tensor[item]
        attention_mask = self.masks_tensor[item]
        domain = self.domain_tensor[item]

        return (review, attention_mask), domain

    def domain_counts(self):
        counts = Counter(self.domain_tensor.tolist())
        return counts


class MultiTaskDataset(Dataset):
    def __init__(self, ignore_index=-1, **datasets):
        super(MultiTaskDataset, self).__init__()

        self.ignore_index = ignore_index
        self.tasks = sorted(datasets.keys())
        self.datasets = OrderedDict(iter((task, datasets[task])
                                         for task in self.tasks))
        lengths = [len(ds) for ds in self.datasets.values()]
        self.borders = [0] + list(accumulate(lengths))

    def __len__(self):
        return self.borders[-1]

    def __getitem__(self, item):
        sample = None
        labels = OrderedDict()

        for ind, (task, border) in enumerate(zip(self.tasks, self.borders[1:])):
            if item < border and sample is None:
                input, task_labels = self.datasets[task][item - self.borders[ind]]
                if not isinstance(task_labels, dict):
                    task_labels = OrderedDict({task: task_labels})

                for task_label, item in task_labels.items():
                    if not isinstance(item, torch.Tensor):
                        item = torch.tensor(item)
                    labels[task_label] = item

                input_ids = input[0]
                input_shape = input_ids.shape
                if len(input) < 3:
                    # Missing token type ids
                    token_type_ids = torch.ones(input_shape, dtype=torch.long)
                else:
                    token_type_ids = input[2]

                if len(input) < 2:
                    # Missing attention masks:
                    attention_mask = torch.ones(input_shape, dtype=torch.long)
                else:
                    attention_mask = input[1]

                sample = (input_ids, attention_mask, token_type_ids)
            else:
                labels[task] = torch.tensor(self.ignore_index)

        return sample, labels


NAME2DATASET = {'amazon': AmazonReviewsDomainDataset,
                'balanced_amazon': BalancedAmazonReviewsDomainDataset,
                'he_small': HeReviewSmall,
                'he_large': HeReviewLarge,
                'airline': AirlineReviewsDataset}

if __name__ == "__main__":
    # Draft testing
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    mydataset = BaseDomainSentimentDataset("Cell_Phones_and_Accessories",
                                           "train", bert_tokenizer)
    print(mydataset[500])
