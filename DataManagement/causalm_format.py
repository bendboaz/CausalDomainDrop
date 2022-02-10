import json
import os
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from transformers import PreTrainedTokenizer, BertTokenizer

from DataManagement.shap_concepts import get_concept_shap_values
from filename_formats import pivot_names_fileformat, full_pivots_csv_format, ConceptType
from paths import DATA_ROOT, GLOVE_PROC_DIR, GLOVE_RAW_DIR
from utils import prepare_glove_vectors, identity


def get_pivot_clusters(tokenizer, mi_values, vectorizer_idx2tokenizer_idx, n_clusters=10):
    embeddings = prepare_glove_vectors(tokenizer, GLOVE_RAW_DIR, GLOVE_PROC_DIR)
    intersection_indices = {idx: int(vectorizer_idx2tokenizer_idx[idx])
                            for idx in range(len(mi_values))
                            if vectorizer_idx2tokenizer_idx[idx] not in
                            tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values())}
    embeddings = embeddings[list(intersection_indices.values())]
    indexed_mi_values = [(tokenizer_idx, mi_values[vectorizer_idx])
                         for vectorizer_idx, tokenizer_idx in intersection_indices.items()]
    sorted_mi_values = [item for _, item in sorted(indexed_mi_values, key=itemgetter(0))]
    enhanced_embedding_mat = torch.cat([embeddings, torch.tensor(sorted_mi_values).reshape(-1, 1)], dim=1)
    clusterer = KMeans(n_clusters)
    cluster_assignment = clusterer.fit_predict(enhanced_embedding_mat)
    clusters = {c_idx: [word for word in range(len(cluster_assignment))
                        if cluster_assignment[word] == c_idx]
                for c_idx in range(n_clusters)}
    cluster_mi_vals = {c_idx: enhanced_embedding_mat[indices, -1].mean()
                       for c_idx, indices in clusters.items()}
    cluster_order = sorted(cluster_mi_vals.items(), key=itemgetter(1))
    cluster_order = map(itemgetter(0), cluster_order)
    clusters = {new_idx: clusters[old_idx] for new_idx, old_idx in enumerate(cluster_order)}
    return clusters


def compute_pivots(dataset_path, tokenizer, labels, n_pivots,
                   concept_type=ConceptType.UNIGRAM,
                   vectorizer_vocab=None):
    partition_path = os.path.join(dataset_path, 'train.csv')
    df = pd.read_csv(partition_path, usecols=['reviewText', 'sentiment'],
                     dtype={'reviewText': str, 'sentiment': int},
                     na_filter=False)
    df = df.rename(columns={'reviewText': 'review'})
    tokenized = tokenizer.batch_encode_plus(
        df['review'],
        max_length=tokenizer.max_len,
        pad_to_max_length=True,
    )

    str_reviews = list(map(lambda seq: " ".join(map(str, seq)), tokenized['input_ids']))

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), preprocessor=identity,
                                 ngram_range=(2, 2) if concept_type is ConceptType.BIGRAM else (1, 1), binary=True,
                                 vocabulary=vectorizer_vocab)
    counts = vectorizer.fit_transform(str_reviews)
    __vectorizer_idx2tokenizer_idx = vectorizer.get_feature_names()
    mi_values = mutual_info_classif(counts, labels, discrete_features=True)
    if concept_type is ConceptType.CLUSTER:
        chosen_tokens = get_pivot_clusters(tokenizer, mi_values,
                                           __vectorizer_idx2tokenizer_idx,
                                           n_clusters=n_pivots)
    else:
        chosen_tokens = np.argsort(mi_values)[-1:-(n_pivots + 1):-1]
        chosen_tokens = [__vectorizer_idx2tokenizer_idx[token]
                         for token in chosen_tokens]
        if concept_type is ConceptType.BIGRAM:
            chosen_tokens = [list(map(int, pair.split()))
                             for pair in chosen_tokens]

    return chosen_tokens


def create_concept_csv(texts, labels, tc=None, cc=None):
    if tc is None:
        tc = np.zeros(len(labels))

    if cc is None:
        cc = np.zeros(len(labels))

    assert len({len(texts), len(labels), len(tc), len(cc)}) == 1

    df_dict = {
        'review': texts,
        'label': labels,
        'TC': tc,
        'CC': cc
    }
    df = pd.DataFrame.from_dict(df_dict)
    return df


def make_pivots_csv(dataset_path: str, partition: str,
                    tokenizer: PreTrainedTokenizer, n_pivots: int = 100,
                    concept_type=ConceptType.UNIGRAM,
                    shap_sorting=False, write_pivots=True, read_pivots=True):
    if partition not in {'train', 'dev', 'test'}:
        raise ValueError(f'Illegal partition {partition} for '
                         f'path {dataset_path}')

    filepath = os.path.join(dataset_path, f'{partition}.csv')
    df = pd.read_csv(filepath, usecols=['reviewText', 'sentiment'],
                     dtype={'reviewText': str, 'sentiment': int},
                     na_filter=False)
    df = df.rename(columns={'reviewText': 'review'})
    tokenized = tokenizer.batch_encode_plus(
        df['review'],
        max_length=tokenizer.max_len,
        pad_to_max_length=True,
    )

    # reviews = tokenized['input_ids']
    str_reviews = list(map(lambda seq: " ".join(map(str, seq)), tokenized['input_ids']))

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), preprocessor=identity,
                                 ngram_range=(2, 2) if concept_type is ConceptType.BIGRAM else (1, 1), binary=True)
    counts = vectorizer.fit_transform(str_reviews)
    vectorizer_idx2tokenizer_idx = vectorizer.get_feature_names()

    tokenizer_idx2vecotrizer_idx = vectorizer.vocabulary_
    assert tokenizer_idx2vecotrizer_idx is not None
    assert all(tokenizer_idx2vecotrizer_idx[t_idx] == v_idx
               for v_idx, t_idx in enumerate(vectorizer_idx2tokenizer_idx))

    pivots_file = os.path.join(dataset_path,
                               pivot_names_fileformat(partition, n_pivots, concept_type, shap=shap_sorting))
    if read_pivots and os.path.isfile(pivots_file):
        with open(pivots_file, 'r') as f:
            pivots_names = json.load(f)

        if concept_type is ConceptType.CLUSTER:
            tokenizer_indices = {c_idx: tokenizer.convert_tokens_to_ids(words)
                                 for c_idx, words in pivots_names.items()}
        else:
            tokenizer_indices = list(map(tokenizer.convert_tokens_to_ids, pivots_names))

    else:
        tokenizer_indices = compute_pivots(dataset_path, tokenizer,
                                           df['sentiment'], n_pivots,
                                           vectorizer_vocab=tokenizer_idx2vecotrizer_idx,
                                           concept_type=concept_type)

        if concept_type is ConceptType.CLUSTER:
            token_names = {c_idx: tokenizer.convert_ids_to_tokens(idx_list)
                           for c_idx, idx_list in tokenizer_indices.items()}
        elif concept_type is ConceptType.BIGRAM:
            token_names = list(map(tokenizer.convert_ids_to_tokens, tokenizer_indices))
        else:
            token_names = tokenizer.convert_ids_to_tokens(tokenizer_indices)

    if concept_type is ConceptType.CLUSTER:
        counts = counts.astype(np.bool)
        pivot_counts = np.zeros((counts.shape[0], n_pivots))
        vectorizer_clusters = {c_idx: [tokenizer_idx2vecotrizer_idx.get(str(w_idx), None)
                                       for w_idx in indices]
                               for c_idx, indices in tokenizer_indices.items()}

        vectorizer_clusters = {int(key): [int(item) for item in val if item is not None]
                               for key, val in vectorizer_clusters.items()}

        for c_idx, cluster_components in vectorizer_clusters.items():
            pivot_counts[:, c_idx] = np.array(counts[:, cluster_components] \
                                              .sum(axis=1)
                                              .astype(np.int)) \
                .reshape((counts.shape[0]))
        pivots_list = list(vectorizer_clusters.values())
    else:
        if concept_type is ConceptType.BIGRAM:
            tokenizer_bigrams = (" ".join(map(str, pair)) for pair in tokenizer_indices)
            pivots_list = []
            for bigram in tokenizer_bigrams:
                if bigram in tokenizer_idx2vecotrizer_idx:
                    pivots_list.append(tokenizer_idx2vecotrizer_idx[bigram])
                else:
                    pivots_list.append(None)
        else:
            pivots_list = list(map(lambda k: tokenizer_idx2vecotrizer_idx.get(str(k)), tokenizer_indices))

        pivot_counts = np.zeros((counts.shape[0], n_pivots))
        for pivot_idx, pivot_feature in enumerate(pivots_list):
            if pivot_feature is None:
                continue
            pivot_column = counts[:, pivot_feature] \
                .todense() \
                .reshape((counts.shape[0],))
            pivot_counts[:, pivot_idx] = pivot_column

    pivot_col_names = [f'pivot{i}' for i in range(pivot_counts.shape[1])]
    pivots_df = pd.DataFrame(pivot_counts,
                             columns=pivot_col_names)
    merged_df = pd.concat([df, pivots_df], axis='columns')
    merged_df = merged_df.dropna(axis='index', how='any')

    if shap_sorting:
        shap_values = get_concept_shap_values(merged_df)
        correct_order = np.argsort(shap_values)
        rename_dict = {f'pivot{old}': f'pivot{new}' for new, old in enumerate(correct_order)}
        # TODO: MAKE SURE THIS RENAME WORKS, OR YOU MIGHT RUIN YOUR CONCEPT DATAFRAME.
        merged_df = merged_df.rename(rename_dict)

    if not os.path.isfile(pivots_file) and write_pivots:
        with open(pivots_file, 'w+') as f:
            json.dump(token_names, f)

    merged_df.to_csv(os.path.join(dataset_path,
                                  full_pivots_csv_format(partition, n_pivots, concept_type, shap=shap_sorting)),
                     index_label='id')

    return merged_df


def get_pivots_concept_csv(dataset_path: str, partition: str,
                           tokenizer: PreTrainedTokenizer,
                           n_pivots: int = 100,
                           concept_type=ConceptType.UNIGRAM, shap=False):
    full_pivots_filename = full_pivots_csv_format(partition, n_pivots, concept_type, shap)
    pivots_filepath = os.path.join(dataset_path, full_pivots_filename)
    if os.path.isfile(pivots_filepath):
        pivots_df = pd.read_csv(pivots_filepath)
    else:
        pivots_df = make_pivots_csv(dataset_path, partition, tokenizer,
                                    n_pivots, concept_type=concept_type,
                                    write_pivots=True, shap_sorting=shap)

    return pivots_df


if __name__ == '__main__':
    domain_path = os.path.join(DATA_ROOT, 'amazon_reviews',
                               'processed_amazon_reviews',
                               'Automotive')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    n_pivots = 100
    res_df = get_pivots_concept_csv(domain_path, 'train', bert_tokenizer,
                                    n_pivots, concept_type=ConceptType.UNIGRAM)
    print(f'Got df with {len(res_df.index)} rows, '
          f'and {len(res_df.columns)} columns.')
    print(res_df)
