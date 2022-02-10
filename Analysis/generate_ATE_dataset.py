import glob
import json
import os
from itertools import product, islice
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import filename_formats
from paths import DATA_ROOT as data_dir
from paths import EXPERIMENTS_ROOT as experiments_dir
from utils import get_tokenizer, get_probs, get_domain_data_dir, split_source_target_str, identity
from filename_formats import full_pivots_csv_format, ConceptType
from DataManagement.shap_concepts import get_concept_shap_values
from Analysis.analysis_utils import ates, source_ates, target_ates, counts, shap_features, performance_ates, \
    target_counts

experiments_dir = Path(experiments_dir)
data_dir = Path(data_dir)

all_pairs = glob.glob(str(experiments_dir) + os.sep + "per_sample" + os.sep + "*")
num_concepts = 6
tcs = ['TC_' + str(i) for i in range(num_concepts)]
cols = ['source', 'target', 'CC', 'shap', 'n_grams'] + source_ates(num_concepts) + \
       target_ates(num_concepts) + ates(num_concepts) + performance_ates(num_concepts) + tcs + \
       counts(num_concepts) + shap_features(num_concepts)

all_shap = [False]
all_ngrams = ['UNI', 'KMEANS']

for shap, n_grams in product(all_shap, all_ngrams):
    df = pd.DataFrame(columns=cols)
    n_pivots = 10 if n_grams == 'KMEANS' else 100
    C = n_pivots - 1
    shap_postfix = 'S' if shap else ''

    for domain_pair in all_pairs:
        if domain_pair.split(os.sep)[-1] == 'desktop.ini':
            continue
        domain_pair_str = domain_pair.split(os.sep)[-1]
        source, target = split_source_target_str(domain_pair_str)
        print(source, target, shap, n_grams)

        source_labels = pd.read_csv(get_domain_data_dir(source) / 'dev.csv',
                                    names=['id', 'reviewText', 'sentiment'], header=0)

        cur_ATES = []
        cur_source_ATEs = []
        cur_target_ATEs = []
        cur_performance_ATEs = []
        for T in range(num_concepts):
            file_MLM = f"bert-base-uncased_{n_pivots}pv_{n_grams}grams_T{T}{shap_postfix}_C{C}{shap_postfix}_MLM_1acc_4batch_0warm_0.01epsilon_0.0001lr_ftbatch128_ftinitlr0.0001_ftlrgamma0.001_ftepochs5_ftwd0.001_CF.csv"
            file_CF = f"bert-base-uncased_{n_pivots}pv_{n_grams}grams_T{T}{shap_postfix}_C{C}{shap_postfix}_CAUSALM_1acc_4batch_0warm_0.01epsilon_0.0001lr_ftbatch128_ftinitlr0.0001_ftlrgamma0.001_ftepochs5_ftwd0.001_CF.csv"

            df_MLM = pd.read_csv(domain_pair + "/" + file_MLM)
            df_CF = pd.read_csv(domain_pair + "/" + file_CF)

            # Sorting df_CF according to the columns (id, is_target) from df_MLM
            df_MLM['row_primary_key'] = df_MLM.apply(lambda row: str(row['id']) + '_' + str(row['is_target']), axis=1)
            df_CF['row_primary_key'] = df_CF.apply(lambda row: str(row['id']) + '_' + str(row['is_target']), axis=1)

            df_CF = df_CF.set_index('row_primary_key')
            df_CF = df_CF.reindex(index=df_MLM['row_primary_key'])
            df_CF = df_CF.reset_index()

            df_MLM['predictions_CF'] = df_CF['predictions']

            df_MLM['predictions_CF'] = get_probs(df_MLM['predictions_CF'])
            df_MLM['predictions'] = get_probs(df_MLM['predictions'])
            # df_MLM['ITE'] = abs(df_MLM['predictions_CF'] - df_MLM['predictions'])
            df_MLM['y_pred'] = (df_MLM['predictions'] > 0.5).astype(int)
            df_MLM['y_pred_CF'] = (df_MLM['predictions_CF'] > 0.5).astype(int)

            correct_labels = df_MLM[df_MLM['is_target'] == 0].merge(source_labels, on='id', how='left', left_index=True)
            correct_labels = correct_labels.sort_index()

            df_MLM['is_correct'] = -1
            df_MLM['is_correct_CF'] = -1
            df_MLM.loc[df_MLM['is_target'] == 0, 'is_correct'] = df_MLM.loc[df_MLM['is_target'] == 0, 'y_pred'] == correct_labels['sentiment']
            df_MLM.loc[df_MLM['is_target'] == 0, 'is_correct_CF'] = df_MLM.loc[df_MLM['is_target'] == 0, 'y_pred_CF'] == \
                                                                 correct_labels['sentiment']

            df_MLM['ITE'] = df_MLM['predictions_CF'] - df_MLM['predictions']
            df_MLM['performance_ITE'] = df_MLM['is_correct_CF'] - df_MLM['is_correct']
            cur_ATE = df_MLM['ITE'].mean()
            cur_performance_ATE = df_MLM[df_MLM['is_target'] == 0]['performance_ITE'].mean()
            source_ate = df_MLM[df_MLM['is_target'] == 0]['ITE'].mean()
            target_ate = df_MLM[df_MLM['is_target'] == 1]['ITE'].mean()
            cur_ATES.append(cur_ATE)
            cur_source_ATEs.append(source_ate)
            cur_target_ATEs.append(target_ate)
            cur_performance_ATEs.append(cur_performance_ATE)

        # Get source pivot phrases
        source_pivots_file_dir = get_domain_data_dir(source)
        with open(source_pivots_file_dir / f"pivot_names_{n_grams.lower()}_{n_pivots}.json") as json_file:
            source_pivots_list = json.load(json_file)

        # Get counts for source pivots in source texts
        source_pivots_filename = filename_formats.full_pivots_csv_format(
            'dev',
            n_pivots,
            ConceptType.str2concept(n_grams),
            shap=shap
        )
        source_pivot_counts = pd.read_csv(
            source_pivots_file_dir / source_pivots_filename,
            usecols=[f'pivot{i}' for i in range(num_concepts)]
        )
        source_pivot_counts = source_pivot_counts.mean(axis=0)

        # Get counts for source pivots in target texts
        target_data_path = get_domain_data_dir(target) / 'dev.csv'
        target_df = pd.read_csv(target_data_path, names=['id', 'reviewText', 'sentiment'])
        target_df = target_df.dropna(axis='index', how='any')
        tokenizer = get_tokenizer('bert-base-uncased')
        tokenized = tokenizer.batch_encode_plus(
            target_df['reviewText'],
            max_length=tokenizer.max_len,
            pad_to_max_length=True,
        )
        str_reviews = list(map(lambda seq: " ".join(map(str, seq)), tokenized['input_ids']))

        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), preprocessor=identity,
                                     ngram_range=(2, 2) if n_grams == 'bi' else (1, 1), binary=True)
        counts = vectorizer.fit_transform(str_reviews)
        vectorizer_idx2tokenizer_idx = vectorizer.get_feature_names()

        tokenizer_idx2vectorizer_idx = vectorizer.vocabulary_
        assert tokenizer_idx2vectorizer_idx is not None
        assert all(tokenizer_idx2vectorizer_idx[t_idx] == v_idx
                   for v_idx, t_idx in enumerate(vectorizer_idx2tokenizer_idx))

        if n_grams == 'kmeans':
            source_pivot_tokenizer_ids = {c_idx: list(map(str, tokenizer.convert_tokens_to_ids(words)))
                                          for c_idx, words in source_pivots_list.items()}
            source_pivot_vectorizer_ids = {
                c_idx: list(map(lambda tokenizer_id: tokenizer_idx2vectorizer_idx.get(tokenizer_id, 0), tokenizer_ids))
                for c_idx, tokenizer_ids in source_pivot_tokenizer_ids.items()}
        else:
            source_pivot_tokenizer_ids = list(map(str, tokenizer.convert_tokens_to_ids(source_pivots_list)))
            source_pivot_vectorizer_ids = list(map(
                lambda tokenizer_id: tokenizer_idx2vectorizer_idx.get(tokenizer_id, 0),
                source_pivot_tokenizer_ids
            ))

        target_pivots_count = np.zeros((len(target_df.index), num_concepts), dtype=int)
        counts = counts.astype(bool)
        if n_grams == 'kmeans':
            for c_idx, words in islice(source_pivot_vectorizer_ids.items(), num_concepts):
                target_pivots_count[:, c_idx] = (np.array(counts[:, words]
                                                          .sum(axis=1)
                                                          .astype(int))
                                                 .reshape(target_pivots_count.shape[0]))
        else:
            for pivot_idx, pivot_feature in islice(enumerate(source_pivot_vectorizer_ids), num_concepts):
                if pivot_feature is None:
                    continue
                pivot_column = counts[:, pivot_feature] \
                    .todense() \
                    .reshape((counts.shape[0],))
                target_pivots_count[:, pivot_idx] = pivot_column
        target_pivots_count = target_pivots_count.mean(axis=0)
        # cols = ['source', 'target', 'CC', 'shap', 'n_grams']+ates+tcs+counts
        if n_grams.lower() == 'kmeans':
            source_pivots_list = list(source_pivots_list.values())

        # Adding SHAP values for the source train data pivots on predicting the sentiment.
        source_train_pivots = pd.read_csv(
            source_pivots_file_dir / full_pivots_csv_format('train', n_pivots, ConceptType.str2concept(n_grams), shap),
            index_col=None
        )
        shap_values = get_concept_shap_values(source_train_pivots)[:num_concepts]

        df = df.append(pd.DataFrame([([source, target, source_pivots_list[C], shap, n_grams]
                                      + cur_source_ATEs + cur_target_ATEs
                                      + cur_ATES + cur_performance_ATEs +
                                      source_pivots_list[:num_concepts] +
                                      source_pivot_counts.tolist() + target_pivots_count.tolist() +
                                      shap_values.tolist())],
                                    columns=cols), ignore_index=True)

    shap_or_no_shap = 'shap' if shap else 'no_shap'
    df.to_csv(experiments_dir / "analysis" / f"ates_and_pivots_{shap_or_no_shap}_{n_grams}.csv")
