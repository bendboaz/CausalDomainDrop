from pathlib import Path
from typing import Tuple, Dict, Iterable, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_validate

from paths import EXPERIMENTS_ROOT as experiments_dir
from utils import get_probs, get_domain_data_dir, split_source_target_str, order_df_by_another_df
from Analysis.analysis_utils import feature_cols, source_ates, target_ates, ates, counts, shap_features, \
    source_target_diff, weighted_source_target_diff, performance_ates, source_counts, target_counts

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

experiments_dir = Path(experiments_dir)


def create_regression_dataset(shap: bool, n_grams: str, num_concepts: int = 6, rows_sorted: bool = False,
                              use_acc: bool = False)\
        -> Tuple[pd.DataFrame, pd.Series]:
    shap_or_no_shap = 'shap' if shap else 'no_shap'

    df_ates = pd.read_csv(experiments_dir / "analysis" / f"ates_and_pivots_{shap_or_no_shap}_{n_grams.upper()}.csv")
    df_ates['source_target'] = df_ates['source'] + '_' + df_ates['target']
    df_ates['performance_degradation'] = 0
    df_ates['performance_degradation_acc'] = 0
    df_ates['source_f1'] = 0
    df_ates['target_f1'] = 0
    df_ates['source_acc'] = 0
    df_ates['target_acc'] = 0

    all_pairs = df_ates['source_target'].unique()
    for pair in all_pairs:
        source, target = split_source_target_str(pair)

        source_df = pd.read_csv(get_domain_data_dir(source) / 'dev.csv',
                                names=['id', 'reviewText', 'sentiment'], header=0)
        target_df = pd.read_csv(get_domain_data_dir(target) / 'dev.csv',
                                names=['id', 'reviewText', 'sentiment'], header=0)
        orig_filename = f'{source}_{target}_ftbatch128_ftinitlr0.0001_ftlrgamma0.001_ftepochs5_ftwd0.001_orig.csv'
        cur_df = pd.read_csv(experiments_dir / 'per_sample' / f'{source}_{target}' / orig_filename, header=0)
        cur_df['y_prob'] = get_probs(cur_df['predictions'])
        cur_df['y_pred'] = cur_df['y_prob'].apply(lambda x: 1 if x > 0.5 else 0)

        temp_source_df = cur_df[cur_df['is_target'] == 0][['id', 'y_pred', 'predictions']]
        temp_source_df = order_df_by_another_df(temp_source_df, source_df, 'id')
        source_df[['y_pred', 'y_prob']] = temp_source_df[['y_pred', 'predictions']]
        source_f1 = f1_score(source_df['sentiment'].to_numpy(), source_df['y_pred'].to_numpy())
        source_acc = accuracy_score(source_df['sentiment'].to_numpy(), source_df['y_pred'].to_numpy())

        temp_target_df = cur_df[cur_df['is_target'] == 1][['id', 'y_pred', 'predictions']]
        temp_target_df = order_df_by_another_df(temp_target_df, target_df, 'id')
        target_df[['y_pred', 'y_prob']] = temp_target_df[['y_pred', 'predictions']]
        target_f1 = f1_score(target_df['sentiment'].to_numpy(), target_df['y_pred'].to_numpy())
        target_acc = accuracy_score(target_df['sentiment'].to_numpy(), target_df['y_pred'].to_numpy())

        performance_degradation = (source_f1 - target_f1)
        performance_degradation_acc = source_acc - target_acc
        df_ates.loc[df_ates['source_target'] == pair, 'performance_degradation'] = performance_degradation
        df_ates.loc[df_ates['source_target'] == pair, 'performance_degradation_acc'] = performance_degradation_acc
        df_ates.loc[df_ates['source_target'] == pair, 'source_f1'] = source_f1
        df_ates.loc[df_ates['source_target'] == pair, 'target_f1'] = target_f1
        df_ates.loc[df_ates['source_target'] == pair, 'source_acc'] = source_acc
        df_ates.loc[df_ates['source_target'] == pair, 'target_acc'] = target_acc

    df_ates = df_ates.assign(**{
        s_t_diff: df_ates[source_ate] - df_ates[target_ate]
        for s_t_diff, source_ate, target_ate in
        zip(source_target_diff(num_concepts), source_ates(num_concepts), target_ates(num_concepts))
    })

    df_ates = df_ates.assign(**{w_s_t_diff: df_ates[s_t_diff] * (df_ates[source_ate] / df_ates[target_ate])
                                for w_s_t_diff, s_t_diff, source_ate, target_ate in zip(
            weighted_source_target_diff(num_concepts),
            source_target_diff(num_concepts),
            source_ates(num_concepts),
            target_ates(num_concepts)
        )})

    X = df_ates[['source', 'target'] + feature_cols(num_concepts)]

    if rows_sorted:
        X_sorted_order = np.argsort(X[ates(num_concepts)].to_numpy(), axis=1)[:, ::-1]
        ates_sorted = np.take_along_axis(X[ates(num_concepts)].to_numpy(), X_sorted_order, axis=1)
        source_ates_sorted = np.take_along_axis(X[source_ates(num_concepts)].to_numpy(), X_sorted_order, axis=1)
        target_ates_sorted = np.take_along_axis(X[target_ates(num_concepts)].to_numpy(), X_sorted_order, axis=1)
        source_counts_sorted = np.take_along_axis(X[source_counts(num_concepts)].to_numpy(), X_sorted_order, axis=1)
        target_counts_sorted = np.take_along_axis(X[target_counts(num_concepts)].to_numpy(), X_sorted_order, axis=1)
        shaps_sorted = np.take_along_axis(X[shap_features(num_concepts)].to_numpy(), X_sorted_order, axis=1)
        diffs_sorted = np.take_along_axis(X[source_target_diff(num_concepts)].to_numpy(), X_sorted_order, axis=1)
        w_diffs_sorted = np.take_along_axis(X[weighted_source_target_diff(num_concepts)].to_numpy(), X_sorted_order, axis=1)
        performance_ates_sorted = np.take_along_axis(X[performance_ates(num_concepts)].to_numpy(), X_sorted_order, axis=1)
        X_sorted = pd.DataFrame(
            np.hstack([source_ates_sorted, target_ates_sorted, ates_sorted, source_counts_sorted, target_counts_sorted,
                       shaps_sorted, diffs_sorted, w_diffs_sorted, performance_ates_sorted]),
            X.index,
            source_ates(num_concepts) + target_ates(num_concepts) +
            ates(num_concepts) + counts(num_concepts) + shap_features(num_concepts) +
            source_target_diff(num_concepts) + weighted_source_target_diff(num_concepts) +
            performance_ates(num_concepts)
        )
        X_sorted[['source_f1', 'source_acc']] = X[['source_f1', 'source_acc']]
        X = pd.concat([X[['source', 'target']], X_sorted], axis=1)

    if use_acc:
        return X, df_ates['performance_degradation_acc']
    return X, df_ates['performance_degradation']


def get_one_domain_out_cv(X: pd.DataFrame, add_domain_name: bool = False) -> Iterable[Tuple[Iterable[int], Iterable[int]]]:
    folds = []
    all_domains = X['source'].unique()
    for domain in all_domains:
        X_train = X.index[(X['source'] != domain) & (X['target'] != domain)].tolist()
        X_test = X.index[(X['source'] == domain) | (X['target'] == domain)].tolist()
        fold = (X_train, X_test)
        if add_domain_name:
            fold += (domain,)
        folds.append(fold)
    return folds


def get_regression_metrics(model_instance, X: pd.DataFrame, y: pd.Series, num_concepts: int = 6,
                           with_counts: bool = True, with_ates: bool = True, with_source_f1: bool = True,
                           return_models: bool = False, features: List[str] = None,
                           **classifier_kwargs) -> Dict[str, Any]:

    if features is None:
        features = feature_cols(num_concepts)

    if not with_ates:
        for ate in ates(num_concepts) + source_ates(num_concepts) + target_ates(num_concepts):
            features.remove(ate)
    if not with_counts:
        for count in counts(num_concepts):
            features.remove(count)
    if not with_source_f1:
        features.remove('source_f1')

    cv_indices = get_one_domain_out_cv(X)
    X = X[features]
    model_class = type(model_instance)
    model = model_class(**classifier_kwargs)
    metrics = cross_validate(model, X, y, cv=cv_indices, scoring=['neg_root_mean_squared_error', 'r2'],
                             return_estimator=return_models)
    results = {
        'rmse': metrics['test_neg_root_mean_squared_error'].mean(),
        'r2': metrics['test_r2'].mean()
    }
    if return_models:
        results['models'] = metrics['estimator']
    return results
