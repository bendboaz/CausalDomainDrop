from itertools import product
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from Analysis.regression_ATE import get_regression_metrics, create_regression_dataset
from Analysis.analysis_utils import feature_cols, shap_features
from paths import EXPERIMENTS_ROOT


def get_results_dataframe(shap: bool = False, n_grams: str = 'UNI', num_concepts: int = 6) -> Tuple[pd.DataFrame, Dict]:
    result_cols = ['sort_rows', 'ates', 'counts', 'source_score', 'model', 'regression_r2', 'regression_rmse']
    results_df = pd.DataFrame(columns=result_cols)
    models = {}
    for sort_rows in [True, False]:
        X, y = create_regression_dataset(shap, n_grams, num_concepts, rows_sorted=sort_rows)
        for ates, counts, f1_score in product([True, False], repeat=3):
            if (not ates) and (not counts) and (not f1_score):
                continue
            for model in MODELS_TO_RUN:
                model_params = model.get_params()
                metrics = get_regression_metrics(model, X, y, num_concepts, counts, ates, f1_score, return_models=True,
                                                 features=[feature for feature in feature_cols(num_concepts)
                                                           if feature not in shap_features(num_concepts)],
                                                 **model_params)
                results_df = results_df.append(pd.DataFrame(
                    [[sort_rows, ates, counts, f1_score, type(model).__name__, metrics['r2'], -1 * metrics['rmse']]],
                    columns=result_cols), ignore_index=True)
                models[(sort_rows, ates, counts, f1_score, type(model).__name__)] = metrics['models']
    return results_df, models


def analyze_ate_effect(results_df: pd.DataFrame):
    results_by_ates = results_df.groupby(by='ates', as_index=False)
    with_ates_df: pd.DataFrame = results_by_ates.get_group(True)
    no_ates_df: pd.DataFrame = results_by_ates.get_group(False)
    score_columns = ['regression_rmse', 'regression_r2']
    ate_effect_columns = [f'ate_effect_{score}' for score in score_columns]
    with_ates_df = with_ates_df[with_ates_df['counts'] | with_ates_df['source_score']]
    with_ate_scores = pd.DataFrame.sort_values(with_ates_df,
                                               by=['sort_rows', 'counts', 'source_score', 'model', 'ates'], axis=0,
                                               ignore_index=True)
    no_ate_scores = pd.DataFrame.sort_values(no_ates_df, by=['sort_rows', 'counts', 'source_score', 'model', 'ates'],
                                             axis=0, ignore_index=True)
    ate_effects = pd.merge_ordered(with_ate_scores, no_ate_scores, on=['sort_rows', 'counts', 'source_score', 'model'],
                                   suffixes=['_ate', '_no_ate'])
    ate_effect_array = ate_effects[[f'{column}_ate' for column in score_columns]].to_numpy() - \
                       ate_effects[[f'{column}_no_ate' for column in score_columns]].to_numpy()
    ate_effects = ate_effects.assign(**{effect: ate_effect_array[:, i] for i, effect in enumerate(ate_effect_columns)})
    ate_effects = ate_effects.drop(['ates_ate', 'ates_no_ate'], axis=1)
    return ate_effects


def get_baseline_stats(shap: bool, n_grams: str, num_concepts: int = 6):
    experiment_dir = Path(EXPERIMENTS_ROOT)
    baseline_results_dir = experiment_dir / 'analysis' / 'regression_results' / 'baseline'
    baseline_results_dir.mkdir(exist_ok=True, parents=True)
    baseline_results_file = baseline_results_dir / f'baseline_{shap}_{n_grams}.csv'
    if baseline_results_file.is_file():
        return
    else:
        result_cols = ['sort_rows', 'counts', 'source_score', 'model', 'regression_r2', 'regression_rmse']
        results_df = pd.DataFrame(columns=result_cols)
        for sort_rows in [True, False]:
            X, y = create_regression_dataset(shap, n_grams, num_concepts, rows_sorted=sort_rows)
            for counts, f1_score in product([True, False], repeat=2):
                for model in MODELS_TO_RUN:
                    model_params = model.get_params()
                    metrics = get_regression_metrics(model, X, y, num_concepts, counts, False, f1_score,
                                                     return_models=True, features=feature_cols(num_concepts),
                                                     **model_params)
                    results_df = results_df.append(pd.DataFrame(
                        [[sort_rows, counts, f1_score, type(model).__name__, metrics['r2'],
                          -1 * metrics['rmse']]],
                        columns=result_cols), ignore_index=True)
        results_df.to_csv(baseline_results_file, index=False)
        return results_df, models


if __name__ == '__main__':
    CONFIGS_TO_RUN = [(True, 'UNI'), (True, 'KMEANS'), (False, 'UNI'), (False, 'KMEANS')]
    MODELS_TO_RUN = [LinearRegression(), Lasso(), RandomForestRegressor(n_estimators=20, max_depth=4)]
    results = {}
    experiment_dir = Path(EXPERIMENTS_ROOT)
    regression_results_dir = experiment_dir / 'analysis' / 'regression_results'
    regression_results_dir.mkdir(exist_ok=True, parents=True)
    models = {}
    track_model_results = not (regression_results_dir / 'regression_params_df.csv').is_file()
    for shap, n_grams in CONFIGS_TO_RUN:
        if (regression_results_dir / f'regression_results_{(shap, n_grams)}.csv').is_file() and not track_model_results:
            print(f'Already have results for {(shap, n_grams)}')
            results[(shap, n_grams)] = pd.read_csv(regression_results_dir / f'regression_results_{(shap, n_grams)}.csv',
                                                   index_col='idx')
            parameters_df = pd.read_csv(regression_results_dir / 'regression_params_df.csv')
        else:
            results[(shap, n_grams)], model_dict = get_results_dataframe(shap, n_grams)
            results[(shap, n_grams)].to_csv(regression_results_dir / f'regression_results_{(shap, n_grams)}.csv',
                                            index_label='idx')
            model_statistics = {}
            for model_type in map(lambda x: type(x).__name__, MODELS_TO_RUN):
                all_models_of_type = {key: value for key, value in model_dict.items() if key[-1] == model_type}
                if model_type in ['LinearRegression', 'Lasso']:
                    model_statistics.update({key: np.mean(
                        [np.concatenate([estimator.coef_, np.array([estimator.intercept_])]) for estimator in value],
                        axis=0
                    )
                        for key, value in all_models_of_type.items()})
                elif model_type == 'RandomForestRegressor':
                    model_statistics.update({key: np.mean(
                        [estimator.feature_importances_ for estimator in value],
                        axis=0
                    ) for key, value in all_models_of_type.items()})
            models[(shap, n_grams)] = model_statistics
    if track_model_results:
        models_flattened = {}
        for key, nested_dict in models.items():
            models_flattened.update({tuple(key) + tuple(nested_key): value for nested_key, value in nested_dict.items()})

        # Making each row into a series to trigger pandas auto padding
        models_flattened = {key: pd.Series(value) for key, value in models_flattened.items()}
        parameters_df = pd.DataFrame.from_dict(models_flattened, orient='index')
        # Flattening the MultiIndex:
        parameters_df = parameters_df.reset_index()
        parameters_df = parameters_df.rename({
            f'level_{i}': real_name
            for i, real_name in enumerate(
                ['shap', 'n_grams', 'sort_rows', 'ates', 'counts', 'f1_score']
            )})
        # models_df[['shap', 'n_grams', 'sort_rows', 'ates', 'counts', 'f1_score']] = models_df['index'].tolist()
        # models_df = models_df.drop(['index'], axis=1)

        parameters_df.to_csv(regression_results_dir / 'regression_params_df.csv', index=False)

    for shap, n_grams in CONFIGS_TO_RUN:
        get_baseline_stats(shap, n_grams)

    for key, value in results.items():
        print(key, ':')
        print(value)
        analyzed_df = analyze_ate_effect(value)
        print(analyzed_df)
        if not (regression_results_dir / f'ate_effects_{key}.csv').is_file():
            analyzed_df.to_csv(regression_results_dir / f'ate_effects_{key}.csv')
    print(parameters_df)
