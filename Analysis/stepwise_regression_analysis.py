from pathlib import Path

import pandas as pd

from Analysis.ForwardStepwiseOLS import ForwardStepwiseOLS
from Analysis.regression_ATE import create_regression_dataset
from paths import EXPERIMENTS_ROOT
from Analysis.analysis_utils import feature_cols, ates, source_ates, target_ates, shap_features, counts


def get_stepwise_regression_features(X: pd.DataFrame, y: pd.Series, n_features: int = 15):
    model = ForwardStepwiseOLS(n_features, add_intercept=True)
    model.fit(X, y)

    feature_order = model.best_predictors
    feature_weights = model.best_model_fwd.params

    def get_feature_name(feature_idx: int) -> str:
        if feature_idx == len(X.columns) - 1:
            return 'intercept_term'
        return X.columns[feature_idx]

    feature_names = list(map(get_feature_name, feature_order))
    return feature_names, feature_weights


def main():
    CONFIGS_TO_RUN = [(True, 'UNI'), (True, 'KMEANS'), (False, 'UNI'), (False, 'KMEANS')]
    N_FEATURES = 15

    nonfeature_columns = ['source', 'target', 'source_target', 'target_f1', 'performance_degradation']
    more_to_drop = shap_features(6)

    results = {}
    experiment_dir = Path(EXPERIMENTS_ROOT)
    stepwise_results_dir = experiment_dir / 'analysis' / 'stepwise_regression_results'
    stepwise_results_dir.mkdir(parents=True, exist_ok=True)

    for shap, n_grams in CONFIGS_TO_RUN:
        for sort_rows in [True, False]:
            X_full, y = create_regression_dataset(shap, n_grams, rows_sorted=sort_rows)
            X = X_full.drop(nonfeature_columns + more_to_drop, axis=1, errors='ignore')
            best_features, feature_weights = get_stepwise_regression_features(X, y, n_features=N_FEATURES)
            results[(shap, n_grams, sort_rows)] = best_features + feature_weights.tolist()

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.reset_index()
    results_df[['shap', 'n_grams', 'sort_rows']] = results_df['index'].tolist()
    results_df = results_df.drop(['index'], axis=1)
    # results_df = results_df.rename({f'level_{i}': real_name for i, real_name in enumerate(['shap', 'n_grams', 'sort_rows'])})
    print(results_df)
    results_df.to_csv(stepwise_results_dir / 'best_features.csv', index=False)


if __name__ == '__main__':
    main()
