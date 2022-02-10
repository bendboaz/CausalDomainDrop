import statsmodels.stats.api as sms
import numpy as np


def ates(num_concepts: int = 6):
    return ['ATE_' + str(i) for i in range(num_concepts)]


def source_ates(num_concepts: int = 6):
    return [f'source_{ate}' for ate in ates(num_concepts)]


def target_ates(num_concepts: int = 6):
    return [f'target_{ate}' for ate in ates(num_concepts)]


def target_counts(num_concepts: int = 6):
    return ['target_counts_' + str(i) for i in range(num_concepts)]


def source_counts(num_concepts: int = 6):
    return ['source_counts_' + str(i) for i in range(num_concepts)]


def counts(num_concepts: int = 6):
    return source_counts(num_concepts) + target_counts(num_concepts)


def performance_ates(num_concepts: int = 6):
    return [f'performance_ATE_{i}' for i in range(num_concepts)]


def shap_features(num_concepts: int = 6):
    return [f'SHAP_{i}' for i in range(num_concepts)]


def source_target_diff(num_concepts: int = 6):
    return [f's_t_ratio_{i}' for i in range(num_concepts)]


def weighted_source_target_diff(num_concepts: int = 6):
    return [f'w_s_t_ratio_{i}' for i in range(num_concepts)]


def feature_cols(num_concepts: int = 6):
    return source_ates(num_concepts) + target_ates(num_concepts) + \
           ates(num_concepts) + counts(num_concepts) + shap_features(num_concepts) + \
           source_target_diff(num_concepts) + weighted_source_target_diff(num_concepts) + \
           performance_ates(num_concepts) + ['source_f1', 'source_acc']


def get_feature_sets(num_concepts: int = 6):
    baseline = ['source_f1', 'source_acc']
    all_but_shap = [feature for feature in feature_cols(num_concepts) if feature not in shap_features(num_concepts)]
    return {
        'baseline': baseline,
        'concept DF': counts(num_concepts),
        'baseline + concept DF': baseline + counts(num_concepts),
        'shap': shap_features(num_concepts),
        'shap + baseline': baseline + shap_features(num_concepts),
        'shap + baseline + concept DF': counts(num_concepts) + baseline + shap_features(num_concepts),
        'ates': ates(num_concepts),
        'ates + baseline': baseline + ates(num_concepts),
        'ates + baseline + concept DF': ates(num_concepts) + baseline + counts(num_concepts),
        'performance + baseline': baseline + performance_ates(num_concepts),
        'performance + baseline + concept DF': baseline + counts(num_concepts) + performance_ates(num_concepts),
        'all - baseline': [feature for feature in all_but_shap if feature not in baseline],
        'all - combinations': [feature for feature in all_but_shap if feature not in (source_target_diff(num_concepts) + weighted_source_target_diff(num_concepts))],
        'all - performance': [feature for feature in all_but_shap if feature not in (performance_ates(num_concepts))],
        'all': all_but_shap
    }


def confidence_intervals(a: np.ndarray):
    return sms.DescrStatsW(a).tconfint_mean(alpha=0.05)
