import shap
from sklearn.tree import DecisionTreeClassifier

from constants import CONCEPT_CLASSIFIER_MAX_DEPTH

CLASSIFIER_TYPE = DecisionTreeClassifier


def get_concept_shap_values(df):
    pivot_cols = list(filter(lambda name: name.startswith('pivot'), df.columns))
    X = df.loc[:, pivot_cols].to_numpy()
    y = df['sentiment'].to_numpy()
    classifier = CLASSIFIER_TYPE(max_depth=CONCEPT_CLASSIFIER_MAX_DEPTH)
    classifier.fit(X, y)
    explainer = shap.TreeExplainer(classifier, feature_perturbation='tree_path_dependent', model_output='raw')
    # Should we use dev set for this check?
    explanation = explainer.shap_values(X, y)

    # Because fuck this library.
    assert isinstance(explanation, list) and len(explanation) == 2
    explanation = explanation[0]

    explanation = explanation.mean(axis=0)
    return explanation
