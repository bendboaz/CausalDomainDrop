# CausalFeatures: Performance Estimation for Domain Adaptation in NLP

## Authors:

Boaz Ben-Dov, Eldar Abraham, Amir Feder, Roi Reichart

## Abstract:

Modern NLP models have impressive in-domain performance, but are very sensitive to domain shifts. One method for
improving performance on a new domain is to annotate new examples from it, but it is unclear how many examples would be
required for a given desired performance. Estimating the expected drop in performance across domains could guide the
annotation effort and also be used to evaluate the generalizability of a given model. Existing solutions rely on the
distance between domains and on model uncertainty, and do not account for inherent differences between the source and
target distributions. As they are dependent on correlation-based methods, they have trouble generalizing across domains.
As an alternative, we propose a method based on causal inference, where we estimate the causal effect of important
features and compare their distribution in the source and target domains. This method will result in an estimator for
the performance degradation, or the generalizability, of a model across domains.
 