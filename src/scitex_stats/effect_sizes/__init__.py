#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effect size computations for statistical tests.

This module provides functions to compute and interpret various effect size measures:

Parametric Effect Sizes:
- Cohen's d: Standardized mean difference for t-tests
- Eta-squared (η²): Proportion of variance explained in ANOVA

Non-parametric Effect Sizes:
- Cliff's delta (δ): Ordinal effect size
- Probability of superiority P(X>Y): Common language effect size
- Epsilon-squared (ε²): Non-parametric analog of eta-squared

CI-Derived Effect Sizes:
- effect_size_from_ci: Standardized effect size from a point estimate
  + CI/SE alone, for metrics with no raw samples (bootstrap/DeLong CIs)

Each effect size comes with an interpretation function following standard
guidelines (Cohen, 1988; McGraw & Wong, 1992).
"""

from ._cliffs_delta import cliffs_delta, interpret_cliffs_delta
from ._cohens_d import cohens_d, interpret_cohens_d
from ._effect_size_from_ci import effect_size_from_ci, interpret_effect_size_from_ci
from ._epsilon_squared import epsilon_squared, interpret_epsilon_squared
from ._eta_squared import eta_squared, interpret_eta_squared
from ._prob_superiority import interpret_prob_superiority, prob_superiority

__all__ = [
    # Parametric
    "cohens_d",
    "interpret_cohens_d",
    "eta_squared",
    "interpret_eta_squared",
    # Non-parametric
    "cliffs_delta",
    "interpret_cliffs_delta",
    "prob_superiority",
    "interpret_prob_superiority",
    "epsilon_squared",
    "interpret_epsilon_squared",
    # CI-derived (no raw samples required)
    "effect_size_from_ci",
    "interpret_effect_size_from_ci",
]
