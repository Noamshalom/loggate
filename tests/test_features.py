import math

import numpy as np

from loggate.features import N_EVENT_TYPES, N_FEATURES, featurize, featurize_many


def test_shape_and_length_feature():
    x = featurize([5, 5, 22])
    assert x.shape == (N_FEATURES,)
    assert math.isclose(x[N_EVENT_TYPES], math.log(3))


def test_counts_are_log1p():
    x = featurize([5, 5, 5, 22])
    assert math.isclose(x[5], math.log1p(3))
    assert math.isclose(x[22], math.log1p(1))
    assert x[7] == 0.0


def test_unknown_event_ids_are_ignored_not_fatal():
    x = featurize([5, 999, -1])
    assert math.isclose(x[5], math.log1p(1))
    assert math.isclose(x[N_EVENT_TYPES], math.log(3))   # length still counts them


def test_empty_session_does_not_crash():
    x = featurize([])
    assert x[N_EVENT_TYPES] == 0.0 and not np.isnan(x).any()


def test_featurize_many_stacks():
    X = featurize_many([[5], [5, 22]])
    assert X.shape == (2, N_FEATURES)
    assert featurize_many([]).shape == (0, N_FEATURES)
