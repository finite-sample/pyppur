"""ProjectionPursuit is a scikit-learn transformer.

The README claimed a "Full scikit-learn compatible API" while the class
inherited from object, had no get_params/set_params, and could not be a
Pipeline step because fit_transform took no y.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from pyppur import ProjectionPursuit


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 6)
    return X, (X[:, 0] + X[:, 1] > 0).astype(int)


def _pp():
    return ProjectionPursuit(n_components=2, random_state=1, n_init=1)


def test_passes_the_full_estimator_check_suite():
    results = check_estimator(_pp(), on_fail=None)
    failed = [r["check_name"] for r in results if r["status"] == "failed"]
    assert failed == [], f"failing checks: {failed}"


def test_get_and_set_params_round_trip():
    pp = _pp()
    assert "alpha" in pp.get_params()
    assert pp.set_params(alpha=0.3).alpha == pytest.approx(0.3)


def test_clone_reproduces_the_unfitted_estimator(data):
    # clone() reads the constructor parameters back off the instance, so it
    # only works because fit() no longer mutates n_components.
    X, _ = data
    pp = _pp()
    pp.fit(X)
    fresh = clone(pp)
    assert fresh.get_params() == _pp().get_params()


def test_works_as_a_pipeline_step(data):
    X, y = data
    pipe = Pipeline([("pp", _pp()), ("lr", LogisticRegression())])
    assert 0.0 <= pipe.fit(X, y).score(X, y) <= 1.0
    assert np.isfinite(cross_val_score(pipe, X, y, cv=3).mean())


def test_works_inside_grid_search(data):
    X, y = data
    gs = GridSearchCV(
        Pipeline([("pp", _pp()), ("lr", LogisticRegression())]),
        {"pp__alpha": [0.1, 0.5]},
        cv=3,
    )
    gs.fit(X, y)
    assert gs.best_params_["pp__alpha"] in (0.1, 0.5)


def test_fitted_attributes_follow_the_convention(data):
    X, _ = data
    pp = _pp()
    pp.fit(X)
    assert pp.n_features_in_ == X.shape[1]
    assert isinstance(pp.n_iter_, int)


def test_transform_rejects_bad_input(data):
    X, _ = data
    pp = _pp()
    pp.fit(X)
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pp.transform(bad)
    with pytest.raises(ValueError, match="features"):
        pp.transform(X[:, :3])
