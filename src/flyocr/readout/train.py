from __future__ import annotations
import warnings
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def fit_readout(train_x, train_y, val_x, val_y, strengths=(.001, .01, .1, 1., 10.)):
    scaler = StandardScaler().fit(train_x)
    train = scaler.transform(train_x)
    val = scaler.transform(val_x)
    trials, best = [], None
    for c in strengths:
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=0).fit(train, train_y)
        accuracy = float(model.score(val, val_y))
        trials.append({"C": c, "validation_accuracy": accuracy, "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught)})
        if best is None or accuracy > best[0]:
            best = (accuracy, model, c)
    _, model, strength = best
    payload = {"mean": scaler.mean_, "scale": scaler.scale_, "coef": model.coef_, "intercept": model.intercept_, "classes": model.classes_}
    return payload, {"C": strength, "trials": trials, "validation_accuracy": best[0]}


def probabilities(payload, x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(payload["mean"]) or not np.isfinite(x).all():
        raise ValueError("Expected finite neural feature matrix with model feature count")
    transform = str(payload.get("count_transform", "identity"))
    if transform == "sqrt":
        if np.any(x < 0): raise ValueError("Spike counts cannot be negative")
        x = np.sqrt(x)
    elif transform != "identity":
        raise ValueError("Unknown spike-count transform")
    standardized = (x-payload["mean"])/payload["scale"]
    if "hidden_coef" in payload:
        standardized = np.clip(standardized,-10,10)
        hidden = np.maximum(0,standardized @ payload["hidden_coef"].T + payload["hidden_intercept"])
        if "hidden2_coef" in payload:
            hidden = np.maximum(0, hidden @ payload["hidden2_coef"].T + payload["hidden2_intercept"])
        logits = hidden @ payload["coef"].T + payload["intercept"]
    else:
        logits = standardized @ payload["coef"].T + payload["intercept"]
    if logits.shape[1] == 1:
        logits = np.c_[np.zeros(len(logits)), logits[:, 0]]
    return softmax(logits, axis=1)


def predict(payload, x):
    return payload["classes"][np.argmax(probabilities(payload, x), axis=1)]
