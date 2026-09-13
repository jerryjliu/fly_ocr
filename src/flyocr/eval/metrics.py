import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def classification(y, predicted, classes):
    y, predicted = np.asarray(y), np.asarray(predicted)
    correct = int((y == predicted).sum()); n = len(y)
    if not n: raise ValueError("Empty evaluation")
    p = correct/n; z = 1.96
    mid = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {"n": n, "correct": correct, "accuracy": p, "macro_f1": float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0)),
            "wilson_95": [float(mid-half), float(mid+half)],
            "interval_caveat": "Image-level binomial interval; shared fonts introduce dependence.",
            "confusion_matrix": confusion_matrix(y, predicted, labels=classes).tolist()}


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(ca != cb)))
        previous = current
    return previous[-1]
