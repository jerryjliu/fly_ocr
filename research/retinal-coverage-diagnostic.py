"""Post-hoc coverage and validation-only input baselines; no model mutation.

Run from the repository root. Outputs go to reports/retinal-diagnostic.
The direct baselines bypass the circuit and do not measure a remapped circuit.
"""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from flyocr.vision.retina import sample_retina

out = Path("reports/retinal-diagnostic")
out.mkdir(parents=True, exist_ok=True)
uv = np.load("data/graph/uv.npy")
coverage = np.zeros((48, 48), bool)
for x, y in uv * 47:
    for yy in {int(np.floor(y)), int(np.ceil(y))}:
        for xx in {int(np.floor(x)), int(np.ceil(x))}:
            coverage[yy, xx] = True

train = np.load("data/glyphs-letters/train.npz")
val = np.load("data/glyphs-letters/validation.npz")
ink = 1 - train["images"].astype(np.float64) / 255
blank = np.full((48, 48), 255, np.uint8)
altered = blank.copy()
altered[~coverage] = 0
assert np.array_equal(sample_retina(blank, uv), np.ones(len(uv), np.float32))
assert np.array_equal(sample_retina(blank, uv), sample_retina(altered, uv))
protocol = {
    "purpose": "Exploratory diagnosis after observing PDF errors; no deployed model changes",
    "splits": "Existing training and validation only; no new held-out or PDF baseline scoring",
    "classifier": "StandardScaler fitted on training, then RidgeClassifier(solver='cholesky')",
    "alphas": [1., 10., 100., 1000.],
    "selection": "Best validation accuracy per input representation; results are validation-selected",
    "conditions": ["full_pixels", "current_retinal_samples", "uniform_retinal_control"],
    "uniform_control": "3335 fixed nearly uniform locations from a 58x58 grid; not an anatomical mapping",
    "caveat": "Direct input classifiers bypass the circuit; no expected gain for a remapped circuit is established",
}
(out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
result = {
    "retinal_samples": len(uv),
    "unique_locations": len(np.unique(uv, axis=0)),
    "quadrant_sample_counts": np.histogram2d(uv[:, 1], uv[:, 0], bins=[0, .5, 1])[0].astype(int).tolist(),
    "bilinear_supported_pixels": int(coverage.sum()),
    "total_pixels": coverage.size,
    "quadrant_pixel_coverage": [[float(coverage[:24, :24].mean()), float(coverage[:24, 24:].mean())],
                                [float(coverage[24:, :24].mean()), float(coverage[24:, 24:].mean())]],
    "training_ink_intensity_fraction_outside_support": float(ink[:, ~coverage].sum() / ink.sum()),
    "blank_samples_all_white": True,
    "changing_all_unsupported_pixels_leaves_samples_identical": True,
    "baseline_trials": {},
}
grid = np.stack(np.meshgrid(np.linspace(0, 1, 58), np.linspace(0, 1, 58)), axis=-1).reshape(-1, 2)
uniform_uv = grid[np.round(np.linspace(0, len(grid)-1, len(uv))).astype(int)].astype(np.float32)
with threadpool_limits(limits=4):
    for name in protocol["conditions"]:
        begin = time.perf_counter()
        def features(images):
            if name == "full_pixels":
                return images.reshape(len(images), -1).astype(np.float64) / 255
            locations = uv if name == "current_retinal_samples" else uniform_uv
            return np.array([sample_retina(im, locations) for im in images], dtype=np.float64)
        tx, vx = features(train["images"]), features(val["images"])
        scaler = StandardScaler().fit(tx)
        tx, vx = scaler.transform(tx), scaler.transform(vx)
        trials = []
        for alpha in protocol["alphas"]:
            model = RidgeClassifier(alpha=alpha, solver="cholesky").fit(tx, train["labels"])
            pred = model.predict(vx)
            letter_mask = val["labels"] >= 16
            trials.append({"alpha": alpha, "correct": int((pred == val["labels"]).sum()),
                           "n": len(pred), "accuracy": float((pred == val["labels"]).mean()),
                           "letters_accuracy": float((pred[letter_mask] == val["labels"][letter_mask]).mean())})
        result["baseline_trials"][name] = {"trials": trials, "selected": max(trials, key=lambda x: x["accuracy"]),
                                           "seconds": time.perf_counter()-begin}
        print(name, result["baseline_trials"][name]["selected"], flush=True)
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run_dir = Path("demo/examples/trained-heading")
run = json.loads((run_dir / "run.json").read_text())
# The seventh source glyph is E, which the recorded circuit predicts as 5.
event = run["events"][6]
glyph = np.array(Image.open(run_dir / event["image"]).convert("L"))
samples = sample_retina(glyph, uv)
assert np.allclose(samples, event["retina"], atol=1e-6)
weights = 1-glyph/255
result["example"] = {"event_id": event["id"], "source": "E", "prediction": event["character"],
                     "ink_intensity_fraction_outside_support": float(weights[~coverage].sum()/weights.sum())}
(out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.3), facecolor="#f5f7f9")
fig.suptitle("The dark patch is missing retinal coverage", x=.035, ha="left", fontsize=21, fontweight="bold", y=.97)
fig.text(.035, .84, "Same fixed map for every glyph. A blank white input reveals the hole.", fontsize=12, color="#485665")
axes[0].imshow(glyph, cmap="gray", vmin=0, vmax=255, extent=(0,1,1,0), interpolation="nearest")
axes[0].set_title("Normalized E", fontsize=13, loc="left", pad=12)
for ax, values, title in [(axes[1], np.ones(len(uv)), "Blank white input"), (axes[2], samples, "E at retinal locations")]:
    ax.set_facecolor("#05090c")
    ax.scatter(uv[:, 0], uv[:, 1], c=values, cmap="gray", vmin=0, vmax=1, s=2, marker="s", linewidths=0)
    ax.set_title(title, fontsize=13, loc="left", pad=12)
overlay = np.zeros((48, 48, 4))
overlay[~coverage] = [1., .4, .16, .85]
axes[3].imshow(glyph, cmap="gray", vmin=0, vmax=255, extent=(0,1,1,0), interpolation="nearest")
axes[3].imshow(overlay, extent=(0,1,1,0), interpolation="nearest")
axes[3].set_title("Orange = unsampled pixels", fontsize=13, loc="left", pad=12)
for ax in axes:
    ax.set(xlim=(0,1), ylim=(1,0), aspect="equal", xticks=[], yticks=[])
    for spine in ax.spines.values(): spine.set_color("#ccd3db")
axes[0].set_xlabel("48 × 48 input pixels", fontsize=10, labelpad=12)
axes[1].set_xlabel("Black void persists with no character", fontsize=10, labelpad=12)
axes[2].set_xlabel("Brightness samples, not firing rates", fontsize=10, labelpad=12)
axes[3].set_xlabel(f"{result['example']['ink_intensity_fraction_outside_support']:.0%} of this E's ink is never sampled", fontsize=10, labelpad=12)
fig.subplots_adjust(left=.035, right=.98, top=.73, bottom=.13, wspace=.17)
fig.savefig(out / "coverage.png", dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k != "baseline_trials"}, indent=2), flush=True)
