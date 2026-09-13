---
date: 2026-09-12T17:11:00-07:00
git_commit: 20849477963f12a9beaa29d96f949a8b6fb4d33b
branch: main
repository: 2026_09_12_fly_parsing_pdfs
topic: "Retinal coverage hole and poor character recognition"
tags: [research, codebase, experiment]
status: complete
last_updated: 2026-09-12
last_updated_by: codex
---

# Research: Retinal coverage and accuracy

## Research Question

Does the large lower-left dark patch in the demo affect recognition, is it an intentional property of the model, and how could character accuracy improve?

## Summary

The patch is an area with few or no sample locations under this implementation's approximate retinal projection. The demo paints missing locations black, making them visually similar to dark character strokes. It does not represent a black occluder fed into neurons, and it has not been validated as a biological blind spot. The fixed mapping is an experimental choice, not a requirement that the missing image coverage must be preserved.

The loss is substantial: 51.1% of total ink intensity across normalized training glyphs lies on pixels outside the bilinear sampler's support. A validation-only comparison using identical classifier families supports an input bottleneck: 70.4% on current retinal samples versus 83.6% on evenly distributed samples. These direct classifiers bypass the circuit; they do not establish the accuracy of a remapped fly circuit.

## Detailed Findings

### What the map and display represent

`src/flyocr/data/connectome.py:52` maps retained R1–R6 inputs to the strongest connected annotated L1/L2/L3 hexagonal column. It converts those coordinates into a plane, independently normalizes each eye by its coordinate extrema, and places the two eyes across overlapping portions of the image. The left eye spans horizontal coordinates 0–0.6, and the right spans 0.4–1 with a horizontal flip. This is an engineering projection derived from circuit annotations.

The resulting graph has 3,335 retinal inputs but only 825 distinct image coordinates. Repeated coordinates are repeated samples, not additional spatial coverage.

`src/flyocr/vision/retina.py:21` bilinearly samples grayscale image brightness at those fixed locations. `viewer/lib/replay.ts:39` paints a black rectangle and draws those samples on it. Empty regions remain black even for an entirely white image.

The panel shows brightness before the sensory transform, not retinal firing rates. The letter checkpoint uses `invert: true`; `src/flyocr/brain/model.py:119` inverts brightness, applies a temporal response and a saturating current transform. Therefore a dark sampled stroke is not equivalent to an inactive input. A black area with no sample is missing coverage, not an inhibitory stimulus.

### Measured coverage

The diagnostic enumerates the input pixels contributing to each bilinear sample. At least one sample depends on 1,198 of 2,304 input pixels (52.0%). Coverage within each 24×24 quadrant is:

| Quadrant | Pixel coverage | Retinal inputs |
|---|---:|---:|
| Upper left | 69.8% | 1,069 |
| Upper right | 74.8% | 1,465 |
| Lower left | 12.7% | 79 |
| Lower right | 50.7% | 722 |

Using darkness `1 − gray/255` as ink intensity, 51.1% of training ink lies outside sampling support. This measures intensity-weighted omitted pixels, not the percentage of classes that are impossible to identify or an accuracy ceiling. Shape correlations can still allow predictions from remaining strokes.

Changing every unsupported pixel in a white image to black produces exactly identical sample arrays. All samples for the unmodified white image equal one. These checks establish that the persistent dark void is a missing-location display effect and that the corresponding missing pixel information cannot reach the circuit.

For the seventh glyph in the heading demo (source E, prediction 5), 42.5% of ink is outside support, including much of the bottom bar. Recomputed samples match the recorded samples. The figure is [coverage.png](../reports/retinal-diagnostic/coverage.png).

### Input recognition comparison

The diagnostic trains StandardScaler plus RidgeClassifier independently for each representation using the existing 6,528 training glyphs and 1,088 validation glyphs over 68 classes. The same regularization choices (1, 10, 100, 1000) are evaluated for each; all select 1000. Training and solves use float64. Neither the circuit nor its deployed readout is modified.

| Input to direct classifier | Correct / validation glyphs | Overall accuracy | Letters only |
|---|---:|---:|---:|
| Full 48×48 pixels | 907/1,088 | 83.4% | 84.1% |
| Current 3,335 retinal samples | 766/1,088 | 70.4% | 67.9% |
| 3,335 nearly uniform sample locations | 910/1,088 | 83.6% | 84.1% |

The uniform control uses fixed evenly selected locations from a 58×58 grid. It is not an anatomical mapping. These are exploratory, validation-selected results, not new held-out test results, and the classifier differs from the deployed neural readout. The 13.2-point difference between sampling conditions must not be presented as a measured improvement to the fly model.

An earlier, separate digit benchmark also found 99.0% for a pixel linear classifier, 91.8% for a retinal-sample linear classifier, and 89.0% for the circuit readout (`docs/results.md:7`). Its alphabet, data, and head differ from the current letter experiment.

### Other potential limitations

The circuit is frozen rather than trained for character separation. Its readout sees four bins of activity from 1,024 selected downstream cells. Training contains only 96 examples per class across six font families, with separate validation and test families (`docs/letters.md:16`). Real PDF recognition also depends on segmentation and line normalization. The experiment has not isolated the respective error contributions of these factors.

Frequent held-out confusions include E→F (9), D→P (10), and Q→O (12). Their distinguishing strokes are consistent with a lower-region coverage problem, but a specific causal attribution for each confusion requires a changed-input circuit experiment. Case ambiguity and absent word context can also matter.

### Improvement options requested by the user

1. Audit and improve glyph-to-retina coverage first. Compare better glyph placement/scale, several translated views with the existing map, and a more complete projection. Multiple views can retain the existing internal wiring and retinal coordinates while exposing previously omitted strokes, at additional inference cost.
2. Recompute circuit responses and retrain the external readout for any changed stimulus representation. Compare on validation data and then a fresh held-out set; do not expect the current head to work unchanged on a new input distribution.
3. After coverage, examine broader training fonts, normalization jitter, feature selection, and readout capacity. The value of these changes remains to be measured.

Painting the display background white or interpolating the preview cannot recover information in the actual input channels. Word correction could improve final text but would be a separate stage from character recognition.

## Code References

- `src/flyocr/data/connectome.py:52` — approximate retinal projection.
- `src/flyocr/vision/retina.py:21` — bilinear brightness sampling.
- `src/flyocr/brain/model.py:119` — inversion, temporal response, and input current.
- `viewer/lib/replay.ts:39` — black background and brightness dots.
- `artifacts/letters/model-card.json` — fixed mapping, readout size, and inverted input.
- `docs/letters.md:16` — corpus size and font split.
- `research/retinal-coverage-diagnostic.py` — reproducible coverage and direct-input comparison.
- `reports/retinal-diagnostic/protocol.json` — diagnostic experiment scope.
- `reports/retinal-diagnostic/results.json` — exact measurements and all regularization trials.

## Architecture Notes

Missing sample locations, dark sampled pixels, and low neuron firing are different quantities. The original preview visually conflates the first two and does not display the third in its retinal panel. The input coverage defect precedes the frozen circuit and cannot be repaired by a downstream decoder using information that never reaches it.

## Open Questions

- How much will improved input coverage raise accuracy after retraining the circuit readout?
- Which projection or presentation scheme provides good coverage while preserving the intended anatomical interpretation?
- How much of the remaining error is caused by normalization, font generalization, feature selection, and simplified circuit dynamics?

No production mapping, checkpoint, or demo was changed during this diagnosis.
