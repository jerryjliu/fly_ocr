# Measured results

All primary results use a frozen, complete retained MaleCNS graph with a trained linear readout. Configuration selection used training/validation data only. These are bounded synthetic-font and single-column experiments.

## Held-out characters

| Experiment | Correct / evaluated | Accuracy |
|---|---:|---:|
| Fly circuit: digits | 445 / 500 | 89.0% |
| Fly circuit: numeric alphabet | 751 / 800 | 93.9% |
| Raw pixels: linear | 495 / 500 | 99.0% |
| Retinal samples: linear | 459 / 500 | 91.8% |
| Image CNN reference | 500 / 500 | 100.0% |
| Shuffled training labels | 52 / 500 | 10.4% |
| Mismatched test images | 53 / 500 | 10.6% |
| Blank input, original head | 50 / 500 | 10.0% |

Digit accuracy is 89.0%, with an image-level 95% Wilson interval of 86.0%–91.5%. The two test families are unseen during training; image dependence within families limits that interval. The proposed 90% digit gate was missed. The full confusion matrices, per-font breakdowns, validation trials and denominators are in [digit recognition](../artifacts/digits/recognition.json) and [numeric recognition](../artifacts/numeric/recognition.json).

## Complete selected PDF column

**18/19 exact cells (94.7%); character error rate 0.8%.** All 120 characters and 19 original cells remain in the evaluation. Coverage is 100%; automatic glyph counts agree for 19/19 cells. The proposed 80% exact-cell gate was met. No correction pass was used.

| Cell | Source truth | Raw prediction | Exact |
|---|---|---|---|
| r01 | `$63,946` | `$63,946` | yes |
| r02 | `217,778` | `217,778` | yes |
| r03 | `281,724` | `281,724` | yes |
| r04 | `13,501` | `13,501` | yes |
| r05 | `74,330` | `74,330` | yes |
| r06 | `87,831` | `87,831` | yes |
| r07 | `193,893` | `193,893` | yes |
| r08 | `32,488` | `32,488` | yes |
| r09 | `25,654` | `25,654` | yes |
| r10 | `7,223` | `7,223` | yes |
| r11 | `128,528` | `128,528` | yes |
| r12 | `(4,901)` | `(4,901)` | yes |
| r13 | `123,627` | `123,627` | yes |
| r14 | `21,795` | `21,795` | yes |
| r15 | `$101,832` | `$101,832` | yes |
| r16 | `$13.70` | `$13.70` | yes |
| r17 | `$13.64` | `$13.64` | yes |
| r18 | `7,433` | `7,433` | yes |
| r19 | `7,465` | `7.465` | no |

This is one manually selected numeric column. Automatic glyph boxes were visually checked; no independent manually boxed recognition benchmark was run. Missing/extra punctuation counts as an error. The [evaluation report](../reports/pdf-crop.json) links the original run and truth checksums.

## Wiring comparison

Randomized graphs retain exact edge in/out degrees and each source's signed weight list. They produce much heavier activity. A full-size attempt was stopped after benchmarking the runtime increase, before control accuracies were observed. The final declared comparison uses 200 training, 100 validation and 200 test images, balanced by digit, for each retrained model. The intact comparator uses those same images and a newly fitted head. Three fixed seeds are reported; there is no best-seed selection.

| Condition | Correct / evaluated | Accuracy | Mean selected feature count |
|---|---:|---:|---:|
| Intact circuit, retrained subset | 164 / 200 | 82.0% | 1.454 |
| Random seed 41 | 130 / 200 | 65.0% | 1.743 |
| Random seed 42 | 127 / 200 | 63.5% | 1.703 |
| Random seed 43 | 109 / 200 | 54.5% | 1.589 |

The direct edge lesion with the original decoder scored 10.0% on 200 images. This intervention also causes distribution shift. Randomized controls differ in activity scale and weighted target input; these checks do not establish a general advantage from biological anatomy. In particular, conventional pixel models are stronger on the primary digit task. See [controls](../reports/controls/controls.json) and per-seed protocols.

## Local execution

The primary digit extraction averaged 0.292 seconds per training glyph, and the numeric extraction averaged 0.327 seconds. These measurements include full canonical reset, sensory sampling and four neural bins on an Apple M2 Pro while some experiments ran concurrently. The recorded PDF recognition used 41.8 seconds for 120 glyphs. Cold-start and feature sizes are in the recognition reports. No cloud runtime was needed.

The 40-second MP4 edits presentation timing. It is rendered from the same event stream and Canvas function as the local viewer. Its final scores are measured, including failed release gates. No live-body or real-fly reading claim is made.
