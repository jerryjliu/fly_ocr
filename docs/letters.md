# English-letter readout

This page records the original letter checkpoint. The calibrated model is now selected by `--letters`; see [updated results](letters-v2.md) and [input/readout methods](input-calibration.md).

The expanded readout recognizes uppercase and lowercase English letters, digits, and numeric punctuation. Only the readout is trained. The 166,700-neuron circuit, retinal projection, and internal weights remain fixed.

## Held-out glyphs

Overall: **1222/1632 (74.9%)** across 68 classes. Validation accuracy: 79.2%.

| Group | Correct / tested | Accuracy |
|---|---:|---:|
| Uppercase | 447/624 | 71.6% |
| Lowercase | 450/624 | 72.1% |
| Digits | 190/240 | 79.2% |
| Punctuation | 135/144 | 93.8% |

The training set contains 96 images per class (6,528 total), validation 16/class (1,088), and test 24/class (1,632). Six font families supply training data, two separate families validation, and two further families testing. Training does not use PDF pixels or transcriptions. These are synthetic printed glyphs, not handwriting or a representative general OCR benchmark.

The test split is opened only after the checkpoint is frozen. Font families are shared within each split, so image-level confidence intervals do not capture all dependence.

## Actual report text

The following complete regions were selected while training was underway, before letter predictions. PDF pixels informed segmentation development; document scores are demonstrations, not an independent benchmark. Spaces are inferred from pixel gaps, not output classes.

| Example | Exact lines | Character edit rate |
|---|---:|---:|
| [An uppercase report heading](../demo/examples/trained-heading/video.mp4) | 0/1 | 28.57% |
| [Two actual report labels](../demo/examples/trained-labels/video.mp4) | 0/2 | 14.89% |
| [More balance-sheet labels](../demo/examples/trained-labels-more/video.mp4) | 0/5 | 20.00% |

Actual output (no correction):

```text
EALANC5 SH55TS
```

```text
Carb and carb equiyalentr
Short-term invertments
```

```text
Ooerating leare right-pf-ure arsetr
Fquitv and other invertments
GpodwiII
IntanqihIe arrets. net
Otber Iong-term arretr
```

## Input and readout

The original tight normalization erased the size distinction between letters such as C/c and O/o. The new line-box-v1 normalization preserves each glyph’s height and baseline position relative to its line. Only its 48×48 pixel image reaches the circuit. Line metrics do not enter the classifier as extra features.

Character boxes come from pixel projections. A conservative thin-bridge heuristic separates some tightly kerned pairs with different top heights. Dots remain attached to their glyphs. Baselines and word spaces come from geometry, with no OCR engine, dictionary, embedded text extraction, or language model.

The readout observes 1024 training-selected downstream cells in four time bins (4096 features). Selection uses response variance on eight training images per class. A linear readout was compared with three small ReLU readouts using validation fonts only. The selected readout has one hidden layer with 64 units. Its inputs are standardized and clipped downstream spike counts; no raw pixels or line metrics bypass the circuit.

## Use it

```sh
uv run flyocr recognize your-text-crop.png --artifact artifacts/letters --output output/letters
uv run flyocr verify-replay --run output/letters --artifact artifacts/letters
```

The existing default numeric checkpoint and numeric-table mode remain available. The new letter checkpoint currently accepts upright text regions; combining it with --table is rejected explicitly. It does not yet interpret tables with text labels.

To retrain and record the examples:

```sh
uv run python scripts/train-letters.py --workers 2
uv run --extra benchmarks python scripts/refine-letter-readout.py
uv run python scripts/finalize-letter-readout.py
uv run python scripts/record-letters.py
uv run --extra video python scripts/export-video.py --examples trained-heading,trained-labels,trained-labels-more
uv run --extra video python scripts/verify-example-videos.py --letters
```

## Limits

The alphabet is `0123456789,.-()$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`. Spaces are a layout result. Other punctuation, accented characters, handwriting, touching cursive, arbitrary rotation, and complex layouts remain outside the supported scope. Letter shapes such as I/l and O/0 can be ambiguous without word context; no word correction is applied. Class scores are not calibrated confidence estimates or an unsupported-character detector. Lines without a useful height reference can confuse case. A small fraction of augmented training rasters touch their canvas boundary.

Both workers maintain independent circuit state, and every glyph starts from a canonical reset. Checked shards make feature extraction resumable. The simulator’s internal weight checksum is unchanged after training. Replay verification recalculates predictions from saved activity and rederives letter boxes and spaces from source pixels.

Artifact: `1ea442a7702901ae5f83860fcbd179a11b10b8c66432afe02cf2b6d1e70280bc`. Total local training/evaluation wall time: 28.3 minutes.

The initial linear head reached 77.3% validation accuracy. The selected 64-unit ReLU head reached 79.2%. Two subsequent exploratory PCA/RBF comparisons (64 and 256 components) were attempted after observing PDF errors; neither improved validation, and the deployed head was left unchanged. The earlier linear checkpoint is preserved in artifacts/letters-linear.

The scripts resume unfinished work and preserve a completed refined checkpoint. For an independent retraining experiment, use a separate working copy with the generated artifacts/letters, artifacts/letters-linear, and artifacts/letter-refinement directories initially absent. Inference uses NumPy/SciPy and does not require PyTorch; the optional PyTorch dependency is used for readout training and an independent serialization check.
