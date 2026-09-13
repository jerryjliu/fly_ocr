# Calibrated fly OCR results

The internal fly circuit and its weights remain fixed. A label-free geometric input adapter exposes the whole glyph, and a compact nonlinear classifier reads only downstream spike counts.

| Measurement | Original letter model | Calibrated model |
|---|---:|---:|
| Matched benchmark, all 68 classes | 74.9% | 87.6% |
| Matched benchmark, letters only | 71.9% | 85.0% |
| Fresh font families, all 68 classes | 77.8% | 84.9% |
| Fresh font families, letters only | 80.5% | 88.5% |
| Same PDF crops, character error rate (lower is better) | 19.3% | 5.7% |
| Same PDF crops, exact lines | 0/8 | 1/8 |

The matched benchmark has 1,632 glyphs; its font families and earlier errors were already seen in prior work. The fresh challenge has 544 glyphs from Verdana and Georgia, generated after the checkpoint was frozen. Neither test selected the new head. Two font families still provide limited evidence of generalization.

## What the classifier comparison found

| Head on calibrated circuit responses | Validation | Matched benchmark | Parameters |
|---|---:|---:|---:|
| raw64 | 88.9% | 87.2% | 266,628 |
| sqrt64 **(selected)** | 89.1% | 87.6% | 266,628 |
| sqrt64x32 | 88.3% | 86.3% | 266,532 |
| sqrt128x64 | 88.3% | 86.1% | 537,092 |

Selected architecture: [64] hidden units, `sqrt` count transform. All inputs are 4,096 binned spike counts. No image pixels, text labels, dictionary, or language-model correction enter the readout.

Mismatching neural responses to glyph labels drops matched-benchmark accuracy to 2.3%.

## Compute

The paired 16-glyph circuit check averaged 0.295 s originally and 0.289 s with calibrated inputs. Both use one 100 ms simulated presentation per glyph. This is a small local timing check, not a universal performance guarantee.

The selected NumPy decoder took 1.117 ms per glyph in its separate microbenchmark; the original took 0.706 ms. Internal graph and original retinal-map files were checksum-verified unchanged.

## Updated document demonstrations

All three regions match the original raster crops exactly. Segmentation and geometric spaces are unchanged. PDF pixels informed the earlier segmenter, so these are paired demonstrations rather than an independent OCR benchmark.

### calibrated-heading

[Video](../demo/examples/calibrated-heading/video.mp4) · 0/1 exact lines · 14.3% character error rate.

```text
BALANCE 5HEET5
```

### calibrated-labels

[Video](../demo/examples/calibrated-labels/video.mp4) · 1/2 exact lines · 2.1% character error rate.

```text
Cash and cash equivaIents
Short-term investments
```

### calibrated-labels-more

[Video](../demo/examples/calibrated-labels-more/video.mp4) · 0/5 exact lines · 6.1% character error rate.

```text
Operating Iease right-of-use assets
Equity and other investmenls
GoodwiII
intangible assels, net
Other lonq-term assets
```

## Run it

```sh
uv run flyocr recognize your-text-crop.png --letters --output output/letters
uv run flyocr verify-replay --run output/letters --artifact artifacts/letters-v2
```

The original letter checkpoint is retained at `artifacts/letters` and can be selected with `--artifact artifacts/letters`. Numeric-table mode retains its separate numeric checkpoint.

[Methods and reproduction](input-calibration.md) · [Complete metrics](../reports/letters-v2/summary.json)

Unsupported alphabets, touching characters, uncertain line height, rotation, and complex layouts remain limitations. Class scores are not calibrated probabilities of correctness.
