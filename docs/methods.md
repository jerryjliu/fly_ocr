# Methods

## What is fixed and what learns

The biological dataset supplies a connectivity scaffold. Every retained node and edge is simulated; the neural feature set is a readout selection, not a subgraph substitute. Internal weights remain contact count × inferred presynaptic sign × 0.275. Acetylcholine is excitatory; GABA, glutamate and histamine are inhibitory in this simplified model. Ambiguous assignments default to positive and their count is recorded. This is an engineering approximation, especially for transmitter effects and heterogeneous cell physiology.

Nodes with assigned superclasses are retained, including uncertain `tbc` groups. Explicit glia are excluded. Weak edges and self-edges are retained. The importer checks exact counts against the pinned source registry and preserves anatomical IDs as 64-bit integers. Displayed anatomical IDs are strings, so JavaScript cannot round them.

The retinal projection infers ommatidial coordinates from R1–R6 connectivity onto L1–L3 cells. The two eyes share an approximate viewport. There are 3,335 mapped retinal inputs. Tonic input is applied to L1/L2/L3/L5 separately. The primary readout excludes both directly driven retinal neurons and tonic-input neurons; candidates are optic-lobe intrinsic and visual projection cells.

## Numerical model

The fixed DOOMFLY CPU kernel uses a 0.1 ms event grid, analytical subthreshold integration, resting/reset voltage −52 mV, threshold −45 mV, membrane/synaptic time constants 20/5 ms, 1.8 ms delivery delay and 2.2 ms refractory interval. The active-set optimization preserves event semantics. Synaptic decay and membrane integration pause during refractory intervals; input arriving while refractory follows the upstream queue/update behavior. This is not a full biophysical model.

The small-network suite compares an analytic single-cell solution, excitatory/inhibitory dynamics against an independent Brian2 implementation, interval splitting, and complete checkpoint continuation. Floating-point tolerances are explicit in the tests. The independent reference test compares final voltages and spike totals on a non-boundary fixture; it is not a proof of numerical equivalence for every graph or threshold-boundary case.

The canonical state includes voltages, synaptic drive, refractory counters, pending event queues, queue lengths, time, previous input, current input, active cells/flags, last update timestamps, counts and luminance filters. It is saved after a 50 ms white warm-up and restored for each glyph. No circuit random state is used. The full-graph diagnostic also checks that a glyph's features are unchanged after presenting a different glyph and that the on-disk internal weight hash stays fixed.

## Image and sensory input

Training images are drawn with the checked-in open fonts. Source raster sizes, weight axes, small rotations, blur and pixel translations vary independently of class. Each image is cropped to its ink bounds, resized preserving aspect ratio to roughly 72% of a 48×48 white canvas, and centered. The training occupancy varies from 65–80%; PDF inference uses 72%. Punctuation is normalized by its own ink box, which discards original baseline and relative type size. This choice limits punctuation discrimination and is disclosed.

Bilinear sampling produces brightness values at the retinal coordinates. In the selected development configuration, these are inverted so dark ink drives input. A 10 ms low-pass filter produces luminance L, then the drive is `20 × L / (0.2 + L)`. Tonic lamina drive is 20. Every glyph is presented for 100 ms in four 25 ms bins. The viewer's retinal panel shows sampled brightness before inversion/current conversion, and its neural panel shows recorded downstream spike counts.

A 12-configuration pilot varied four retinal gains and three tonic levels. A second, separately logged eight-configuration batch tested a different half-saturation value with both contrast polarities. Both used only 120 training and 80 validation images. The selected configuration reached 78.75% validation accuracy in this small pilot. Both complete result sets are included. No test fonts or target PDF labels selected the configuration.

## Learned decoder and evaluation

Each final task selects at most 512 downstream cells by response variance on 12 training images per class, producing 2,048 count features. StandardScaler is fit only to training features. Multinomial logistic regression tries C ∈ {0.001, 0.01, 0.1, 1, 10}; validation accuracy selects C, ties favor the smaller value. There is no train+validation refit. Parameters, class order, selected cells and identity records are saved before loading the test images.

The digit corpus contains 1,000 training, 200 validation and 500 test images. The numeric corpus contains 2,400/480/800 images covering 16 classes. Training families: Roboto, Open Sans, Lato, Noto Sans, Source Sans 3, Montserrat. Validation: PT Sans and Lora. Test: Fira Sans and Noto Serif. Font files and the Google Fonts commit are pinned in `assets/fonts/manifest.json`.

The held-out reports include accuracy, macro F1, confusion matrices, per-font scores, and image-level Wilson intervals. Only two test font families are represented. These intervals ignore within-font dependence and do not imply broad document generalization.

Raw-pixel and retinal-sample linear baselines use the identical splits and C-selection rule. A separate two-layer image CNN uses 20 epochs, selects its checkpoint by validation accuracy, and never supplies features or answers to the fly model. Shuffled training labels, mismatched test features and blank input check dependency. The edge-removal intervention uses the original head, so distribution shift limits interpretation.

Three randomized-target controls each retrain their own variance selection, normalization and linear head. After measuring a large runtime increase, the final bounded comparison was declared as 200 training, 100 validation and 200 test glyphs per condition, balanced by digit, before observing control accuracies. The intact comparator is separately fitted on those same images. Permuting the full edge-target vector preserves exact in/out edge degrees and each source's signed weight list. It destroys spatial organization, can create parallel edges, and does not preserve weighted input totals per target. Activity-scale comparisons are reported; this is a narrow structural comparison, not a biological causality claim.

## PDF experiment and boundaries

The target is a preselected, manually positioned numeric column from Microsoft 2025's annual report: all 19 numeric cells, printed page 38 / PDF page 40. Poppler rasterizes at 300 DPI. The right crop boundary was adjusted by inspecting pixels before predictions to preserve a complete closing parenthesis and exclude the next column.

Segmentation sees only pixels. Horizontal rules spanning at least 80% of the crop width are removed. Row and column ink projections identify glyph boxes; nearby vertical bands merge to preserve punctuation. A synthetic table test covers dollar signs, commas, decimals, minus and parentheses. All 120 automatic glyph boxes were visually checked against this source. Their counts match all 19 truth strings; this does not establish segmentation reliability on arbitrary tables. There is no independent manually boxed PDF benchmark or separate document-development corpus in this release.

The recognizer records all raw characters, counts, retinal samples and class scores before the separate evaluator reads its truth file. The viewer bundle omits truth strings: it receives predictions plus per-row correctness flags and aggregate scores for the final frame. No accounting identities, number parsing, language model, embedded text, or known financial values repair predictions. Currency signs count toward exact match. Missing/extra characters count in edit distance. All originally selected rows stay in the denominator.

The closed vocabulary has no calibrated open-set rejection. Rotated text, letters, touching glyphs and arbitrary paragraph layouts are unsupported and can produce erroneous numeric output. Blank crops produce no rows. A successful numeric-column demo would not establish paragraph OCR.

## Provenance and execution

Graph arrays, source tables, corpora, models, glyph images and replay records have checksum/identity metadata. Cached feature shards bind source image archives, model settings, selected cell indices and intervals; completed shards are atomically written and validated before reuse. Reproduction also requires the checked-in software version and lockfiles. Cache format 2 additionally fingerprints the Python sensory/feature/native-wrapper implementation, so code changes invalidate reuse. The initial measured extraction used format 1 under fixed code; its original protocol records are retained. Model IDs encode the graph, stimulus configuration and native kernel, and the release manifest separately fingerprints software sources.

The Mac builds its own native library. Full-graph work ran locally; no Modal job, GPU, cloud upload or payment was required. Linux is covered by the portable build path and CI configuration but the full Mac/Linux graph-equivalence experiment has not been run. The standalone video is a deterministic rendering of recorded results, not a live desktop capture. The browser and video share the same TypeScript Canvas renderer.
