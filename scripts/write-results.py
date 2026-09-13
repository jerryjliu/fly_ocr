"""Generate human-readable results from completed machine-readable reports."""
from pathlib import Path
from flyocr.common import read_json

d=read_json('artifacts/digits/recognition.json')
n=read_json('artifacts/numeric/recognition.json')
p=read_json('reports/pdf-crop.json')
c=read_json('reports/controls/controls.json')
cnn=read_json('reports/cnn.json')
rows=['| Experiment | Correct / evaluated | Accuracy |','|---|---:|---:|']
for label,r in [('Fly circuit: digits',d['neural']),('Fly circuit: numeric alphabet',n['neural']),('Raw pixels: linear',d['baselines']['raw_pixels']),('Retinal samples: linear',d['baselines']['retinal_samples']),('Image CNN reference',cnn['test']),('Shuffled training labels',d['baselines']['shuffled_training_labels']),('Mismatched test images',d['baselines']['mismatched_test_images']),('Blank input, original head',d['baselines']['blank_input_original_head'])]:
 rows.append(f"| {label} | {r['correct']} / {r['n']} | {r['accuracy']:.1%} |")
cr=['| Condition | Correct / evaluated | Accuracy | Mean selected feature count |','|---|---:|---:|---:|']
r=c['intact_matched_subset'];cr.append(f"| Intact circuit, retrained subset | {r['correct']} / {r['n']} | {r['accuracy']:.1%} | {r['mean_feature_count']:.3f} |")
for ctrl in c['randomized']:
 if 'error' in ctrl:cr.append(f"| Random seed {ctrl['seed']} | no varying features | — | — |")
 else:
  r=ctrl['neural'];cr.append(f"| Random seed {ctrl['seed']} | {r['correct']} / {r['n']} | {r['accuracy']:.1%} | {ctrl['mean_feature_count']:.3f} |")
pdf=['| Cell | Source truth | Raw prediction | Exact |','|---|---|---|---|']
for r in p['rows']:pdf.append(f"| {r['row_id']} | `{r['truth']}` | `{r['prediction']}` | {'yes' if r['exact'] else 'no'} |")
text=f'''# Measured results

All primary results use a frozen, complete retained MaleCNS graph with a trained linear readout. Configuration selection used training/validation data only. These are bounded synthetic-font and single-column experiments.

## Held-out characters

{chr(10).join(rows)}

Digit accuracy is {d['neural']['accuracy']:.1%}, with an image-level 95% Wilson interval of {d['neural']['wilson_95'][0]:.1%}–{d['neural']['wilson_95'][1]:.1%}. The two test families are unseen during training; image dependence within families limits that interval. The proposed 90% digit gate was {'met' if d['neural']['accuracy']>=.9 else 'missed'}. The full confusion matrices, per-font breakdowns, validation trials and denominators are in [digit recognition](../artifacts/digits/recognition.json) and [numeric recognition](../artifacts/numeric/recognition.json).

## Complete selected PDF column

**{p['correct_cells']}/{p['n_cells']} exact cells ({p['cell_exact_match']:.1%}); character error rate {p['character_error_rate']:.1%}.** All {p['truth_characters']} characters and 19 original cells remain in the evaluation. Coverage is {p['coverage']:.0%}; automatic glyph counts agree for {p['glyph_count_match_cells']}/19 cells. The proposed 80% exact-cell gate was {'met' if p['threshold_met'] else 'missed'}. No correction pass was used.

{chr(10).join(pdf)}

This is one manually selected numeric column. Automatic glyph boxes were visually checked; no independent manually boxed recognition benchmark was run. Missing/extra punctuation counts as an error. The [evaluation report](../reports/pdf-crop.json) links the original run and truth checksums.

## Wiring comparison

Randomized graphs retain exact edge in/out degrees and each source's signed weight list. They produce much heavier activity. A full-size attempt was stopped after benchmarking the runtime increase, before control accuracies were observed. The final declared comparison uses 200 training, 100 validation and 200 test images, balanced by digit, for each retrained model. The intact comparator uses those same images and a newly fitted head. Three fixed seeds are reported; there is no best-seed selection.

{chr(10).join(cr)}

The direct edge lesion with the original decoder scored {c['edge_lesion_original_head']['accuracy']:.1%} on {c['edge_lesion_original_head']['n']} images. This intervention also causes distribution shift. Randomized controls differ in activity scale and weighted target input; these checks do not establish a general advantage from biological anatomy. In particular, conventional pixel models are stronger on the primary digit task. See [controls](../reports/controls/controls.json) and per-seed protocols.

## Local execution

The primary digit extraction averaged {d['benchmark']['train']['mean_seconds']:.3f} seconds per training glyph, and the numeric extraction averaged {n['benchmark']['train']['mean_seconds']:.3f} seconds. These measurements include full canonical reset, sensory sampling and four neural bins on an Apple M2 Pro while some experiments ran concurrently. The recorded PDF recognition used {read_json('demo/run/run.json')['wall_seconds']:.1f} seconds for 120 glyphs. Cold-start and feature sizes are in the recognition reports. No cloud runtime was needed.

The 40-second MP4 edits presentation timing. It is rendered from the same event stream and Canvas function as the local viewer. Its final scores are measured, including failed release gates. No live-body or real-fly reading claim is made.
'''
Path('docs/results.md').write_text(text)
card=f'''# Model card

**Intended use:** a reproducible research/demo experiment in character recognition through a fixed connectome model. It is not a production PDF parser, a model of a real fly reading, or financial-data extraction software.

**Primary results:** 445/500 held-out digits (89%); {n['neural']['correct']}/{n['neural']['n']} held-out numeric glyphs ({n['neural']['accuracy']:.1%}); {p['correct_cells']}/19 exact PDF cells ({p['cell_exact_match']:.1%}). Raw-pixel linear classification achieves 99% on the same digit split. The proposed digit gate was missed; PDF gate {'met' if p['threshold_met'] else 'missed'}.

**Fixed:** all 166,700 retained nodes, 25,582,938 edges and internal weights; simplified LIF dynamics; inferred retinal geometry; stimulus transform and canonical state. **Learned:** training-derived feature selection/standardization and a regularized linear head. There is no internal synaptic plasticity, LLM, OCR service, text-layer input, numerical correction or learned movement policy.

**Data:** synthetic printed fonts with six training, two validation and two test families; pinned open font files. Full split sizes and classes are in the corpus manifests. The PDF source is Microsoft's 2025 annual report, printed page 38/PDF page 40. The target column was selected before predictions; 19 cells and all punctuation are retained. Evaluation truth is outside the recognition package.

**Limits:** two test font families; only one PDF column; manually selected region; simplified and unvalidated fly vision; closed alphabet; no calibrated rejection of unsupported symbols; segmentation assumes upright separated numeric glyphs. Punctuation normalization discards relative font size and baseline. Softmax scores are not calibrated confidence. Errors remain in raw outputs. Do not rely on this experiment for financial reporting.

**Controls:** shuffled labels, mismatched images, blank input, edge lesion, pixel/retinal linear baselines, image CNN, and three separately retrained degree-preserving randomized-target graphs on a smaller matched dataset. Activity-scale mismatch and weighted-input changes limit causal interpretation. [Full outcomes](docs/results.md).

**Artifacts:** [digit head](artifacts/digits/model-card.json), [numeric head](artifacts/numeric/model-card.json), [replay](demo/run/run.json). Graph, readout, source and event hashes are recorded. The [methods](docs/methods.md) describe solver tolerances, selection rules and cache identity. The public export manifest (`PUBLIC_MANIFEST.json`) fingerprints distributable artifacts and software.

**Runtime:** tested locally on Apple M2 Pro, 32 GiB RAM. Native C++ CPU kernel. No Modal deployment or GPU was required. Linux CI configuration is included; full-graph Mac/Linux equivalence has not been measured. The fly in the viewer is an animated cursor; the activity is recorded simulation data.

**Licenses:** project code MIT; adapted DOOMFLY MIT; MaleCNS CC BY 4.0; bundled fonts SIL OFL; Microsoft report pixels remain Microsoft's. See [notices](THIRD_PARTY_NOTICES.md).
'''
Path('MODEL_CARD.md').write_text(card)
