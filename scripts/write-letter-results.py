"""Write the letter experiment summary directly from saved evaluations."""
from pathlib import Path
from flyocr.common import read_json,save_json

r=read_json('artifacts/letters/recognition.json')
card=read_json('artifacts/letters/model-card.json')
spec=read_json('examples/microsoft-2025/letter-examples.json')
lines=['# English-letter readout','',
'The expanded readout recognizes uppercase and lowercase English letters, digits, and numeric punctuation. Only the readout is trained. The 166,700-neuron circuit, retinal projection, and internal weights remain fixed.','',
'## Held-out glyphs','',
f"Overall: **{r['neural']['correct']}/{r['neural']['n']} ({r['neural']['accuracy']:.1%})** across 68 classes. Validation accuracy: {r['validation']['validation_accuracy']:.1%}.",'',
'| Group | Correct / tested | Accuracy |','|---|---:|---:|']
for name,value in r['by_group'].items():lines.append(f"| {name.title()} | {value['correct']}/{value['n']} | {value['accuracy']:.1%} |")
lines+=['','The training set contains 96 images per class (6,528 total), validation 16/class (1,088), and test 24/class (1,632). Six font families supply training data, two separate families validation, and two further families testing. Training does not use PDF pixels or transcriptions. These are synthetic printed glyphs, not handwriting or a representative general OCR benchmark.','',
'The test split is opened only after the checkpoint is frozen. Font families are shared within each split, so image-level confidence intervals do not capture all dependence.','',
'## Actual report text','',
'The following complete regions were selected while training was underway, before letter predictions. PDF pixels informed segmentation development; document scores are demonstrations, not an independent benchmark. Spaces are inferred from pixel gaps, not output classes.','',
'| Example | Exact lines | Character edit rate |','|---|---:|---:|']
examples=[]
for item in spec['examples']:
 name=item['id'];s=read_json('reports/letters/'+name+'.json');run=read_json('demo/examples/'+name+'/run.json')
 lines.append(f"| [{item['caption']}](../demo/examples/{name}/video.mp4) | {s['correct_cells']}/{s['n_cells']} | {s['character_error_rate']:.2%} |")
 examples.append({'id':name,'correct_lines':s['correct_cells'],'lines':s['n_cells'],'edit_errors':s['edit_errors'],'truth_characters':s['truth_characters'],'measured_seconds':run['wall_seconds'],'raw':[v['raw'] for v in run['rows']]})
lines+=['','Actual output (no correction):','']
for e in examples:
 lines+=['```text',*e['raw'],'```','']
lines+=['## Input and readout','',
'The original tight normalization erased the size distinction between letters such as C/c and O/o. The new line-box-v1 normalization preserves each glyph’s height and baseline position relative to its line. Only its 48×48 pixel image reaches the circuit. Line metrics do not enter the classifier as extra features.','',
'Character boxes come from pixel projections. A conservative thin-bridge heuristic separates some tightly kerned pairs with different top heights. Dots remain attached to their glyphs. Baselines and word spaces come from geometry, with no OCR engine, dictionary, embedded text extraction, or language model.','',
f"The readout observes {card['feature_cells']} training-selected downstream cells in four time bins ({card['feature_count']} features). Selection uses response variance on eight training images per class. A linear readout was compared with three small ReLU readouts using validation fonts only. The selected readout has one hidden layer with {card.get('hidden_units',0)} units. Its inputs are standardized and clipped downstream spike counts; no raw pixels or line metrics bypass the circuit.",'',
'## Use it','',
'```sh','uv run flyocr recognize your-text-crop.png --artifact artifacts/letters --output output/letters','uv run flyocr verify-replay --run output/letters --artifact artifacts/letters','```','',
'The existing default numeric checkpoint and numeric-table mode remain available. The new letter checkpoint currently accepts upright text regions; combining it with --table is rejected explicitly. It does not yet interpret tables with text labels.','',
'To retrain and record the examples:','',
'```sh','uv run python scripts/train-letters.py --workers 2','uv run --extra benchmarks python scripts/refine-letter-readout.py','uv run python scripts/finalize-letter-readout.py','uv run python scripts/record-letters.py','uv run --extra video python scripts/export-video.py --examples trained-heading,trained-labels,trained-labels-more','uv run --extra video python scripts/verify-example-videos.py --letters','```','',
'## Limits','',
'The alphabet is `0123456789,.-()$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`. Spaces are a layout result. Other punctuation, accented characters, handwriting, touching cursive, arbitrary rotation, and complex layouts remain outside the supported scope. Letter shapes such as I/l and O/0 can be ambiguous without word context; no word correction is applied. Class scores are not calibrated confidence estimates or an unsupported-character detector. Lines without a useful height reference can confuse case. A small fraction of augmented training rasters touch their canvas boundary.','',
'Both workers maintain independent circuit state, and every glyph starts from a canonical reset. Checked shards make feature extraction resumable. The simulator’s internal weight checksum is unchanged after training. Replay verification recalculates predictions from saved activity and rederives letter boxes and spaces from source pixels.','',
f"Artifact: `{card['artifact_id']}`. Total local training/evaluation wall time: {r['wall_seconds']/60:.1f} minutes.",'']
lines += ['The initial linear head reached 77.3% validation accuracy. The selected 64-unit ReLU head reached 79.2%. Two subsequent exploratory PCA/RBF comparisons (64 and 256 components) were attempted after observing PDF errors; neither improved validation, and the deployed head was left unchanged. The earlier linear checkpoint is preserved in artifacts/letters-linear.','',
'The scripts resume unfinished work and preserve a completed refined checkpoint. For an independent retraining experiment, use a separate working copy with the generated artifacts/letters, artifacts/letters-linear, and artifacts/letter-refinement directories initially absent. Inference uses NumPy/SciPy and does not require PyTorch; the optional PyTorch dependency is used for readout training and an independent serialization check.','']
Path('docs/letters.md').write_text('\n'.join(lines))
save_json('reports/letters/summary.json',{'artifact_id':card['artifact_id'],'heldout':{k:v for k,v in r['neural'].items() if k!='confusion_matrix'},'groups':{k:{f:v[f] for f in ['n','correct','accuracy']} for k,v in r['by_group'].items()},'examples':examples,'internal_weights_unchanged':r['internal_weight_hash_unchanged']})
print('Wrote docs/letters.md and reports/letters/summary.json')
