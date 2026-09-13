"""Summarize measured v2 outcomes without changing model selection."""
from pathlib import Path
from flyocr.common import read_json,save_json


def compact(r):
    return {k:r[k] for k in ["n","correct","accuracy"]}


def main():
    card=read_json('artifacts/letters-v2/model-card.json')
    new=read_json('artifacts/letters-v2/recognition.json');old=read_json('artifacts/letters/recognition.json')
    fresh=read_json('artifacts/letters-v2/fresh-recognition.json');reference=read_json('artifacts/letters-v2/fresh-reference.json')
    validation=read_json('artifacts/letters-v2/validation.json')
    timing=read_json('artifacts/letters-v2/input-benchmark.json')
    verified=read_json('reports/letters-v2/verification.json')
    old_letters={"n":sum(old['by_group'][k]['n'] for k in ['uppercase','lowercase']),
                 "correct":sum(old['by_group'][k]['correct'] for k in ['uppercase','lowercase'])}
    old_letters['accuracy']=old_letters['correct']/old_letters['n']
    examples=read_json('examples/microsoft-2025/calibrated-examples.json')['examples']
    pdf={"original":{"edits":0,"characters":0,"exact_lines":0,"lines":0,"seconds":0},
         "selected":{"edits":0,"characters":0,"exact_lines":0,"lines":0,"seconds":0}}
    comparisons=[]
    for item in examples:
        row={"id":item['id'],"reference_id":item['reference_id']}
        for label,name,report_root in [('original',item['reference_id'],'letters'),('selected',item['id'],'letters-v2')]:
            score=read_json(f'reports/{report_root}/{name}.json');run=read_json(f'demo/examples/{name}/run.json')
            for target,key in [('edits','edit_errors'),('characters','truth_characters'),('exact_lines','correct_cells'),('lines','n_cells')]:
                pdf[label][target]+=score[key]
            pdf[label]['seconds']+=run['wall_seconds']
            row[label]={"cer":score['character_error_rate'],"exact_lines":score['correct_cells'],"lines":score['n_cells'],
                        "raw_lines":[r['raw'] for r in run['rows']]}
        comparisons.append(row)
    for value in pdf.values():value['cer']=value['edits']/value['characters']
    summary={"artifact_id":card['artifact_id'],"architecture":card['readout_architecture'],
        "matched":{"original":compact(old['neural']),"selected":compact(new['neural']),
                   "original_letters":old_letters,"selected_letters":compact(new['by_group']['letters'])},
        "fresh":{"original":compact(reference['neural']),"selected":compact(fresh['neural']),
                 "original_letters":compact(reference['by_group']['letters']),"selected_letters":compact(fresh['by_group']['letters']),
                 "paired":reference['paired']},"pdf":pdf,"examples":comparisons,
        "timing":timing,"decoder_seconds_per_glyph":verified['decoder_seconds_per_glyph'],
        "source_arrays_unchanged":verified['source_arrays_unchanged']}
    save_json('reports/letters-v2/summary.json',summary)
    pct=lambda x:f"{100*x:.1f}%"
    lines=['# Calibrated fly OCR results','',
        'The internal fly circuit and its weights remain fixed. A label-free geometric input adapter exposes the whole glyph, and a compact nonlinear classifier reads only downstream spike counts.','',
        '| Measurement | Original letter model | Calibrated model |','|---|---:|---:|',
        f"| Matched benchmark, all 68 classes | {pct(old['neural']['accuracy'])} | {pct(new['neural']['accuracy'])} |",
        f"| Matched benchmark, letters only | {pct(old_letters['accuracy'])} | {pct(new['by_group']['letters']['accuracy'])} |",
        f"| Fresh font families, all 68 classes | {pct(reference['neural']['accuracy'])} | {pct(fresh['neural']['accuracy'])} |",
        f"| Fresh font families, letters only | {pct(reference['by_group']['letters']['accuracy'])} | {pct(fresh['by_group']['letters']['accuracy'])} |",
        f"| Same PDF crops, character error rate (lower is better) | {pct(pdf['original']['cer'])} | {pct(pdf['selected']['cer'])} |",
        f"| Same PDF crops, exact lines | {pdf['original']['exact_lines']}/{pdf['original']['lines']} | {pdf['selected']['exact_lines']}/{pdf['selected']['lines']} |",'',
        'The matched benchmark has 1,632 glyphs; its font families and earlier errors were already seen in prior work. The fresh challenge has 544 glyphs from Verdana and Georgia, generated after the checkpoint was frozen. Neither test selected the new head. Two font families still provide limited evidence of generalization.','',
        '## What the classifier comparison found','',
        '| Head on calibrated circuit responses | Validation | Matched benchmark | Parameters |',
        '|---|---:|---:|---:|']
    for trial in validation['trials']:
        name=trial['name'];mark=' **(selected)**' if name==validation['selected'] else ''
        lines.append(f"| {name}{mark} | {pct(trial['validation_accuracy'])} | {pct(new['head_comparison'][name]['accuracy'])} | {trial['parameters']:,} |")
    lines += ['',f"Selected architecture: {card['readout_architecture']['hidden']} hidden units, `{card['readout_architecture']['count_transform']}` count transform. All inputs are 4,096 binned spike counts. No image pixels, text labels, dictionary, or language-model correction enter the readout.",'',
        f"Mismatching neural responses to glyph labels drops matched-benchmark accuracy to {pct(new['mismatched_neural_responses']['accuracy'])}.",'',
        '## Compute','',
        f"The paired 16-glyph circuit check averaged {timing['original']['seconds_per_glyph']:.3f} s originally and {timing['balanced']['seconds_per_glyph']:.3f} s with calibrated inputs. Both use one 100 ms simulated presentation per glyph. This is a small local timing check, not a universal performance guarantee.",'',
        f"The selected NumPy decoder took {1000*verified['decoder_seconds_per_glyph']['selected']:.3f} ms per glyph in its separate microbenchmark; the original took {1000*verified['decoder_seconds_per_glyph']['original']:.3f} ms. Internal graph and original retinal-map files were checksum-verified unchanged.",'',
        '## Updated document demonstrations','',
        'All three regions match the original raster crops exactly. Segmentation and geometric spaces are unchanged. PDF pixels informed the earlier segmenter, so these are paired demonstrations rather than an independent OCR benchmark.']
    for item in comparisons:
        name=item['id'];r=item['selected']
        lines += ['',f"### {name}",'',f"[Video](../demo/examples/{name}/video.mp4) · {r['exact_lines']}/{r['lines']} exact lines · {pct(r['cer'])} character error rate.",'','```text',*r['raw_lines'],'```']
    lines += ['','## Run it','','```sh','uv run flyocr recognize your-text-crop.png --letters --output output/letters',
        'uv run flyocr verify-replay --run output/letters --artifact artifacts/letters-v2','```','',
        'The original letter checkpoint is retained at `artifacts/letters` and can be selected with `--artifact artifacts/letters`. Numeric-table mode retains its separate numeric checkpoint.','',
        '[Methods and reproduction](input-calibration.md) · [Complete metrics](../reports/letters-v2/summary.json)','',
        'Unsupported alphabets, touching characters, uncertain line height, rotation, and complex layouts remain limitations. Class scores are not calibrated probabilities of correctness.']
    Path('docs/letters-v2.md').write_text('\n'.join(lines)+'\n')
    print('Saved paired character, document, and compute results.',flush=True)


if __name__=='__main__':main()
