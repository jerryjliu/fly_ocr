"""Repeat all three prior document crops with the frozen calibrated checkpoint."""
from pathlib import Path
import shutil
from flyocr.common import read_json,save_json,digest_file
from flyocr.demo.record import record
from flyocr.demo.schema import verify
from flyocr.eval.pdf import evaluate


def main():
    specs=read_json('examples/microsoft-2025/letter-examples.json')
    recognition=read_json('artifacts/letters-v2/recognition.json')
    titles={"trained-heading":"Full glyph coverage through the fly circuit",
            "trained-labels":"A compact nonlinear character readout",
            "trained-labels-more":"Reading five lines with calibrated inputs"}
    catalog={"pdf_sha256":specs["pdf_sha256"],"source_url":specs["source_url"],
             "selection":"Exact same three raster crops as the earlier letter demonstration; no PDF data used to fit the new head.","examples":[]}
    scores=[]
    for item in specs['examples']:
        old=item['id']; name=old.replace('trained-','calibrated-')
        source=Path('demo/examples')/old;output=Path('demo/examples')/name
        run=record(source/'crop.png','data/graph','artifacts/letters-v2',output)
        if run['artifact_id']!=recognition['artifact_id']:raise ValueError('Checkpoint identity mismatch')
        if run['crop_sha256']!=digest_file(source/'crop.png'):raise ValueError('Comparison crop changed')
        shutil.copy2(source/'source-page.png',output/'source-page.png')
        presentation={**item,"id":name,"title":titles[old],"reference_id":old,
            "note":"One 100 ms circuit pass per glyph. Calibrated input; compact neural readout; no word correction."}
        save_json(output/'source.json',{**presentation,'pdf_sha256':specs['pdf_sha256'],'source_url':specs['source_url']})
        save_json(output/'verification.json',verify(output,'artifacts/letters-v2'))
        score=evaluate(output/'run.json',Path('examples/microsoft-2025')/(old+'-truth.json'),Path('reports/letters-v2')/(name+'.json'))
        score.update({'scope':'Same selected upright text lines; raw predictions and geometric spaces',
                      'uncertainty':'These PDF pixels informed segmentation development; not a new independent OCR benchmark.'})
        save_json(Path('reports/letters-v2')/(name+'.json'),score)
        scores.append({"id":name,"reference_id":old,"character_error_rate":score['character_error_rate'],
                       "correct_cells":score['correct_cells'],"n_cells":score['n_cells'],"wall_seconds":run['wall_seconds']})
        catalog['examples'].append(presentation)
        print(name,score['correct_cells'],'/',score['n_cells'],'exact lines;',score['character_error_rate'],'CER',flush=True)
    save_json('examples/microsoft-2025/calibrated-examples.json',catalog)
    save_json('reports/letters-v2/pdf-summary.json',{"artifact_id":recognition['artifact_id'],"examples":scores})


if __name__=='__main__':main()
