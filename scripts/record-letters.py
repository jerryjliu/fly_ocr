"""Record the declared real-document letter examples, then evaluate separately."""
from pathlib import Path
import json
import shutil
from PIL import Image
from flyocr.common import read_json,save_json,digest_file
from flyocr.pdf.render import render
from flyocr.demo.record import record
from flyocr.demo.schema import verify
from flyocr.eval.pdf import evaluate


def main():
    specs=read_json('examples/microsoft-2025/letter-examples.json')
    source=Path('data/source/microsoft-2025.pdf')
    if digest_file(source)!=specs['pdf_sha256']:raise ValueError('Source report changed')
    # A completed held-out report must precede target-PDF inference.
    recognition=read_json('artifacts/letters/recognition.json')
    for item in specs['examples']:
        name=item['id']; raster=Path('data/letter-pdf')/name; output=Path('demo/examples')/name
        crop=render(source,raster,item['page'],300,item['region'])
        run=record(crop,'data/graph','artifacts/letters',output)
        if run['artifact_id']!=recognition['artifact_id']:raise ValueError('Checkpoint identity mismatch')
        (output/'run.json').write_text(json.dumps(run,separators=(',',':')))
        page=Image.open(raster/'page.png');page.thumbnail((650,850));page.save(output/'source-page.png')
        save_json(output/'source.json',{**item,'pdf_sha256':specs['pdf_sha256'],'source_url':specs['source_url'],'selection':specs['selection']})
        save_json(output/'verification.json',verify(output,'artifacts/letters'))
        score=evaluate(output/'run.json',Path('examples/microsoft-2025')/(name+'-truth.json'),Path('reports/letters')/(name+'.json'))
        score.update({'scope':'Selected upright printed text lines, with geometric spaces; no text correction','uncertainty':'PDF pixels informed segmentation development; not a general OCR benchmark.'})
        save_json(Path('reports/letters')/(name+'.json'),score)
        print(name,score['correct_cells'],'/',score['n_cells'],'exact lines;',score['character_error_rate'],'CER',flush=True)


if __name__=='__main__':main()
