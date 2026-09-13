from pathlib import Path
import subprocess
from PIL import Image
from flyocr.common import save_json, digest_file


def render(pdf, output, page=40, dpi=300, region=None):
    if page < 1 or not 72 <= dpi <= 600: raise ValueError("Invalid page or raster resolution")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True, check=True).stderr.splitlines()[0]
    prefix = output/"page"
    subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-gray", "-png", "-singlefile", str(pdf), str(prefix)], check=True)
    image = Image.open(prefix.with_suffix(".png")).convert("L")
    if region is None: region = [0, 0, 1, 1]
    if len(region) != 4 or not 0 <= region[0] < region[2] <= 1 or not 0 <= region[1] < region[3] <= 1:
        raise ValueError("Invalid normalized crop")
    box = [round(region[i]*(image.width if i%2 == 0 else image.height)) for i in range(4)]
    image.crop(box).save(output/"crop.png")
    manifest = {"pdf_sha256": digest_file(pdf), "page_one_based": page, "dpi": dpi,
        "renderer": version, "region_normalized": region, "crop_page_box": box,
        "crop_sha256": digest_file(output/"crop.png"), "uses_embedded_text": False}
    save_json(output/"raster.json", manifest)
    return output/"crop.png"
