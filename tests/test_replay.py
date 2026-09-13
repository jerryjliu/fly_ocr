from pathlib import Path
import shutil
import pytest
from flyocr.demo.schema import verify
from flyocr.common import read_json, save_json


def test_saved_predictions_are_derived_from_recorded_neural_features(tmp_path):
    source = Path("demo/run")
    if not (source/"run.json").exists(): pytest.skip("Requires bundled replay")
    assert verify(source, "artifacts/numeric")["verified_events"] == 120
    shutil.copy(source/"crop.png", tmp_path/"crop.png")
    (tmp_path/"glyphs").symlink_to((source/"glyphs").resolve(), target_is_directory=True)
    altered = read_json(source/"run.json")
    altered["events"][0]["character"] = "0"
    save_json(tmp_path/"run.json", altered)
    with pytest.raises(ValueError, match="prediction mismatch"):
        verify(tmp_path, "artifacts/numeric")


def test_table_exports_are_the_same_raw_predictions_and_positions(tmp_path):
    source = Path("demo/examples/income-table")
    if not (source/"run.json").exists(): pytest.skip("Requires recorded table example")
    assert verify(source, "artifacts/numeric")["verified_events"] == 108
    for name in ["run.json", "crop.png", "table.json", "table.csv"]:
        shutil.copy(source/name, tmp_path/name)
    (tmp_path/"glyphs").symlink_to((source/"glyphs").resolve(), target_is_directory=True)
    table = read_json(tmp_path/"table.json")
    table["matrix"][0][0] = "repaired value"
    save_json(tmp_path/"table.json", table)
    with pytest.raises(ValueError, match="prediction mismatch"):
        verify(tmp_path, "artifacts/numeric")


def test_letter_replay_recomputes_geometric_spaces(tmp_path):
    source = Path("demo/examples/trained-heading")
    if not (source/"run.json").exists(): pytest.skip("Requires trained letter replay")
    assert verify(source,"artifacts/letters")["verified_events"] == 13
    shutil.copy(source/"crop.png",tmp_path/"crop.png")
    (tmp_path/"glyphs").symlink_to((source/"glyphs").resolve(),target_is_directory=True)
    altered=read_json(source/"run.json")
    altered["events"][1]["prefix"]=" "
    altered["rows"][0]["raw"]="".join(e.get("prefix","")+e["character"] for e in altered["events"])
    save_json(tmp_path/"run.json",altered)
    with pytest.raises(ValueError,match="prediction mismatch"):verify(tmp_path,"artifacts/letters")
