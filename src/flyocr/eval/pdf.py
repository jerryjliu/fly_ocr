"""Separate post-hoc evaluator. Never imported by the recognizer."""
from flyocr.common import read_json, save_json, digest_file
from flyocr.eval.metrics import edit_distance


def evaluate(run_path, truth_path, output):
    run, truth = read_json(run_path), read_json(truth_path)
    predicted = {r["row_id"]: r for r in run["rows"]}
    rows, errors, chars = [], 0, 0
    for row in truth["rows"]:
        actual = predicted.get(row["row_id"], {}).get("raw", "")
        error = edit_distance(row["raw"], actual)
        rows.append({"row_id": row["row_id"], "truth": row["raw"], "prediction": actual, "exact": row["raw"] == actual,
                     "edit_distance": error, "expected_glyphs": len(row["raw"]), "segmented_glyphs": len(actual)})
        errors += error; chars += len(row["raw"])
    extras = [r for r in run["rows"] if r["row_id"] not in {x["row_id"] for x in truth["rows"]}]
    errors += sum(len(r["raw"]) for r in extras)
    n = len(rows); correct = sum(r["exact"] for r in rows)
    report = {"run_sha256": digest_file(run_path), "truth_sha256": digest_file(truth_path),
        "n_cells": n, "correct_cells": correct, "cell_exact_match": correct/n,
        "character_error_rate": errors/chars, "edit_errors": errors, "truth_characters": chars,
        "coverage": sum(r["row_id"] in predicted for r in truth["rows"])/n,
        "glyph_count_match_cells": sum(r["expected_glyphs"] == r["segmented_glyphs"] for r in rows),
        "extra_rows": extras, "rows": rows, "threshold_met": correct/n >= .8 and n >= 15,
        "scope": "one manually selected column; automatic pixel glyph segmentation; no text correction",
        "uncertainty": "One document column; not evidence of general PDF OCR performance."}
    if "layout" in run:
        expected = truth.get("shape")
        actual = [run["layout"]["n_rows"], run["layout"]["n_columns"]]
        report.update({"scope": "manually selected numeric table region; pixel column and row inference; no correction",
            "uncertainty": "Selected examples, not a representative general table benchmark.",
            "expected_shape": expected, "predicted_shape": actual, "shape_match": expected == actual,
            "table_exact_match": expected == actual and correct == n and not extras})
    save_json(output, report)
    return report
