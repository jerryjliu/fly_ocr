"""Select validation-only winner from both predeclared development batches."""
from flyocr.common import read_json, save_json
results=[read_json('reports/pilot/selected.json'),read_json('reports/contrast-pilot/selected.json')]
best=max(results,key=lambda r:r['validation_accuracy'])
best['selection_note']='Best validation result across the original 12-setting sweep and separately logged eight-setting contrast batch; frozen before any test evaluation.'
save_json('reports/frozen-pilot/selected.json',best)
print(best['config'])
