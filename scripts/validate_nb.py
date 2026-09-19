import json
path = r'c:\Users\Sol\cellsam_project\cellsam\notebooks\cellsam_actinomicetos_clean.ipynb'
nb = json.load(open(path, encoding='utf-8'))
print(f'nbformat: {nb["nbformat"]}.{nb["nbformat_minor"]}')
print(f'Celdas: {len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    print(f'  [{i:02d}] {c["cell_type"]:8}  id={c["id"]}')
