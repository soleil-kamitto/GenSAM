import json
nb = json.load(open(r'c:\Users\Sol\cellsam_project\cellsam\notebooks\cellsam_actinomicetos_clean.ipynb', encoding='utf-8'))
print(f'Total celdas: {len(nb["cells"])}')
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        outs = cell.get('outputs', [])
        types = [o['output_type'] for o in outs]
        has_img = any('image/png' in o.get('data', {}) for o in outs)
        print(f'  [{cell["id"]}]  outputs={len(outs)}  img={has_img}  {types}')
