import pandas as pd

comp = pd.read_csv(r"c:\Users\Sol\cellsam_project\cellsam\results\colonies\comparison_summary.csv")
scout = pd.read_csv(r"c:\Users\Sol\cellsam_project\cellsam\results\colonias\experimentos\07_scout_adaptivo\summary.csv")

rows = []
for i, row in comp.iterrows():
    sc = scout.iloc[i]
    for placa in ["A", "B"]:
        gt = row["gt_" + placa]
        csam = row["cellsam_" + placa]
        sv = sc["plate_" + placa]
        rows.append({
            "Imagen": row["image"].replace("actinomicetos_", ""),
            "Placa": placa,
            "GT": gt,
            "CellSAM": csam,
            "Scout": sv,
            "Err_CellSAM": csam - gt,
            "Err_Scout": sv - gt,
            "AE_CellSAM": abs(csam - gt),
            "AE_Scout": abs(sv - gt),
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print()
mae_c = df["AE_CellSAM"].mean()
mae_s = df["AE_Scout"].mean()
bias_c = df["Err_CellSAM"].mean()
bias_s = df["Err_Scout"].mean()
print(f"MAE CellSAM : {mae_c:.1f}")
print(f"MAE Scout   : {mae_s:.1f}")
print(f"Bias CellSAM: {bias_c:+.1f}")
print(f"Bias Scout  : {bias_s:+.1f}")
