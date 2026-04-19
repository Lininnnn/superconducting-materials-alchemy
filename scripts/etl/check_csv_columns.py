import pandas as pd
import os

PROJECT_ROOT = r'D:\works\science\材料\当前论文\论文文件夹'
path_icsd = os.path.join(PROJECT_ROOT, 'data', 'raw', '3DSC', '3DSC_ICSD_only_IDs.csv')
path_mp = os.path.join(PROJECT_ROOT, 'data', 'raw', '3DSC', '3DSC_MP.csv')

for p in [path_icsd, path_mp]:
    if os.path.exists(p):
        df = pd.read_csv(p, nrows=2)
        print(f"\n文件: {os.path.basename(p)}")
        print(f"真实列名: {df.columns.tolist()}")
    else:
        print(f"找不到文件: {p}")