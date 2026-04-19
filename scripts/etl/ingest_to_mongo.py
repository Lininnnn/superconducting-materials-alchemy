import pandas as pd
from pymongo import MongoClient, UpdateOne
import os

PROJECT_ROOT = r'D:\works\science\材料\当前论文\论文文件夹'
DATA_RAW_3DSC = os.path.join(PROJECT_ROOT, 'data', 'raw', '3DSC')
DATA_PROC_SUPERCON = os.path.join(PROJECT_ROOT, 'data', 'processed', 'supercon_cleaned.csv')

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "SuperconductorDB"
COLLECTION_NAME = "materials_v1"

def load_csv_smart(path):
    """自适应读取：跳过带注释的行，直到找到包含 'sc_id' 或 'formula' 的表头"""
    for skip in range(10):  # 最多尝试跳过前10行
        try:
            df = pd.read_csv(path, skiprows=skip)
            cols = [c.lower() for c in df.columns.astype(str)]
            # 检查是否包含核心字段
            if 'sc_id' in cols or 'formula' in cols or 'formula_sc' in cols:
                return df
        except:
            continue
    return pd.read_csv(path, comment='#') # 兜底方案

def ingest_all():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]
    col.delete_many({})
    print("🗑️ 已清空旧记录。")

    sc_lookup = {}
    if os.path.exists(DATA_PROC_SUPERCON):
        df_sc = pd.read_csv(DATA_PROC_SUPERCON)
        sc_lookup = df_sc.set_index('formula_exact')['tc'].to_dict()
        print(f"✅ 加载 {len(sc_lookup)} 条 SuperCon 实验修正。")

    files_3dsc = [
        {'file': '3DSC_ICSD_only_IDs.csv', 'type': 'ICSD'},
        {'file': '3DSC_MP.csv', 'type': 'MP'}
    ]

    for item in files_3dsc:
        path = os.path.join(DATA_RAW_3DSC, item['file'])
        if not os.path.exists(path): continue
        
        df = load_csv_smart(path)
        print(f"📦 处理 {item['type']}... 识别到有效列: {df.columns.tolist()[:5]}...")

        ops = []
        for _, row in df.iterrows():
            # ID 提取优先级逻辑
            mat_id = row.get('sc_id') or row.get('material_id_2') or row.get('vca_id')
            if pd.isna(mat_id): continue
            mat_id = str(mat_id)

            # 化学式提取 (MP 叫 formula_sc, ICSD 叫 formula)
            formula = row.get('formula_sc') or row.get('formula')
            tc_3dsc = row.get('tc', 0.0)
            
            tc_exp = sc_lookup.get(str(formula), None)
            tc_final = tc_exp if tc_exp is not None else tc_3dsc
            
            cif_rel_path = os.path.join('data', 'structures', '3DSC_CIFs', item['type'], f"{mat_id}.cif")
            cif_full_path = os.path.join(PROJECT_ROOT, cif_rel_path)
            
            doc = {
                "material_id": mat_id,
                "source_type": item['type'],
                "formula": str(formula),
                "tc_target": float(tc_final),
                "metadata": {
                    "tc_3dsc": float(tc_3dsc),
                    "tc_experimental": float(tc_exp) if tc_exp else None
                },
                "structure": {
                    "cif_path": cif_rel_path,
                    "file_exists": os.path.exists(cif_full_path)
                }
            }
            ops.append(UpdateOne({"material_id": mat_id}, {"$set": doc}, upsert=True))
            
            if len(ops) >= 1000:
                col.bulk_write(ops)
                ops = []
        
        if ops:
            col.bulk_write(ops)
    
    print(f"🚀 最终入库总条数: {col.count_documents({})}")

if __name__ == "__main__":
    ingest_all()