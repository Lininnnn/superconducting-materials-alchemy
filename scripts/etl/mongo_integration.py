import pandas as pd
from pymongo import MongoClient
import os

# 路径配置
PROJECT_ROOT = r'D:\works\science\材料\当前论文\论文文件夹'
PATH_3DSC_ICSD = os.path.join(PROJECT_ROOT, 'data', 'raw', '3DSC', '3DSC_ICSD_only_IDs.csv')
PATH_SUPERCON = os.path.join(PROJECT_ROOT, 'data', 'processed', 'supercon_cleaned.csv')
CIF_DIR_ICSD = os.path.join(PROJECT_ROOT, 'data', 'structures', '3DSC_CIFs', 'ICSD')

def integrate_to_mongodb():
    # 1. 连接数据库
    client = MongoClient('mongodb://localhost:27017/')
    db = client['SuperconductorDB']
    col = db['materials']
    
    # 2. 读取数据源
    print("读取数据源...")
    df_3dsc = pd.read_csv(PATH_3DSC_ICSD)
    df_sc = pd.read_csv(PATH_SUPERCON)
    
    # 将 SuperCon 转换为字典方便快速查询
    sc_lookup = df_sc.set_index('formula_exact')['tc'].to_dict()
    
    count = 0
    print("开始整合并入库...")
    
    for _, row in df_3dsc.iterrows():
        # 提取 3DSC 基础信息
        sc_id = str(row['sc_id']) # 假设 ID 列名为 sc_id
        formula = str(row['formula'])
        tc_3dsc = row['tc']
        
        # 尝试匹配 SuperCon 的实验值
        tc_experimental = sc_lookup.get(formula, None)
        
        # 确定最终用于训练的 Tc (优先使用 SuperCon)
        tc_final = tc_experimental if tc_experimental is not None else tc_3dsc
        
        # 构建 CIF 相对路径 (请根据你实际的文件名微调)
        cif_filename = f"{sc_id}.cif" 
        cif_path = os.path.join('data', 'structures', '3DSC_CIFs', 'ICSD', cif_filename)
        
        # 构建 MongoDB 文档
        doc = {
            "id_3dsc": sc_id,
            "formula": formula,
            "tc_final": tc_final,
            "tc_sources": {
                "3dsc": tc_3dsc,
                "supercon": tc_experimental
            },
            "structure": {
                "cif_path": cif_path,
                "source": "ICSD"
            },
            "status": {
                "has_experimental_tc": tc_experimental is not None,
                "has_cif": os.path.exists(os.path.join(PROJECT_ROOT, cif_path))
            }
        }
        
        # Upsert 操作
        col.update_one({"id_3dsc": sc_id}, {"$set": doc}, upsert=True)
        count += 1
        
    print(f"✅ 成功整合 {count} 条记录到 MongoDB！")

if __name__ == "__main__":
    integrate_to_mongodb()