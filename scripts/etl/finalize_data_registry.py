import os
import pandas as pd
from pymongo import MongoClient

PROJECT_ROOT = r'D:\works\science\材料\当前论文\论文文件夹'

def finalize():
    client = MongoClient("mongodb://localhost:27017/")
    col = client["SuperconductorDB"]["materials_v1"]
    
    print("开始最终路径校验与注册...")
    
    # 1. 扫描所有文档
    cursor = col.find({})
    valid_records = []
    
    for doc in cursor:
        mat_id = doc['material_id']
        source = doc['source_type'] # MP 或 ICSD
        
        # 匹配逻辑：mp-id 对应 MP 目录，icsd-id 对应 ICSD 目录
        sub_dir = "MP" if source == "MP" else "ICSD"
        file_name = f"{mat_id}.cif" if mat_id.startswith(("mp-", "icsd_")) else f"{sub_dir.lower()}_{mat_id}.cif"
        
        rel_path = os.path.join("data", "structures", "3DSC_CIFs", sub_dir, file_name)
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        
        if os.path.exists(full_path):
            # 更新数据库状态
            col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"structure.cif_path": rel_path, "structure.file_exists": True}}
            )
            # 添加到有效列表
            valid_records.append({
                "material_id": mat_id,
                "formula": doc['formula'],
                "tc": doc['tc_target'],
                "cif_path": rel_path
            })

    # 2. 生成 final_training_set.csv
    if valid_records:
        df = pd.DataFrame(valid_records)
        out_path = os.path.join(PROJECT_ROOT, "data", "processed", "final_training_set.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"✅ 成功连接 {len(df)} 个结构文件，索引表已生成：{out_path}")
    else:
        print("❌ 未匹配到任何文件，请检查文件路径和 ID 格式！")

if __name__ == "__main__":
    finalize()