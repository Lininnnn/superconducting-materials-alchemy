import pandas as pd
from pymongo import MongoClient
import os

PROJECT_ROOT = r'D:\works\science\材料\当前论文\论文文件夹'
# 请确保将 SuperMat.csv 复制到这个路径，或者修改下方的路径
SUPERMAT_CSV = os.path.join(PROJECT_ROOT, 'data', 'raw', 'SuperMat', 'SuperMat.csv')

def inject_labels_from_csv():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["SuperconductorDB"]
    col = db["materials_v1"]

    if not os.path.exists(SUPERMAT_CSV):
        print(f"❌ 未找到 SuperMat.csv: {SUPERMAT_CSV}")
        return

    print("正在读取 SuperMat.csv...")
    # 自动识别编码，防止中文或特殊字符报错
    df_mat = pd.read_csv(SUPERMAT_CSV, encoding='utf-8')
    
    # 打印前几行看看列名，确保匹配正确
    print(f"列名如下: {df_mat.columns.tolist()}")
    
    # 假设列名是 'formula' 和 'type'，如果不对请根据打印结果修改
    # 建立映射字典
    label_map = df_mat.set_index('formula')['type'].to_dict()

    print("开始同步到 MongoDB...")
    updated_count = 0
    for formula, s_type in label_map.items():
        if pd.isna(s_type): continue
        
        # 匹配化学式，更新分类字段
        result = col.update_many(
            {"formula": str(formula)},
            {"$set": {"classification": str(s_type)}}
        )
        updated_count += result.modified_count

    print(f"✅ 完成！共为 {updated_count} 条记录添加了 SuperMat 分类标签。")

if __name__ == "__main__":
    inject_labels_from_csv()