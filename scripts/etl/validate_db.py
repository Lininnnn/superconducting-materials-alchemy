from pymongo import MongoClient
import os

PROJECT_ROOT = r'D:\works\science\材料\当前论文\论文文件夹'
client = MongoClient("mongodb://localhost:27017/")
db = client["SuperconductorDB"]
col = db["materials_v1"]

def run_audit():
    total = col.count_documents({})
    
    # 1. 检查实验值覆盖情况
    exp_count = col.count_documents({"metadata.is_experimental": True})
    
    # 2. 检查 CIF 文件物理存在情况
    # 我们查一下入库时标记为 file_exists: False 的数量
    missing_cif = col.count_documents({"structure.file_exists": False})
    
    # 3. 统计不同来源
    icsd_count = col.count_documents({"source_type": "ICSD"})
    mp_count = col.count_documents({"source_type": "MP"})

    print(f"--- 📊 数据库审计报告 ---")
    print(f"总记录数: {total}")
    print(f"ICSD 来源: {icsd_count} | MP 来源: {mp_count}")
    print(f"成功匹配实验 Tc (SuperCon): {exp_count} ({exp_count/total*100:.2f}%)")
    print(f"❌ 缺失 CIF 物理文件的记录: {missing_cif}")
    
    if missing_cif > 0:
        print("\n⚠️ 警告: 部分 CIF 路径未对齐，请检查文件名格式！")
        # 打印前5个缺失样本方便排查
        sample = col.find({"structure.file_exists": False}).limit(5)
        for s in sample:
            print(f"缺失 ID: {s['material_id']} | 预期路径: {s['structure']['cif_path']}")

if __name__ == "__main__":
    run_audit()