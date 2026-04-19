import os
import torch
import pandas as pd
from pymatgen.core import Structure
from torch_geometric.data import Data
from mendeleev import element
from tqdm import tqdm

# ================= 1. 核心路径设定 (相对路径化) =================
# 我们先切换到论文文件夹，缩小路径长度
BASE_DIR = r'D:\works\science\材料\当前论文\论文文件夹'
os.chdir(BASE_DIR)

# 使用相对于 BASE_DIR 的路径
RAW_ROOT = os.path.join('data', 'structures')
CSV_PATH = os.path.join('data', 'processed', 'final_training_set.csv')
# 我们直接把生成的测试文件放在当前目录下，看看能不能写进去
TEST_SAVE_DIR = "test_tensors" 

if not os.path.exists(TEST_SAVE_DIR):
    os.makedirs(TEST_SAVE_DIR, exist_ok=True)
# =============================================================

def get_atom_features(symbol, cache):
    if symbol in cache: return cache[symbol]
    try:
        el = element(symbol)
        feats = [float(el.atomic_number), float(el.en_pauling or 0), float(el.atomic_radius or 0),
                 float(el.mendeleev_number or 0), float(el.group_id or 0), float(el.period or 0),
                 float(el.nvalence() or 0), float(el.ionenergies[0] if el.ionenergies else 0),
                 float(el.electron_affinity or 0)]
        cache[symbol] = feats
        return feats
    except: return [0.0] * 9

def smoke_test():
    print(f"🚀 切换工作目录至: {os.getcwd()}")
    print(f"📂 测试保存目录: {os.path.abspath(TEST_SAVE_DIR)}")

    # 磁盘扫描
    disk_files = {}
    for root, _, files in os.walk(RAW_ROOT):
        for f in files:
            if f.lower().endswith('.cif'):
                disk_files[os.path.splitext(f)[0].lower()] = os.path.join(root, f)
    
    print(f"📢 扫描到 {len(disk_files)} 个文件")

    df = pd.read_csv(CSV_PATH).head(5)
    element_cache = {}
    success = 0

    for i, row in df.iterrows():
        mid = str(row['material_id']).lower()
        cif_path = disk_files.get(mid)
        if not cif_path: continue
            
        try:
            struct = Structure.from_file(cif_path)
            node_feats = [get_atom_features(site.specie.symbol, element_cache) for site in struct]
            x = torch.tensor(node_feats, dtype=torch.float)
            y = torch.tensor([row['tc']], dtype=torch.float)
            data = Data(x=x, y=y)
            
            # --- 改变保存方式：先进入目录，再保存文件名 ---
            file_name = f"test_{mid}.pt"
            # 尝试直接保存
            save_path = os.path.join(TEST_SAVE_DIR, file_name)
            torch.save(data, save_path)
            
            if os.path.exists(save_path):
                print(f"✅ [样本 {i+1}] 成功！保存于 {save_path}")
                success += 1
        except Exception as e:
            print(f"❌ [样本 {i+1}] 失败: {e}")

    if success > 0:
        print(f"\n🟢 突破成功！已成功保存 {success} 个文件到 {TEST_SAVE_DIR}")
        print("💡 既然这个目录能写，我们就把全量数据也转到这个新文件夹下。")
    else:
        print("\n🔴 依然失败。这暗示 D 盘可能存在某种写保护或路径字符集冲突。")
        print("👉 请尝试将代码中的 TEST_SAVE_DIR 改为 'C:\\Users\\Lenovo\\Desktop\\test_tensors' 看看桌面上能不能写。")

if __name__ == "__main__":
    smoke_test()