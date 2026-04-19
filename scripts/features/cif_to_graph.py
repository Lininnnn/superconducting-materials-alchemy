import os
import torch
import pandas as pd
from pymatgen.core import Structure
from torch_geometric.data import Data
from mendeleev import element
from tqdm import tqdm

# ================= 核心相对路径配置 =================
BASE_DIR = r'D:\works\science\材料\当前论文\论文文件夹'
os.chdir(BASE_DIR)

# 所有的路径现在都相对于 BASE_DIR
RAW_ROOT = os.path.join('data', 'structures')
CSV_PATH = os.path.join('data', 'processed', 'final_training_set.csv')
SAVE_DIR = "graph_tensors_v3" # 直接在根目录创建，避开长路径

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)
# ===================================================

def get_atom_features(symbol, cache):
    if symbol in cache: return cache[symbol]
    try:
        el = element(symbol)
        # 提取 9 维深度物理描述符
        feats = [
            float(el.atomic_number),
            float(el.en_pauling or 0),
            float(el.atomic_radius or 0),
            float(el.mendeleev_number or 0),
            float(el.group_id or 0),
            float(el.period or 0),
            float(el.nvalence() or 0),
            float(el.ionenergies[0] if el.ionenergies else 0),
            float(el.electron_affinity or 0)
        ]
        cache[symbol] = feats
        return feats
    except:
        return [0.0] * 9

def run_full_conversion():
    print(f"🚀 启动全量转换任务...")
    print(f"📂 目标文件夹: {os.path.abspath(SAVE_DIR)}")

    # 1. 扫描磁盘
    print("🔍 正在扫描磁盘文件...")
    disk_files = {}
    for root, _, files in os.walk(RAW_ROOT):
        for f in files:
            if f.lower().endswith('.cif'):
                disk_files[os.path.splitext(f)[0].lower()] = os.path.join(root, f)
    print(f"📢 磁盘索引完毕，共发现 {len(disk_files)} 个 CIF 文件")

    # 2. 加载数据索引
    df = pd.read_csv(CSV_PATH)
    element_cache = {}
    success_count = 0
    fail_count = 0

    # 3. 执行循环
    for i, row in tqdm(df.iterrows(), total=len(df), desc="9D特征提取"):
        mid = str(row['material_id']).lower()
        if mid.endswith('.cif'): mid = mid[:-4]
        
        cif_path = disk_files.get(mid)
        # 备选匹配
        if not cif_path:
            alt_id = os.path.splitext(os.path.basename(str(row['cif_path'])))[0].lower()
            cif_path = disk_files.get(alt_id)

        if not cif_path:
            fail_count += 1
            continue

        try:
            struct = Structure.from_file(cif_path)
            
            # 节点特征 (N x 9)
            node_feats = [get_atom_features(site.specie.symbol, element_cache) for site in struct]
            x = torch.tensor(node_feats, dtype=torch.float)
            
            # 边提取 (4.0 埃)
            edge_index = []
            all_neighbors = struct.get_all_neighbors(r=4.0)
            for idx, neighbors in enumerate(all_neighbors):
                for neighbor in neighbors:
                    edge_index.append([idx, neighbor.index])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
            
            # 标签
            y = torch.tensor([row['tc']], dtype=torch.float)
            
            # 打包
            data = Data(x=x, edge_index=edge_index, y=y)
            
            # 保存 (使用相对路径)
            save_name = f"{mid.replace('/', '_')}_v3.pt"
            torch.save(data, os.path.join(SAVE_DIR, save_name))
            success_count += 1
            
        except Exception:
            fail_count += 1

    print(f"\n✨ 转换任务结束!")
    print(f"✅ 成功生成: {success_count} 个 9D深度张量")
    print(f"❌ 失败数: {fail_count}")
    print(f"📍 文件存放在: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    run_full_conversion()