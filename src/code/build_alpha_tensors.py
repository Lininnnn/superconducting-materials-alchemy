import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from pymatgen.core import Structure
from torch_geometric.data import Data
from tqdm import tqdm
import mendeleev
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# ================= 1. 物理特征缓存 (8 维) =================
print("🧠 正在构建进化版元素属性缓存...")
ELEMENT_CACHE = {}
EN_RAW_CACHE = {} 
for z in range(1, 119):
    try:
        el = mendeleev.element(z)
        is_metal = 1 if (el.is_alkali() or el.is_transition_metal() or el.is_lanthanide()) else 0
        en = float(el.en_pauling or 2.0)
        EN_RAW_CACHE[z] = en
        ELEMENT_CACHE[z] = [
            float(el.mass or 0), en, 
            float(el.atomic_radius or 1.5), float(el.group_id or 0), 
            float(el.period or 0), float(is_metal), float(z), float(el.mendeleev_number or 0)
        ]
    except:
        EN_RAW_CACHE[z] = 2.0
        ELEMENT_CACHE[z] = [float(z*2), 2.0, 1.5, 0.0, 0.0, 1.0, float(z), float(z)]

# ================= 2. 边特征：RBF 矢量化配置 (32 维) =================
DIST_CENTERS = np.linspace(0.5, 6.0, 16)
EN_DIFF_CENTERS = np.linspace(0.0, 4.0, 16)

GAMMA_DIST = 1.0 / (DIST_CENTERS[1] - DIST_CENTERS[0])**2
GAMMA_EN = 1.0 / (EN_DIFF_CENTERS[1] - EN_DIFF_CENTERS[0])**2

# ================= 3. 执行引擎 =================
def build_v12_rbf_tensors():
    BASE_DIR = Path(r'D:\works\science\material\paper_now\paper')
    CIF_DIRS = [
        BASE_DIR / 'data/structures/3DSC_CIFs/ICSD', 
        BASE_DIR / 'data/structures/3DSC_CIFs/MP'
    ]
    # 必须确保此 CSV 包含 id, tc, sc_class, weight, confidence
    SUMMARY_PATH = BASE_DIR / 'data/raw/final_training_set.csv' 
    
    # 数据库改名
    OUTPUT_DIR = BASE_DIR / 'graph_tensors_v12_rbf' 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not SUMMARY_PATH.exists():
        print(f"❌ 缺失关键索引文件: {SUMMARY_PATH}")
        return

    print("📖 正在读取索引数据...")
    df = pd.read_csv(SUMMARY_PATH)
    id_info_groups = defaultdict(list)
    for _, row in df.iterrows():
        id_info_groups[str(row['id'])].append({
            'tc': float(row['tc']), 
            'sc_class': int(row['sc_class']),
            'weight': float(row.get('weight', 1.0)),
            'conf': float(row.get('confidence', 1.0))
        })

    success = 0
    for d in CIF_DIRS:
        if not d.exists(): continue
        cif_files = list(d.glob('*.cif'))
        print(f"📂 正在处理目录: {d.name}")
        
        for cif_path in tqdm(cif_files):
            cid = cif_path.stem
            # ID 匹配逻辑
            matched_id = cid if cid in id_info_groups else cid.replace('icsd_', 'ICSD-')
            if matched_id not in id_info_groups: continue

            try:
                struct = Structure.from_file(cif_path)
                
                # A. 节点特征 (8 维)
                x = torch.tensor([ELEMENT_CACHE[s.specie.number] for s in struct], dtype=torch.float)
                
                # B. 边特征 (32 维 RBF)
                all_neighbors = struct.get_all_neighbors(r=6.0)
                edge_idx, edge_attrs = [], []
                
                for i, neighbors in enumerate(all_neighbors):
                    en_i = EN_RAW_CACHE[struct[i].specie.number]
                    for nbr in neighbors:
                        # 距离 RBF (16维)
                        dist = nbr.nn_distance
                        d_rbf = np.exp(-GAMMA_DIST * (dist - DIST_CENTERS)**2)
                        
                        # 电负性差 RBF (16维)
                        en_j = EN_RAW_CACHE[nbr.specie.number]
                        e_rbf = np.exp(-GAMMA_EN * (abs(en_i - en_j) - EN_DIFF_CENTERS)**2)
                        
                        edge_idx.append([i, nbr.index])
                        edge_attrs.append(np.concatenate([d_rbf, e_rbf]))

                edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

                # C. 保存注入了分类标签和权重的 Data 对象
                for idx, info in enumerate(id_info_groups[matched_id]):
                    data = Data(
                        x=x, 
                        edge_index=edge_index, 
                        edge_attr=edge_attr, 
                        y=torch.tensor([info['tc']], dtype=torch.float),
                        sc_class=torch.tensor([info['sc_class']], dtype=torch.long),
                        weight=torch.tensor([info['weight']], dtype=torch.float),
                        confidence=torch.tensor([info['conf']], dtype=torch.float),
                        material_id=matched_id
                    )
                    torch.save(data, OUTPUT_DIR / f"{cid}_{idx}.pt")
                    success += 1
            except Exception:
                continue
            
    print(f"\n✨ 数据库构建完成！")
    print(f"📍 存储位置: {OUTPUT_DIR}")
    print(f"📊 样本总数: {success}")
    print(f"🧬 特征维度: 节点(8) | 边(32 RBF展开)")

if __name__ == "__main__":
    build_v12_rbf_tensors()