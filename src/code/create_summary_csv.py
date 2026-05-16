import pandas as pd
from pathlib import Path

# ================= 路径配置 =================
BASE_DIR = Path(r'D:\works\science\material\paper_now\paper')
RAW_3DSC_DIR = BASE_DIR / 'data/raw/3DSC'
OUTPUT_CSV = BASE_DIR / 'data/raw/final_training_set.csv'

MP_CSV = RAW_3DSC_DIR / '3DSC_MP.csv'
ICSD_CSV = RAW_3DSC_DIR / '3DSC_ICSD_only_IDs.csv'

# ================= 映射关系 =================
# 0: Non-SC, 1: Cu-based, 2: Fe-based, 3: Other
CLASS_MAP = {
    'Non-SC': 0,
    'Cu-based': 1,
    'Fe-based': 2,
    'Other': 3
}

def get_class_index(tc, formula):
    """
    根据 Tc 和化学式返回数值标签
    """
    if tc <= 0:
        return CLASS_MAP['Non-SC']
    
    f_lower = str(formula).lower()
    
    # 铜基判断
    if 'cu' in f_lower and any(x in f_lower for x in ['o', 's', 'se']):
        return CLASS_MAP['Cu-based']
    
    # 铁基判断
    if 'fe' in f_lower and any(x in f_lower for x in ['p', 'as', 'se', 's']):
        return CLASS_MAP['Fe-based']
    
    return CLASS_MAP['Other']

def build_full_summary():
    all_data = []

    # 1. 处理 MP 数据
    if MP_CSV.exists():
        print(f"📖 正在处理: {MP_CSV.name}")
        df_mp = pd.read_csv(MP_CSV, comment='#', skipinitialspace=True)
        df_mp.columns = df_mp.columns.str.strip().str.replace('"', '')
        
        id_col = 'material_id_2' if 'material_id_2' in df_mp.columns else 'material_id'
        f_col = 'formula' if 'formula' in df_mp.columns else 'formula_sc'
        
        for _, row in df_mp.dropna(subset=[id_col, 'tc']).iterrows():
            m_id = str(row[id_col])
            full_id = m_id if m_id.startswith('mp-') else f"mp-{m_id}"
            tc_val = float(row['tc'])
            formula_val = row.get(f_col, '')
            
            all_data.append({
                'id': full_id,
                'tc': tc_val,
                'formula': formula_val,
                'sc_class': get_class_index(tc_val, formula_val), # 存储 0,1,2,3
                'weight': 1.0, # 初始权重全部为 1
                'source': '3DSC_MP'
            })

    # 2. 处理 ICSD 数据
    if ICSD_CSV.exists():
        print(f"📖 正在处理: {ICSD_CSV.name}")
        df_icsd = pd.read_csv(ICSD_CSV, skiprows=1, skipinitialspace=True)
        df_icsd.columns = df_icsd.columns.str.strip().str.replace('"', '').str.replace("'", "")
        
        if 'database_id_2' in df_icsd.columns and 'tc' in df_icsd.columns:
            f_col = 'formula_sc' if 'formula_sc' in df_icsd.columns else 'formula'
            
            for _, row in df_icsd.dropna(subset=['database_id_2', 'tc']).iterrows():
                raw_id = str(row['database_id_2'])
                full_id = raw_id if raw_id.startswith('ICSD-') else f"ICSD-{raw_id}"
                tc_val = float(row['tc'])
                formula_val = row.get(f_col, '')
                
                all_data.append({
                    'id': full_id,
                    'tc': tc_val,
                    'formula': formula_val,
                    'sc_class': get_class_index(tc_val, formula_val),
                    'weight': 1.0,
                    'source': '3DSC_ICSD'
                })

    # 3. 保存结果
    if all_data:
        final_df = pd.DataFrame(all_data)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🚀 数据处理完毕！数值标签已生成。")
        print(f"📊 映射关系: {CLASS_MAP}")
        print(f"📊 各类样本数:\n{final_df['sc_class'].value_counts().sort_index()}")
        print(f"📍 路径: {OUTPUT_CSV}")
    else:
        print("❌ 未提取到数据。")

if __name__ == "__main__":
    build_full_summary()