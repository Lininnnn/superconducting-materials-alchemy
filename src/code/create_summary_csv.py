import pandas as pd
from pathlib import Path
from pymatgen.core import Composition

# ================= routes configuration =================
BASE_DIR = Path(r'/path/to/your/dataset')
RAW_3DSC_DIR = BASE_DIR / 'data'
OUTPUT_CSV = BASE_DIR / 'data/final_training_set.csv'

MP_CSV = RAW_3DSC_DIR / '3DSC_MP.csv'
ICSD_CSV = RAW_3DSC_DIR / '3DSC_ICSD_only_IDs.csv'

# ================= mapping relation =================
CLASS_MAP = {
    'Non-SC': 0,
    'Cu-based': 1,
    'Fe-based': 2,
    'Other': 3
}

def get_class_index(tc, formula):

    if tc <= 0:
        return CLASS_MAP['Non-SC']

    try:
        comp = Composition(formula)
        el_set = {el.symbol for el in comp.elements}
        n_elem = len(el_set)

        if (
            'Cu' in el_set
            and 'O' in el_set
            and n_elem >= 3
        ):

            if not (
                'P' in el_set and 'O' in el_set
            ):
                return CLASS_MAP['Cu-based']

        fe_anions = {'As', 'P', 'Se', 'S', 'Te'}
        if (
            'Fe' in el_set
            and len(el_set & fe_anions) > 0
        ):
            if not (
                'O' in el_set and n_elem > 5
            ):
                if not (
                    'F' in el_set and n_elem <= 3
                ):
                    return CLASS_MAP['Fe-based']

    except Exception:
        pass

    return CLASS_MAP['Other']

def build_full_summary():
    all_data = []

    if MP_CSV.exists():
        print(f"processing: {MP_CSV.name}")
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
                'sc_class': get_class_index(tc_val, formula_val),
                'weight': 1.0, 
                'source': '3DSC_MP'
            })

    if ICSD_CSV.exists():
        print(f"processing: {ICSD_CSV.name}")
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

    if all_data:
        final_df = pd.DataFrame(all_data)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nData processing is complete! Numerical labels have been generated.")
        print(f"mapping relation: {CLASS_MAP}")
        print(f"The number of various types of samples:\n{final_df['sc_class'].value_counts().sort_index()}")
        print(f"path: {OUTPUT_CSV}")
    else:
        print("No data was extracted.")

if __name__ == "__main__":
    build_full_summary()
