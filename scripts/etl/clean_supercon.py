import pandas as pd
import os

PROJECT_ROOT = r'D:\works\science\材料\当前论文\论文文件夹'
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'SuperCon', 'primary.tsv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'supercon_cleaned.csv')

def refine_supercon():
    print(f"正在深度处理: {INPUT_FILE}")
    
    # 1. 跳过前两行描述行，手动指定列名
    # 根据你的输出：6是化学式，92是Tc，198是期刊
    df = pd.read_csv(INPUT_FILE, sep='\t', skiprows=2, header=None, 
                     usecols=[1, 2, 4, 5, 6], # 根据预览选择需要的索引
                     names=['id', 'formula_common', 'formula_exact', 'tc_unit', 'tc'])

    # 2. 清洗数据
    # 强制将 Tc 转换为数字，无法转换的变为 NaN 随后剔除
    df['tc'] = pd.to_numeric(df['tc'], errors='coerce')
    df = df.dropna(subset=['tc', 'formula_exact'])
    
    # 3. 简单的化学式去重（取同一种材料 Tc 的平均值，或根据需求取最大）
    df_clean = df.groupby('formula_exact').agg({
        'tc': 'mean',
        'formula_common': 'first',
        'id': 'first'
    }).reset_index()

    print(f"清理完成。有效数据条数: {len(df_clean)}")
    print(df_clean.head())

    # 4. 保存
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 最终对齐表已生成: {OUTPUT_FILE}")

if __name__ == "__main__":
    refine_supercon()