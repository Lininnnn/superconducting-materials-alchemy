import pandas as pd
import os

PROJECT_ROOT = r'D:\works\science\材料\当前论文\论文文件夹'
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'SuperCon', 'primary.tsv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'supercon_cleaned.csv')

def refine_supercon():
    print("🚀 开始最终清洗 SuperCon 数据...")
    
    # 根据 debug 结果：索引 2 是公式，索引 5 是 Tc
    # skiprows=3 是因为：0:数字索引, 1:标题, 2:单位/简称
    try:
        df = pd.read_csv(INPUT_FILE, sep='\t', skiprows=3, header=None, 
                         usecols=[2, 5], 
                         names=['formula_exact', 'tc'])
        
        # 1. 强制数值转换
        df['tc'] = pd.to_numeric(df['tc'], errors='coerce')
        
        # 2. 剔除无效值
        initial_count = len(df)
        df = df.dropna(subset=['tc', 'formula_exact'])
        
        # 3. 数据去重（同一物质取平均实验温）
        df_clean = df.groupby('formula_exact')['tc'].mean().reset_index()
        
        print(f"📊 原始条数: {initial_count} | 清洗后条数: {len(df_clean)}")
        print(df_clean.head())

        # 4. 保存结果
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_clean.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ 成功！清洗后的数据已存至: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ 清洗失败: {e}")

if __name__ == "__main__":
    refine_supercon()