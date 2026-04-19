import pandas as pd
import os

PROJECT_ROOT = r'D:\works\science\材料\当前论文\论文文件夹'
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'SuperCon', 'primary.tsv')

def debug_columns():
    # 1. 读入两行，看看真实结构
    df_preview = pd.read_csv(INPUT_FILE, sep='\t', nrows=5, header=None)
    
    print("\n--- 原始列索引与内容对应关系 ---")
    for i, col_content in enumerate(df_preview.iloc[0]): # 第一行描述
        second_row = df_preview.iloc[1, i] # 第二行单位/简称
        third_row = df_preview.iloc[2, i]  # 第三行实际数据示例
        print(f"索引 [{i}]: '{col_content}' | 子标题: '{second_row}' | 示例数据: '{third_row}'")

if __name__ == "__main__":
    debug_columns()