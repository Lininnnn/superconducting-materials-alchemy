import joblib
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# ============================================================
# ⚙️ 配置
# ============================================================
PROJ_ROOT = Path(r'D:\works\science\material\paper_now\paper')
SAVE_DIR = PROJ_ROOT / 'checkpoints'
CACHE_PATH = SAVE_DIR / 'v12_xgb_feature_cache.pkl'

CLASS_NAMES = ["Non-SC", "Cu-based", "Fe-based", "Other"]

def run_v12_refine():
    print("🚀 开始 V12 分类模型专项精修...")

    # 1. 加载缓存数据
    if not CACHE_PATH.exists():
        print("❌ 找不到特征缓存，请先运行训练脚本！")
        return
    X, y = joblib.load(CACHE_PATH)

    # 2. 划分数据 (必须保持与 train 相同的随机种子 42)
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]

    # 3. 加载已有模型
    try:
        xgb_bin = joblib.load(SAVE_DIR / 'v12_xgb_bin.pkl')
        xgb_multi = joblib.load(SAVE_DIR / 'v12_xgb_multi.pkl')
    except:
        print("❌ 未找到预训练模型权重。")
        return

    # ------------------------------------------------------------
    # 🛠️ 策略 A: 精修二分类器 (重点解决 Non-SC 误判)
    # ------------------------------------------------------------
    print("\n--- 正在精修二分类器 (重点优化 Non-SC 判定) ---")
    y_train_bin = (y_train > 0).astype(int)
    
    # 手动增加 Non-SC (Label 0) 的样本权重，迫使模型更关注这类错题
    sample_weights = np.where(y_train_bin == 0, 1.5, 1.0)

    xgb_bin.set_params(
        n_estimators=xgb_bin.n_estimators + 100, 
        learning_rate=0.01,
        gamma=0.1,  # 增加分裂所需的最小损失减少
        min_child_weight=2 # 增加正则化，防止过拟合到异常点
    )
    xgb_bin.fit(X_train, y_train_bin, sample_weight=sample_weights, xgb_model=xgb_bin.get_booster())

    # ------------------------------------------------------------
    # 🛠️ 策略 B: 精修三分类器
    # ------------------------------------------------------------
    print("--- 正在精修三分类器 (微调细分边界) ---")
    mask = y_train > 0
    xgb_multi.set_params(n_estimators=xgb_multi.n_estimators + 50, learning_rate=0.01)
    xgb_multi.fit(X_train[mask], y_train[mask] - 1, xgb_model=xgb_multi.get_booster())

    # 4. 推理评估
    bin_p = xgb_bin.predict(X_val)
    final_p = np.zeros_like(y_val)
    sc_idx = np.where(bin_p == 1)[0]
    if len(sc_idx) > 0:
        final_p[sc_idx] = xgb_multi.predict(X_val[sc_idx]) + 1

    # 5. 结果对比
    print("\n" + "="*45)
    print(f"✨ 精修后总分类 ACC: {accuracy_score(y_val, final_p):.2%}")
    print(f"✨ 判定是否超导 ACC: {accuracy_score((y_val > 0).astype(int), bin_p):.2%}")
    print("="*45)
    print(classification_report(y_val, final_p, target_names=CLASS_NAMES))

    # 6. 保存精修模型
    joblib.dump(xgb_bin, SAVE_DIR / 'v12_xgb_bin_refined.pkl')
    joblib.dump(xgb_multi, SAVE_DIR / 'v12_xgb_multi_refined.pkl')
    print(f"💾 精修权重已保存。")

if __name__ == "__main__":
    run_v12_refine()