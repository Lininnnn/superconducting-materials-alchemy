import joblib
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

PROJ_ROOT = Path(r'/path/to/your/dataset')
SAVE_DIR = PROJ_ROOT / 'checkpoints'
CACHE_PATH = SAVE_DIR / 'xgb_feature_cache.pkl'

CLASS_NAMES = ["Non-SC", "Cu-based", "Fe-based", "Other"]

def run_refine():

    if not CACHE_PATH.exists():
        print("The feature cache cannot be found. Please run the training script first!")
        return
    X, y = joblib.load(CACHE_PATH)

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]

    try:
        xgb_bin = joblib.load(SAVE_DIR / 'xgb_bin.pkl')
        xgb_multi = joblib.load(SAVE_DIR / 'xgb_multi.pkl')
    except:
        print("The pre-trained model weights were not found.")
        return

    y_train_bin = (y_train > 0).astype(int)
    
    sample_weights = np.where(y_train_bin == 0, 1.5, 1.0)

    xgb_bin.set_params(
        n_estimators=xgb_bin.n_estimators + 100, 
        learning_rate=0.01,
        gamma=0.1, 
        min_child_weight=2 
    )
    xgb_bin.fit(X_train, y_train_bin, sample_weight=sample_weights, xgb_model=xgb_bin.get_booster())

    mask = y_train > 0
    xgb_multi.set_params(n_estimators=xgb_multi.n_estimators + 50, learning_rate=0.01)
    xgb_multi.fit(X_train[mask], y_train[mask] - 1, xgb_model=xgb_multi.get_booster())

    bin_p = xgb_bin.predict(X_val)
    final_p = np.zeros_like(y_val)
    sc_idx = np.where(bin_p == 1)[0]
    if len(sc_idx) > 0:
        final_p[sc_idx] = xgb_multi.predict(X_val[sc_idx]) + 1

    print("\n" + "="*45)
    print(f"Final total classification after meticulous refinement ACC: {accuracy_score(y_val, final_p):.2%}")
    print(f"Determine whether it is superconducting ACC: {accuracy_score((y_val > 0).astype(int), bin_p):.2%}")
    print("="*45)
    print(classification_report(y_val, final_p, target_names=CLASS_NAMES))

    joblib.dump(xgb_bin, SAVE_DIR / 'xgb_bin_refined.pkl')
    joblib.dump(xgb_multi, SAVE_DIR / 'xgb_multi_refined.pkl')

if __name__ == "__main__":
    run_refine()
