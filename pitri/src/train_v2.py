import os
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report

# ===== PATH =====
TRAIN = "pitri/data/processed/feat_train.parquet"
TEST = "pitri/data/processed/feat_test.parquet"
MODEL_OUT = "pitri/models/lightgbm_final.txt"
SUB_OUT = "pitri/results/submission_pitri.csv"

os.makedirs("pitri/models", exist_ok=True)
os.makedirs("pitri/results", exist_ok=True)

print("=== 1. CEK KOLOM ===")
cols = pq.ParquetFile(TRAIN).schema.names
print(f"Total kolom train: {len(cols)}")

# Kolom yang dibuang biar RAM ga jebol - ini kolom ID / string gede
DROP = [
    'shipment_id','customer_id','courier_id','hub_id','package_id','promo_id',
    'kota_asal','kota_tujuan','cus_region','cus_segment','cus_business',
    'co_hub','co_vehicle','prm_name','first_hub','last_hub','moda_first','moda_last'
]
use_cols = [c for c in cols if c not in DROP and c != 'status_penanganan']
print(f"Fitur yang dipakai: {len(use_cols)}")
print(f"Fitur yang dibuang: {DROP}")

# ===== 2. BACA DATA - CUMA KOLOM KEPAKE =====
print("\n=== 2. BACA DATA ===")
train = pd.read_parquet(TRAIN, columns=use_cols + ['status_penanganan'])
test = pd.read_parquet(TEST, columns=use_cols + ['shipment_id'])
print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")

# ===== 3. ENCODE TARGET =====
print("\n=== 3. ENCODE TARGET ===")
le = LabelEncoder()
y = le.fit_transform(train['status_penanganan'].astype(str))
print("Classes:", list(le.classes_))
print(pd.Series(y).value_counts(normalize=True))

# ===== 4. ENCODE FITUR KATEGORIKAL =====
print("\n=== 4. ENCODE KATEGORIKAL ===")
for c in use_cols:
    if train[c].dtype == 'object':
        train[c] = train[c].astype('category')
        # samain kategori di test biar ga error
        test[c] = pd.Categorical(test[c].astype(str), categories=train[c].cat.categories)
        print(f" - {c} jadi category")

X = train[use_cols]
X_test = test[use_cols]

# ===== 5. SPLIT VALIDASI =====
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain split: {X_tr.shape}, Val split: {X_val.shape}")

# ===== 6. TRAINING =====
print("\n=== 5. TRAINING MULAI ===")
model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=42,
    verbose=100
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100)]
)

print(f"\nBest iteration: {model.best_iteration_}")

# ===== 7. CEK SCORE VALIDASI LOKAL - INI SCORE LO =====
print("\n=== 6. SCORE VALIDASI LOKAL LO ===")
pred_val = model.predict(X_val)
proba_val = model.predict_proba(X_val)
acc = accuracy_score(y_val, pred_val)
ll = log_loss(y_val, proba_val)
print(f"Accuracy Val: {acc:.5f}")
print(f"LogLoss Val: {ll:.5f}")
print("\nClassification Report:")
print(classification_report(y_val, pred_val, target_names=list(le.classes_)))

# ===== 8. SIMPAN MODEL =====
print("\n=== 7. SIMPAN MODEL ===")
model.booster_.save_model(MODEL_OUT)
print(f"Model tersimpan: {MODEL_OUT}")

# ===== 9. PREDIKSI TEST + BUAT SUBMISSION =====
print("\n=== 8. BUAT SUBMISSION ===")
proba_test = model.predict_proba(X_test)
sub = pd.DataFrame(proba_test, columns=list(le.classes_))
sub.insert(0, 'shipment_id', test['shipment_id'])
sub.to_csv(SUB_OUT, index=False)
print(f"Submission tersimpan: {SUB_OUT}")
print(f"Submission shape: {sub.shape}")
print(sub.head())
print("\n=== SELESAI ===")
print(f"Score lokal lo (Accuracy): {acc:.5f}")
print(f"Score Ghani (Leaderboard): 0.60163")
print("Kalo score lokal lo udah 0.58+, langsung bisa di-ensemble sama Ghani!")