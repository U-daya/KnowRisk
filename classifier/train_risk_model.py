#!/usr/bin/env python3
"""
train_risk_model.py
Trains a GradientBoostingClassifier on the DataCo Supply Chain dataset.
Falls back to synthetic data if the CSV is missing.

Outputs:
  classifier/risk_model.joblib
  classifier/feature_columns.json
  classifier/metrics.json
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "DataCoSupplyChainDataset.csv"
MODEL_PATH = SCRIPT_DIR / "risk_model.joblib"
FEATURES_PATH = SCRIPT_DIR / "feature_columns.json"
METRICS_PATH = SCRIPT_DIR / "metrics.json"


# ── Real DataCo dataset ───────────────────────────────────────────────────

def load_real_data() -> tuple[pd.DataFrame, str]:
    """Load and engineer features from DataCoSupplyChainDataset.csv.

    Target: Late_delivery_risk (binary 0/1) - the canonical DataCo Kaggle task.
    This column encodes whether the shipping-mode + scheduled-days combination
    is inherently risky, NOT whether the shipment actually arrived late, so
    using scheduled_ship_days and shipping_mode_enc as features is not leakage.
    """
    print(f"📂 Loading real dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="latin-1", low_memory=False)
    print(f"   Raw shape: {df.shape}")

    # ── Feature engineering (causal — known at order time) ───────────────
    engineered = pd.DataFrame()

    # Scheduled shipping days (planned, not actual — no leakage)
    if "Days for shipment (scheduled)" in df.columns:
        engineered["scheduled_ship_days"] = pd.to_numeric(
            df["Days for shipment (scheduled)"], errors="coerce"
        ).clip(0, 30).fillna(5)
    else:
        engineered["scheduled_ship_days"] = 5

    # Discount rate (aggressive discounts correlate with fraud)
    if "Order Item Discount Rate" in df.columns:
        engineered["discount_rate"] = pd.to_numeric(
            df["Order Item Discount Rate"], errors="coerce"
        ).clip(0, 1).fillna(0)
        engineered["high_discount_flag"] = (engineered["discount_rate"] > 0.3).astype(int)
    else:
        engineered["discount_rate"] = 0.0
        engineered["high_discount_flag"] = 0

    # Order value
    if "Sales" in df.columns:
        engineered["order_value"] = pd.to_numeric(
            df["Sales"], errors="coerce"
        ).fillna(0).clip(0, 10000)
    elif "Order Item Total" in df.columns:
        engineered["order_value"] = pd.to_numeric(
            df["Order Item Total"], errors="coerce"
        ).fillna(0)
    else:
        engineered["order_value"] = 0

    # Profit per order (negative = loss-leader or pricing error)
    if "Benefit per order" in df.columns:
        engineered["benefit_per_order"] = pd.to_numeric(
            df["Benefit per order"], errors="coerce"
        ).clip(-500, 500).fillna(0)
    else:
        engineered["benefit_per_order"] = 0

    # Shipping mode (planned at order time)
    if "Shipping Mode" in df.columns:
        ship_enc = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 3}
        engineered["shipping_mode_enc"] = (
            df["Shipping Mode"].map(ship_enc).fillna(2)
        )
    else:
        engineered["shipping_mode_enc"] = 2

    # Market geography (known at order time)
    if "Market" in df.columns:
        market_risk = {
            "Africa": 0.7, "Pacific Asia": 0.5, "LATAM": 0.5,
            "Europe": 0.2, "USCA": 0.1,
        }
        engineered["market_geo_risk"] = df["Market"].map(market_risk).fillna(0.4)
    else:
        engineered["market_geo_risk"] = 0.3

    # Category (known at order time)
    if "Category Name" in df.columns:
        cat_risk = {
            "Electronics": 0.5, "Computers": 0.5, "Smartphones": 0.6,
            "Cameras": 0.4, "Sporting Goods": 0.3, "Clothing": 0.2,
            "Office Supplies": 0.1, "Furniture": 0.15,
        }
        engineered["category_risk"] = df["Category Name"].map(cat_risk).fillna(0.25)
    else:
        engineered["category_risk"] = 0.25

    # Product price (known at order time)
    price_col = "Product Price" if "Product Price" in df.columns else "Order Item Product Price"
    if price_col in df.columns:
        engineered["product_price_log"] = np.log1p(
            pd.to_numeric(df[price_col], errors="coerce").fillna(0).clip(0)
        )
    else:
        engineered["product_price_log"] = 0

    # Order quantity
    if "Order Item Quantity" in df.columns:
        engineered["order_quantity"] = pd.to_numeric(
            df["Order Item Quantity"], errors="coerce"
        ).clip(0, 100).fillna(1)
    else:
        engineered["order_quantity"] = 1

    # Target: Late_delivery_risk (canonical DataCo binary prediction task)
    if "Late_delivery_risk" in df.columns:
        y = pd.to_numeric(df["Late_delivery_risk"], errors="coerce").fillna(0).astype(int)
    elif "Order Status" in df.columns:
        high_risk = {"SUSPECTED_FRAUD", "CANCELED", "LATE"}
        y = df["Order Status"].isin(high_risk).astype(int)
    else:
        y = (engineered["benefit_per_order"] < -50).astype(int)

    X = engineered.fillna(0)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    print(f"   Engineered features: {list(X.columns)}")
    print(f"   Target (Late_delivery_risk) class balance: {y.value_counts().to_dict()}")
    return X, y, "real_dataco"


# ── Synthetic fallback ────────────────────────────────────────────────────

def generate_synthetic_data(n: int = 5000) -> tuple[pd.DataFrame, pd.Series, str]:
    print("WARNING: CSV missing -- using synthetic fallback data")
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "scheduled_ship_days": rng.integers(1, 7, n).astype(float),
        "discount_rate":       rng.uniform(0, 0.5, n),
        "high_discount_flag":  (rng.uniform(0, 0.5, n) > 0.3).astype(int),
        "order_value":         rng.uniform(10, 5000, n),
        "benefit_per_order":   rng.uniform(-200, 500, n),
        "shipping_mode_enc":   rng.choice([0, 1, 2, 3], n).astype(float),
        "market_geo_risk":     rng.uniform(0.1, 0.7, n),
        "category_risk":       rng.uniform(0.1, 0.6, n),
        "product_price_log":   rng.uniform(0, 8, n),
        "order_quantity":      rng.integers(1, 10, n).astype(float),
    })
    # Simulate Late_delivery_risk: longer scheduled days + standard shipping = risky
    y = (
        (X["scheduled_ship_days"] >= 4) &
        (X["shipping_mode_enc"] >= 2)
    ).astype(int)
    return X, y, "synthetic_fallback"


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    if CSV_PATH.exists():
        X, y, data_source = load_real_data()
    else:
        X, y, data_source = generate_synthetic_data()

    print(f"\n🔧 Training GradientBoostingClassifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_split=20,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    # Cross-val
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)

    metrics = {
        "data_source": data_source,
        "n_samples_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(float(acc), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "n_features": len(X.columns),
        "feature_importances": dict(
            zip(X.columns.tolist(), model.feature_importances_.round(4).tolist())
        ),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
    }

    print(f"\n📊 Results ({data_source}):")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(f"   CV AUC   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    with open(FEATURES_PATH, "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)
    print(f"💾 Features saved → {FEATURES_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metrics saved → {METRICS_PATH}")

    if data_source == "synthetic_fallback":
        print("\n⚠️  WARNING: data_source=synthetic_fallback. Place DataCoSupplyChainDataset.csv")
        print("   in classifier/ and re-run for production-quality metrics.")

    return metrics


if __name__ == "__main__":
    main()
