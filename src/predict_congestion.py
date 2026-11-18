"""Train and evaluate a congestion prediction model on synthetic data.

Run from the project root:  python src/predict_congestion.py
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "port_data.csv"
FEATURES = ["hour", "day_of_week", "scheduled_trucks", "vessels_in_port", "customs_delay_min"]
TARGET = "congested"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python data/generate_synthetic_data.py"
        )
    return pd.read_csv(path)


def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test))
    return model, report


if __name__ == "__main__":
    data = load_data()
    model, report = train_model(data)
    print("Evaluation on held-out synthetic data (not real port data):")
    print(report)
    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("Feature importance:")
    for name, score in importances:
        print(f"  {name}: {score:.2f}")
