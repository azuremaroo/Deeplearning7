import os
import argparse
import random
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_diabetes_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        # Try common fallback locations in this workspace
        candidates = [
            os.path.join("20250808", "diabetes.csv"),
            os.path.join("tensorflow를_활용한_인공신경망_구현", "data", "diabetes.csv"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                csv_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Could not find diabetes.csv. Tried: {csv_path} and fallbacks."
            )

    df = pd.read_csv(csv_path)
    required_columns = {
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")
    return df


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    # Treat physiologically impossible zeros as missing and impute with median
    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    df = df.copy()
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
    df[zero_as_missing] = df[zero_as_missing].fillna(df[zero_as_missing].median())

    X = df.drop(columns=["Outcome"]).values.astype(np.float32)
    y = df["Outcome"].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a Keras model on diabetes.csv")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join("20250808", "diabetes.csv"),
        help="Path to diabetes.csv (will try fallbacks if missing)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--model-out",
        type=str,
        default=os.path.join("20250812", "diabetes_best_model.h5"),
        help="Where to save the best model",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    df = load_diabetes_csv(args.csv)
    X_all, y_all, _ = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=args.seed
    )

    model = build_model(input_dim=X_train.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=args.model_out,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nBest model saved to:", os.path.abspath(args.model_out))

    # Evaluation
    train_loss, train_acc, train_auc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(
        f"Train -> loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc:.4f}"
    )
    print(
        f"Test  -> loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc:.4f}"
    )

    # Save final model (same weights as best due to restore_best_weights)
    final_path = os.path.join(os.path.dirname(args.model_out), "diabetes_final_model.h5")
    model.save(final_path)
    print("Final model saved to:", os.path.abspath(final_path))


if __name__ == "__main__":
    main()


