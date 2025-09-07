#!/usr/bin/env python3
"""scripts/train.py

Train an ANN on the Pima Indians Diabetes dataset.
Features:
- zero-value handling for specific columns
- median imputation
- StandardScaler
- callbacks: EarlyStopping, ModelCheckpoint
- evaluation: classification report, confusion matrix, ROC-AUC
- CLI: --epochs, --batch-size, --smoke-test, --data-path, --model-out
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DEFAULT_URL = "https://raw.githubusercontent.com/slmsshk/pima-indians-diabetes.data.csv/main/pima-indians-diabetes.csv"

def load_data(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(DEFAULT_URL, header=None)
        # if dataset has no header, add names commonly used for Pima dataset
        df.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    return df

def preprocess(df):
    df = df.copy()
    # columns where 0 is not a valid value and likely represents missingness
    cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    # replace zeros with NaN then impute with median
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test

def build_model(input_dim, dropout_rate=0.2):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test, model_out=None):
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    print('\nClassification report:')
    print(classification_report(y_test, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f'ROC AUC: {auc:.4f}')

    # plot ROC curve to file
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')
    print('Saved ROC curve to roc_curve.png')

    if model_out:
        try:
            model.save(model_out)
            print(f'Model saved to {model_out}')
        except Exception as e:
            print('Error saving model:', e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--smoke-test', action='store_true', help='Run quick smoke test (very small subset, 1 epoch)')
    parser.add_argument('--data-path', type=str, default=None, help='Optional local CSV path for dataset')
    parser.add_argument('--model-out', type=str, default='best_model.h5', help='Path to save best model')
    args = parser.parse_args()

    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = preprocess(df)

    # For smoke test keep tiny subset and run 1 epoch by default
    if args.smoke_test:
        X_train = X_train.sample(n=min(128, len(X_train)), random_state=RANDOM_SEED)
        y_train = y_train.loc[X_train.index]
        args.epochs = 1
        print('Running smoke test: small subset, 1 epoch')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(X_train.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(args.model_out, monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2
    )

    # Save training history to CSV
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('training_history.csv', index=False)
    print('Saved training history to training_history.csv')

    evaluate_model(model, X_test, y_test, model_out=args.model_out)

if __name__ == '__main__':
    main()
