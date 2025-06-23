# Import necessary libraries

# Basic libraries for numerical operations, data manipulation, and visualization
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation and analysis
import seaborn as sns  # For statistical data visualization
import matplotlib.pyplot as plt  # For creating plots and figures
import warnings  # To handle warnings in the code
warnings.filterwarnings('ignore')  # Ignore warnings to reduce clutter

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Input, LSTM, Concatenate, Add, Attention

# Data Preprocessing and Feature Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # For feature scaling (Standard and Min-Max scaling)

# Data Splitting and Model Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,cross_validate  # For splitting data, cross-validation, and hyperparameter tuning
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix  # For model evaluation and performance metrics

# Handling Imbalanced Datasets
from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets

# Machine Learning Models (for classification)
from sklearn.linear_model import LogisticRegression  # Logistic Regression for binary classification
from sklearn.neural_network import MLPClassifier  # Multi-layer Perceptron (Neural Network) for classification
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.ensemble import (
    RandomForestClassifier,  # Random Forest classifier
    GradientBoostingClassifier,  # Gradient Boosting classifier
    AdaBoostClassifier,  # AdaBoost classifier
    ExtraTreesClassifier,
    StackingClassifier,  # Stacking classifier
    VotingClassifier  # Voting classifier
)
from xgboost import XGBClassifier  # XGBoost classifier


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2


# Model Interpretability (SHAP)
import shap
# Step 1: Data Preprocessing
X = df.drop(columns=['HadHeartAttack'])  # Features
y = df['HadHeartAttack']  # Target variable

# --- 1. Split data into train (70%), validation (15%), and test (15%) ---
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val)
# (0.1765 * 0.85 ≈ 0.15)

# --- 2. Normalize and reshape ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_3d = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# --- 3. Fix dtype errors ---
X_train_scaled = X_train_scaled.astype(np.float32)
X_val_scaled = X_val_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)
X_train_3d = X_train_3d.astype(np.float32)
X_val_3d = X_val_3d.astype(np.float32)
X_test_3d = X_test_3d.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)



# --- 3. Define models ---
def build_fnn(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 4. Build model dictionary ---
models = {
    'FNN': build_fnn(X_train_scaled.shape[1]),
    'CNN': build_cnn((X_train_scaled.shape[1], 1)),
    'LSTM': build_lstm((X_train_scaled.shape[1], 1)),
    'CNN_LSTM': build_cnn_lstm((X_train_scaled.shape[1], 1)),
    'Wide_Deep': build_wide_deep(X_train_scaled.shape[1]),
    'Residual': build_residual(X_train_scaled.shape[1]),
    'Attention': build_attention((X_train_scaled.shape[1], 1))
}

# --- 5. Train and evaluate ---
results = {}
roc_data = {}
histories = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in ['FNN', 'Wide_Deep', 'Residual']:
        history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0,
                            validation_data=(X_val_scaled, y_val))
        y_pred_prob = model.predict(X_test_scaled).flatten()
    else:
        history = model.fit(X_train_3d, y_train, epochs=10, batch_size=32, verbose=0,
                            validation_data=(X_val_3d, y_val))
        y_pred_prob = model.predict(X_test_3d).flatten()

    # Store history
    histories[name] = history

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in ['FNN', 'Wide_Deep', 'Residual']:
        model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
        y_pred_prob = model.predict(X_test_scaled).flatten()
    else:
        model.fit(X_train_3d, y_train, epochs=10, batch_size=32, verbose=0)
        y_pred_prob = model.predict(X_test_3d).flatten()

    y_pred = (y_pred_prob > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_prob),
        'classification_report': classification_report(y_test, y_pred)
    }
    roc_data[name] = (fpr, tpr)

# --- 6. Plot ROC curves ---
plt.figure(figsize=(10, 8))
for name, (fpr, tpr) in roc_data.items():
    auc = results[name]['roc_auc']
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of All Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_all_models.png")
plt.show()

# --- 7. Create and save performance table ---
performance_df = pd.DataFrame({
    model_name: {
        "Accuracy": round(metrics['accuracy'], 2),
        "Precision": round(metrics['precision'], 2),
        "Recall": round(metrics['recall'], 2),
        "F1 Score": round(metrics['f1_score'], 2),
        "ROC AUC": round(metrics['roc_auc'], 2)
    }
    for model_name, metrics in results.items()
}).T

performance_df.index.name = 'Model'
performance_df.to_csv("performance_table.csv")
print("\nPerformance table saved as 'performance_table.csv'")
print(performance_df)
