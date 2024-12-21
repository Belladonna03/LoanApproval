import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

def metrcics(y_pred, y_test, model):
    TP = ((y_pred == 1) & (y_test == 1)).sum()
    TN = ((y_pred == 0) & (y_test == 0)).sum()
    FP = ((y_pred == 0) & (y_test == 1)).sum()
    FN = ((y_pred == 1) & (y_test == 0)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Accuracy for {model}: {accuracy:.3f}")

    precision = TP / (TP + FP)
    print(f"Precision for {model}: {precision:.3f}")

    recall = TP / (TP + FN)
    print(f"Recall for {model}: {recall:.3f}")

    F1 = 2 * precision * recall / (precision + recall)
    print(f"F1 for {model}: {F1:.3f}")

    results = {
        f"Accuracy for {model}": accuracy,
        f"Precision for {model}": precision,
        f"Recall for {model}": recall,
        f"F1 for {model}": F1
    }

    return results


filename = 'update_df.csv'
df = pd.read_csv(filename)

test_df = df.drop(['loan_status'], axis=1)

X = df.drop(['loan_status'], axis=1)
y = df.loc[:, 'loan_status']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()  # Стандартизация числовых признаков
categorical_transformer = OneHotEncoder()  # Кодирование категориальных признаков

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=101)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

"""Logistic Regression"""
LR = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    max_iter=1000,
    multi_class='auto',
    class_weight='balanced',
    random_state=101
)

LR.fit(X_train_processed, y_train)

y_pred = LR.predict(X_test_processed)
y_pred_proba = LR.predict_proba(X_test_processed)[:, 1] # Вероятность для 1 класса

results = [metrcics(y_pred, y_test, 'Logisctic Regression')]

"""SVM"""



