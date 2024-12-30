import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json

def train_test(df):
    X = df.drop(['loan_status'], axis=1)
    y = df['loan_status']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Обрабатываем отдельно количественные и категориальные столбцы
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Делим на тестовую и обучающую выбоорки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=101)

    # Применяем предобработку уже после разделения
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test

def plot_roc_auc(y_true, y_scores, model_name):
    # Получаем индексы после сортировки scores по убыванию
    sorted_indices = np.argsort(y_scores)[::-1]
    # Применяем индексы в векторам
    y_true = np.array(y_true)[sorted_indices]
    y_scores = np.array(y_scores)[sorted_indices]

    FPR = []
    TPR = []

    P = sum(y_true)
    N = len(y_true) - P

    # Идеально строит кривую как в sklearn
    #thresholds = np.unique(y_scores)
    thresholds = np.arange(0.0, 1.01, 0.2)

    for thresh in thresholds:
        FP = 0
        TP = 0
        thresh = round(thresh, 2)
        for i in range(len(y_scores)):
            if y_scores[i] >= thresh:
                if y_true[i] == 1:
                    TP += 1
                if y_true[i] == 0:
                    FP += 1

        TPR.append(TP / P if P != 0 else 0)
        FPR.append(FP / N if N != 0 else 0)

    manual_auc = -1 * np.trapz(TPR, FPR)

    # Встроенная функция для площади auc и для графика ROC-кривой
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    sklearn_auc = auc(fpr, tpr)

    # Построение графиков
    plt.figure(figsize=(8, 6))
    plt.plot(FPR, TPR, label=f"Manual ROC Curve (AUC = {manual_auc:.2f})", linestyle="--", color="blue")
    plt.plot(fpr, tpr, label=f"Sklearn ROC Curve (AUC = {sklearn_auc:.2f})", linestyle="-", color="red")
    plt.plot([0, 1], [0, 1], linestyle="--", color="green", label="Random Guess")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve Comparison for {model_name}")
    plt.legend()
    plt.grid(True)
    filepath = f"results/ROC_Curve_Comparison_{model_name}.png"
    plt.savefig(filepath)
    plt.show()

    return manual_auc, sklearn_auc

def plot_pr_auc(y_true, y_scores, model_name):
    # Получаем индексы после сортировки scores по убыванию
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true = np.array(y_true)[sorted_indices]
    y_scores = np.array(y_scores)[sorted_indices]

    Precision = []
    Recall = []

    P = sum(y_true)  # Количество положительных примеров

    # Идеальный случай
    # thresholds = np.unique(y_scores)
    # Таким способом получается большая погрешность
    thresholds = np.arange(0.0, 1.01, 0.2)

    for thresh in thresholds:
        FP = 0
        TP = 0
        thresh = round(thresh, 2)
        for i in range(len(y_scores)):
            if y_scores[i] >= thresh:
                if y_true[i] == 1:
                    TP += 1
                if y_true[i] == 0:
                    FP += 1

        Recall.append(TP / P if P != 0 else 0)
        Precision.append(TP / (TP + FP) if (TP + FP) != 0 else 1) # Обрабатывая деление на 0 по дефолту пишем 1

    manual_auc = -1 * np.trapz(Precision, Recall)

    # Встроенная функция для площади auc и для графика PR-кривой
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    sklearn_auc = auc(recall, precision)

    # Построение графиков
    plt.figure(figsize=(8, 6))
    plt.plot(Recall, Precision, label=f"Manual PR Curve (AUC = {manual_auc:.2f})", linestyle="--", color="blue")
    plt.plot(recall, precision, label=f"Sklearn PR Curve (AUC = {sklearn_auc:.2f})", linestyle="-", color="red")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve Comparison for {model_name}")
    plt.legend()
    plt.grid(True)
    filepath = f"results/PR_Curve_Comparison_Fixed_{model_name}.png"
    plt.savefig(filepath)
    plt.show()

    return manual_auc, sklearn_auc

def metrics(y_pred, y_pred_proba, y_test, model_name):
    results = {}
    TP = ((y_pred == 1) & (y_test == 1)).sum()
    TN = ((y_pred == 0) & (y_test == 0)).sum()
    FP = ((y_pred == 1) & (y_test == 0)).sum()
    FN = ((y_pred == 0) & (y_test == 1)).sum()

    manual_accuracy = (TP + TN) / (TP + TN + FP + FN)
    manual_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    manual_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    manual_F1 = (2 * manual_precision * manual_recall) / (manual_precision + manual_recall) if manual_precision + manual_recall > 0 else 0
    manual_roc_auc, sklearn_roc_auc = plot_roc_auc(y_test, y_pred_proba, model_name)
    manual_pr_auc, sklearn_pr_auc = plot_pr_auc(y_test, y_pred_proba, model_name)

    results['manual'] = {
        f"Accuracy": f"{manual_accuracy:.3f}",
        f"Precision": f"{manual_precision:.3f}",
        f"Recall": f"{manual_recall:.3f}",
        f"F1": f"{manual_F1:.3f}",
        f"ROC-AUC: ": f"{manual_roc_auc:.3f}",
        f"PR-AUC: ": f"{manual_pr_auc:.3f}",
    }

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)

    results['sklearn'] = {
        f"Accuracy": f"{accuracy:.3f}",
        f"Precision": f"{precision:.3f}",
        f"Recall": f"{recall:.3f}",
        f"F1": f"{F1:.3f}",
        f"ROC-AUC: ": f"{sklearn_roc_auc:.3f}",
        f"PR-AUC: ": f"{sklearn_pr_auc:.3f}",
    }

    return results

def Random_Forest(X_train, X_test, y_train):
    RFC = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5, # Минимальное количество объектов для разделения узла
        min_samples_leaf=2, # Минимальное количество выборок в листовом узле
        max_features='sqrt',
        class_weight='balanced',
        random_state=101
    )
    RFC.fit(X_train, y_train)
    y_pred = RFC.predict(X_test)
    y_pred_proba = RFC.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_proba

if __name__ == "__main__":
    filename = 'update_df.csv'
    df = pd.read_csv(filename)
    results = {}

    X_train, X_test, y_train, y_test = train_test(df)

    model_name = 'RandomForest'
    y_pred, y_pred_proba = Random_Forest(X_train, X_test, y_train)
    results[model_name] = metrics(y_pred, y_pred_proba, y_test, model_name)

    filename = 'results/results.json'
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)