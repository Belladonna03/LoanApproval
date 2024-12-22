import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import json

def ROC_AUC(y_test, y_pred, model):
    y_test = y_test.tolist()
    y_pred = y_pred.tolist()

    count_ones = y_test.count(1)
    count_zeros = y_test.count(0)

    m = 1 / count_ones
    n = 1 / count_zeros

    zipped = zip(y_test, y_pred)
    zipped_sorted = sorted(zipped, key=lambda x: x[1], reverse=True)

    x = []
    y = []
    x_cnt = 0
    y_cnt = 0
    x.append(x_cnt)
    y.append(y_cnt)
    for test, pred in zipped_sorted:
        x_cnt += (n if test == 0 else 0)
        y_cnt += (m if test == 1 else 0)
        x.append(x_cnt)
        y.append(y_cnt)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x, y, marker='o', linestyle='-', color='b', label='Точки')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='r', label='Прямая (0, 0) -> (1, 1)')
    ax1.set_title('ROC-кривая нарисованная поточечно')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()

    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_title('ROC-кривая встроенная')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc="lower right")

    fig.text(0.5, 0.02, f"Сравнение ROC-кривых для {model}", ha='center', fontsize=12)
    plt.savefig(f'roc_curves_for_{model}.png', dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    return roc_auc

def metrcics(y_pred, y_test, model):
    TP = ((y_pred == 1) & (y_test == 1)).sum()
    TN = ((y_pred == 0) & (y_test == 0)).sum()
    FP = ((y_pred == 0) & (y_test == 1)).sum()
    FN = ((y_pred == 1) & (y_test == 0)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    F1 = (2 * precision * recall) / (precision + recall)

    auc = ROC_AUC(y_test, y_pred, model)

    results = {
        f"Accuracy for {model}": accuracy,
        f"Precision for {model}": precision,
        f"Recall for {model}": recall,
        f"F1 for {model}": F1,
        f"ROC-AUC: ": auc
    }

    return results


filename = 'update_df.csv'
df = pd.read_csv(filename)
results = {}

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
    tol=10**-4,
    multi_class='auto',
    class_weight='balanced',
    random_state=101
)

LR.fit(X_train_processed, y_train)

y_pred = LR.predict(X_test_processed)
y_pred_proba = LR.predict_proba(X_test_processed)[:, 1] # Вероятность для 1 класса

results['Logistic Regression'] = metrcics(y_pred, y_test, 'Logistic Regression')

"""SVM"""
SVC = SVC(
    C=1.0,
    kernel='rbf',
    tol=10**-4,
    gamma='scale',
    probability=True,
    random_state=42
)

SVC.fit(X_train_processed, y_train)
y_pred = SVC.predict(X_test_processed)
y_pred_proba = SVC.predict_proba(X_test_processed)[:, 1]

results['SVC'] = metrcics(y_pred, y_test, 'SVC')

"""Random Forest"""
RFC = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=101
)

RFC.fit(X_train_processed, y_train)
y_pred = RFC.predict(X_test_processed)
y_pred_proba = RFC.predict_proba(X_test_processed)[:, 1]

results['RandomForest'] = metrcics(y_pred, y_test, 'RandomForest')

filename = 'results.csv'
with open(filename, 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)