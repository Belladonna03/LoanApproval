import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def metrcics(y_pred, y_test, model):
    TP = ((y_pred == 1) & (y_test == 1)).sum()
    TN = ((y_pred == 0) & (y_test == 0)).sum()
    FP = ((y_pred == 0) & (y_test == 1)).sum()
    FN = ((y_pred == 1) & (y_test == 0)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    F1 = (2 * precision * recall) / (precision + recall)

    results = {
        f"Accuracy for {model}": accuracy,
        f"Precision for {model}": precision,
        f"Recall for {model}": recall,
        f"F1 for {model}": F1
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

results['Logistic Regression'] = metrcics(y_pred, y_test, 'Logisctic Regression')

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

for model, metrics_dict in results.items():
    print(f"\nResults for {model}:")
    for metric, value in metrics_dict.items():
        print(f"{metric}: {value:.3f}")