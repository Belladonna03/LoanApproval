import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import json

# Function to plot ROC curve with points and calculate AUC
def plot_roc_auc(y_test, y_pred, model):
    y_test = y_test.tolist()
    y_pred = y_pred.tolist()

    # Count of 1's and 0's in y_test
    count_ones = y_test.count(1)
    count_zeros = y_test.count(0)

    # Normalization factors for each class
    m = 1 / count_ones
    n = 1 / count_zeros

    # Sorting by predicted values
    zipped = zip(y_test, y_pred)
    zipped_sorted = sorted(zipped, key=lambda x: x[1], reverse=True)

    # Initialize lists for the ROC curve points
    x = [0]  # First point is always (0, 0)
    y = [0]  # First point is always (0, 0)

    x_cnt = 0
    y_cnt = 0

    # Generate the ROC curve
    for test, pred in zipped_sorted:
        x_cnt += (n if test == 0 else 0)
        y_cnt += (m if test == 1 else 0)
        x.append(x_cnt)
        y.append(y_cnt)

    # Add the final point (1, 1) to ensure the last point is always (1, 1)
    x.append(1)
    y.append(1)

    # Built-in ROC curve calculation
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Custom ROC curve with points
    ax1.plot(x, y, marker='o', linestyle='-', color='b', label='ROC Points')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess Line')
    ax1.set_title('Pointwise ROC Curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()

    # Built-in ROC curve
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_title('Built-in ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc="lower right")

    fig.text(0.5, 0.02, f"ROC Comparison for {model}", ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'roc_curves_for_{model}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return roc_auc

# def plot_pr_auc():
# Function to plot PR curve with points and calculate AUC
def plot_pr_auc(y_test, y_pred_proba, model):
    y_test = y_test.tolist()
    y_pred_proba = y_pred_proba.tolist()

    zipped_sorted = sorted(zip(y_test, y_pred_proba), key=lambda x: x[1], reverse=True)

    precision_list = []
    recall_list = []

    for threshold in [i / 10.0 for i in range(10, -1, -1)]:
        tp = fp = 0
        fn = y_test.count(1)

        for y_true, y_prob in zipped_sorted:
            if y_prob >= threshold:
                if y_true == 1:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
            else:
                break

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        precision_list.append(precision)
        recall_list.append(recall)

    pr_auc_manual = auc(recall_list, precision_list)

    precision_sklearn, recall_sklearn, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc_sklearn = auc(recall_sklearn, precision_sklearn)

    plt.figure(figsize=(10, 6))
    plt.plot(recall_list, precision_list, marker='o', linestyle='-', color='b',
             label=f'Manual Calculation (PR-AUC = {pr_auc_manual:.2f})')
    plt.plot(recall_sklearn, precision_sklearn, linestyle='--', color='r',
             label=f'Scikit-learn (PR-AUC = {pr_auc_sklearn:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model}')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(f'pr_curves_for_{model}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return pr_auc_sklearn


# Function to calculate classification metrics
def metrics(y_pred, y_pred_proba, y_test, model):
    TP = ((y_pred == 1) & (y_test == 1)).sum()
    TN = ((y_pred == 0) & (y_test == 0)).sum()
    FP = ((y_pred == 0) & (y_test == 1)).sum()
    FN = ((y_pred == 1) & (y_test == 0)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    F1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    roc_auc = plot_roc_auc(y_test, y_pred, model)
    pr_auc = plot_pr_auc(y_test, y_pred, model)

    results = {
        f"Accuracy": f"{accuracy:.3f}",
        f"Precision": f"{precision:.3f}",
        f"Recall": f"{recall:.3f}",
        f"F1": f"{F1:.3f}",
        f"ROC-AUC: ": f"{roc_auc:.3f}",
        f"PR-AUC: ": f"{pr_auc:.3f}",
    }

    return results

# Loading dataset
filename = 'update_df.csv'
df = pd.read_csv(filename)
results = {}

X = df.drop(['loan_status'], axis=1)
y = df['loan_status']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing pipeline
numeric_transformer = StandardScaler()  # Standardizing numeric features
categorical_transformer = OneHotEncoder()  # Encoding categorical features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=101)

# Applying preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Logistic Regression Model
LR = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='liblinear',
    tol=1e-4,
    multi_class='auto',
    class_weight='balanced',
    random_state=101
)
LR.fit(X_train_processed, y_train)
y_pred = LR.predict(X_test_processed)
y_pred_proba = LR.predict_proba(X_test_processed)[:, 1]
metrics(y_pred, y_pred_proba, y_test, 'Logistic Regression')
results['Logistic Regression'] = metrics(y_pred, y_pred_proba, y_test, 'Logistic Regression')

# Support Vector Classifier Model
svc = SVC(
    C=1.0,
    kernel='rbf',
    tol=1e-4,
    gamma='scale',
    probability=True,
    random_state=42
)
svc.fit(X_train_processed, y_train)
y_pred = svc.predict(X_test_processed)
y_pred_proba = svc.predict_proba(X_test_processed)[:, 1]
results['SVC'] = metrics(y_pred, y_pred_proba, y_test, 'SVC')

# Random Forest Classifier Model
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
results['RandomForest'] = metrics(y_pred, y_pred_proba, y_test, 'RandomForest')

# Saving results
filename = 'results.json'
with open(filename, 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)
