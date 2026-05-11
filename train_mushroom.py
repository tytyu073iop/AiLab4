import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from decision_tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

# Load dataset
print("Loading mushroom dataset...")
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets.iloc[:, 0]  # 'poisonous' column

print(f"Dataset shape: {X.shape}")
print(f"Missing values in each column:")
print(X.isnull().sum())

# Split into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# Train decision tree with default parameters
print("\nTraining decision tree...")
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    missing_value_strategy='separate_category',
    random_state=42
)
clf.fit(X_train, y_train)
print("Training completed.")

# Predict on train and test
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Compute metrics
def evaluate_metrics(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label='e', zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label='e', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='e', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=['e', 'p'])
    print(f"\n{label} Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    return acc, prec, rec, f1

train_acc, train_prec, train_rec, train_f1 = evaluate_metrics(y_train, y_train_pred, "Train")
test_acc, test_prec, test_rec, test_f1 = evaluate_metrics(y_test, y_test_pred, "Test")

# Feature importance (simple count of splits)
def get_feature_importance(tree_node, feature_names, importance_dict):
    if tree_node['type'] == 'split':
        feat = tree_node['feature_name']
        importance_dict[feat] = importance_dict.get(feat, 0) + 1
        get_feature_importance(tree_node['left'], feature_names, importance_dict)
        get_feature_importance(tree_node['right'], feature_names, importance_dict)

importance = {}
if clf.tree_ is not None:
    get_feature_importance(clf.tree_, X.columns, importance)
    print("\nFeature importance (split count):")
    for feat, count in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {count}")

# Print tree depth
def max_depth(node):
    if node['type'] == 'leaf':
        return 0
    return 1 + max(max_depth(node['left']), max_depth(node['right']))

if clf.tree_ is not None:
    depth = max_depth(clf.tree_)
    print(f"\nTree depth: {depth}")

# Compare with sklearn's DecisionTreeClassifier (for reference)
try:
    from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Encode categorical features (sklearn requires numeric)
    X_train_encoded = X_train.apply(LabelEncoder().fit_transform)
    X_test_encoded = X_test.apply(LabelEncoder().fit_transform)
    y_train_encoded = LabelEncoder().fit_transform(y_train)
    y_test_encoded = LabelEncoder().fit_transform(y_test)
    
    sk_clf = SklearnDecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    sk_clf.fit(X_train_encoded, y_train_encoded)
    sk_test_pred = sk_clf.predict(X_test_encoded)
    sk_acc = accuracy_score(y_test_encoded, sk_test_pred)
    print(f"\nSklearn Decision Tree Accuracy (for reference): {sk_acc:.4f}")
except ImportError:
    print("\nSklearn not available for comparison.")