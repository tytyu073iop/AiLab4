import pandas as pd
import numpy as np
from decision_tree import DecisionTreeClassifier

# Create a simple categorical dataset
data = {
    'color': ['red', 'red', 'green', 'blue', 'blue', 'green', 'red'],
    'size': ['small', 'large', 'large', 'medium', 'medium', 'small', 'large'],
    'label': ['yes', 'yes', 'no', 'no', 'yes', 'no', 'yes']
}
df = pd.DataFrame(data)
X = df[['color', 'size']]
y = df['label']

print("Dataset:")
print(df)
print()

# Train decision tree
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X, y)

print("Tree structure:")
print(clf.tree_)
print()

# Predict
y_pred = clf.predict(X)
print("Predictions:", y_pred)
print("Accuracy:", np.mean(y_pred == y))

# Test with missing values
df_missing = df.copy()
df_missing.loc[0, 'color'] = np.nan
X_missing = df_missing[['color', 'size']]
print("\nDataset with missing values:")
print(X_missing)

clf2 = DecisionTreeClassifier(missing_value_strategy='separate_category', random_state=42)
clf2.fit(X_missing, y)
y_pred2 = clf2.predict(X_missing)
print("Predictions with missing:", y_pred2)
print("Accuracy:", np.mean(y_pred2 == y))