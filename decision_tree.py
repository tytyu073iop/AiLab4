import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional, Union, List, Tuple


class DecisionTreeClassifier:
    """
    Decision Tree Classifier for categorical features.
    
    Parameters
    ----------
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.
    max_depth : int, default=None
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    missing_value_strategy : {'separate_category', 'most_frequent'}, default='separate_category'
        How to handle missing values.
        - 'separate_category': treat missing as a distinct category.
        - 'most_frequent': assign missing to the most frequent category in the feature.
    random_state : int, default=None
        Seed for random number generator (for tie-breaking).
    
    Attributes
    ----------
    tree_ : dict
        The tree structure.
    n_features_ : int
        Number of features seen during fit.
    classes_ : ndarray
        Unique class labels.
    """
    
    def __init__(
        self,
        criterion: str = 'gini',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        missing_value_strategy: str = 'separate_category',
        random_state: Optional[int] = None
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.missing_value_strategy = missing_value_strategy
        self.random_state = random_state
        
        self.tree_ = None
        self.n_features_ = None
        self.classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'DecisionTreeClassifier':
        """
        Build a decision tree classifier from the training set (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.
        """
        # Convert to pandas DataFrame/Series for easier handling of missing values and categorical features
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Preprocess missing values
        X_processed = self._preprocess_missing(X)
        
        # Build tree recursively
        self.tree_ = self._build_tree(X_processed, y, depth=0)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if self.tree_ is None:
            raise ValueError("Tree not fitted. Call fit first.")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_processed = self._preprocess_missing(X)
        
        predictions = []
        for i in range(len(X_processed)):
            sample = X_processed.iloc[i]
            pred = self._traverse_tree(sample, self.tree_)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _preprocess_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to strategy.
        Returns a copy of X with missing values handled.
        """
        X_copy = X.copy()
        
        if self.missing_value_strategy == 'separate_category':
            # Replace NaN with a special marker
            for col in X_copy.columns:
                if X_copy[col].isnull().any():
                    X_copy[col] = X_copy[col].fillna('__MISSING__')
        elif self.missing_value_strategy == 'most_frequent':
            for col in X_copy.columns:
                if X_copy[col].isnull().any():
                    most_freq = X_copy[col].mode()[0]
                    X_copy[col] = X_copy[col].fillna(most_freq)
        else:
            raise ValueError(f"Unknown missing value strategy: {self.missing_value_strategy}")
        
        return X_copy
    
    def _build_tree(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        depth: int
    ) -> dict:
        """
        Recursively build a decision tree.
        
        Returns a node dictionary representing the subtree.
        """
        # Stopping criteria
        if self._should_stop(X, y, depth):
            leaf_value = self._compute_leaf_value(y)
            return {'type': 'leaf', 'value': leaf_value}
        
        # Find best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            leaf_value = self._compute_leaf_value(y)
            return {'type': 'leaf', 'value': leaf_value}
        
        feature_idx, feature_name, categories_left, categories_right = best_split
        
        # Split data
        mask = X.iloc[:, feature_idx].isin(categories_left)
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]
        
        # Ensure min_samples_leaf constraint
        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            leaf_value = self._compute_leaf_value(y)
            return {'type': 'leaf', 'value': leaf_value}
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'type': 'split',
            'feature_idx': feature_idx,
            'feature_name': feature_name,
            'categories_left': categories_left,
            'categories_right': categories_right,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _should_stop(self, X: pd.DataFrame, y: pd.Series, depth: int) -> bool:
        """Check if we should stop splitting."""
        # Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        
        # Min samples split not met
        if len(X) < self.min_samples_split:
            return True
        
        # Pure node (all same class)
        if len(np.unique(y)) == 1:
            return True
        
        # No features left to split (should not happen)
        if X.shape[1] == 0:
            return True
        
        return False
    
    def _compute_leaf_value(self, y: pd.Series) -> int:
        """Compute the class label for a leaf node (majority class)."""
        counts = Counter(y)
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def _find_best_split(self, X: pd.DataFrame, y: pd.Series) -> Optional[Tuple]:
        """
        Find the best split among all features and categories.
        Returns (feature_idx, feature_name, categories_left, categories_right) or None.
        """
        best_gain = -float('inf')
        best_split = None
        
        for feature_idx, feature_name in enumerate(X.columns):
            # Get unique categories (excluding missing marker if present)
            categories = X.iloc[:, feature_idx].unique()
            if len(categories) <= 1:
                continue  # Cannot split on a feature with only one category
            
            # For categorical features, we consider binary splits by partitioning categories
            # We'll use a greedy approach: sort categories by impurity and try splits
            # For simplicity, we'll evaluate all possible binary partitions (2^(k-1)-1) is expensive,
            # so we'll use a heuristic: sort by proportion of majority class and try splits.
            # Alternatively, we can treat each category as a separate branch (multi-way split),
            # but we'll implement binary splits for simplicity.
            
            # Compute impurity for each category
            category_impurities = []
            for cat in categories:
                mask = X.iloc[:, feature_idx] == cat
                if mask.sum() == 0:
                    continue
                y_subset = y[mask]
                impurity = self._compute_impurity(y_subset)
                category_impurities.append((cat, impurity, mask.sum()))
            
            if not category_impurities:
                continue
            
            # Sort categories by impurity (or by proportion of majority class)
            # We'll try splits that separate categories into two groups
            # For simplicity, we'll try each category as left group and rest as right
            for i, (cat_left, _, _) in enumerate(category_impurities):
                categories_left = [cat_left]
                categories_right = [c for c, _, _ in category_impurities if c != cat_left]
                
                # Compute gain
                gain = self._compute_split_gain(X, y, feature_idx, categories_left, categories_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, feature_name, categories_left, categories_right)
        
        return best_split
    
    def _compute_impurity(self, y: pd.Series) -> float:
        """Compute impurity (Gini or entropy) of a node."""
        n = len(y)
        if n == 0:
            return 0
        
        counts = np.bincount([np.where(self.classes_ == val)[0][0] for val in y])
        proportions = counts / n
        
        if self.criterion == 'gini':
            return 1 - np.sum(proportions ** 2)
        elif self.criterion == 'entropy':
            # Avoid log(0)
            return -np.sum(proportions * np.log2(proportions + 1e-10))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _compute_split_gain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_idx: int,
        categories_left: List,
        categories_right: List
    ) -> float:
        """Compute information gain of a split."""
        parent_impurity = self._compute_impurity(y)
        
        mask = X.iloc[:, feature_idx].isin(categories_left)
        y_left = y[mask]
        y_right = y[~mask]
        
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        
        if n_left == 0 or n_right == 0:
            return 0
        
        impurity_left = self._compute_impurity(y_left)
        impurity_right = self._compute_impurity(y_right)
        
        weighted_impurity = (n_left / n_total) * impurity_left + (n_right / n_total) * impurity_right
        gain = parent_impurity - weighted_impurity
        return gain
    
    def _traverse_tree(self, sample: pd.Series, node: dict) -> int:
        """Traverse the tree for a single sample."""
        if node['type'] == 'leaf':
            return node['value']
        
        # Split node
        feature_val = sample.iloc[node['feature_idx']]
        if feature_val in node['categories_left']:
            return self._traverse_tree(sample, node['left'])
        else:
            return self._traverse_tree(sample, node['right'])
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """Return the mean accuracy on the given test data and labels."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)