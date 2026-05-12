import numpy as np
import pandas as pd
from typing import Optional, Union
import matplotlib.pyplot as plt
from task1 import DecisionTreeClassifier


class AdaBoostClassifier:
    
    def __init__(
        self,
        n_estimators: int = 50,
        weak_learner_depth: int = 3,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.weak_learner_depth = weak_learner_depth
        self.random_state = random_state
        
        self.estimators_ = []          # list of weak learners
        self.estimator_weights_ = []   # alpha for each learner
        self.classes_ = None           # unique class labels
        self.n_classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'AdaBoostClassifier':
        # Convert to pandas
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Check binary classification
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Map classes to +1 and -1 (inverted sign to match weak learner predictions)
        self.class_map_ = {self.classes_[0]: -1, self.classes_[1]: 1}
        y_mapped = y.map(self.class_map_).values.astype(float)
        
        n_samples = X.shape[0]
        
        # Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples
        
        self.estimators_ = []
        self.estimator_weights_ = []
        
        for t in range(self.n_estimators):
            # Train weak learner using weighted bootstrap sampling
            indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=sample_weights
            )
            X_boot = X.iloc[indices].reset_index(drop=True)
            y_boot = y.iloc[indices].reset_index(drop=True)
            
            # Weak learner: decision tree with configurable depth
            weak_learner = DecisionTreeClassifier(
                max_depth=self.weak_learner_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state
            )
            weak_learner.learn(X_boot, y_boot)
            
            # Predict on original data
            y_pred = weak_learner.predict(X)
            y_pred_mapped = pd.Series(y_pred).map(self.class_map_).values.astype(float)
            
            # Compute weighted error
            incorrect = (y_pred_mapped != y_mapped)
            error = np.dot(sample_weights, incorrect) / sample_weights.sum()
            
            # Clip error to avoid extreme alpha values
            eps = 1e-10
            error = max(eps, min(error, 1 - eps))
            
            # If error is >= 0.5, break or adjust (algorithm stops)
            if error >= 0.5:
                # Weak learner is no better than random guessing; break
                print(f"  Iteration {t+1}: error = {error:.4f} >= 0.5, stopping early.")
                break
            
            # Compute learner weight
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights *= np.exp(-alpha * y_mapped * y_pred_mapped)
            sample_weights /= sample_weights.sum()  # normalize
            
            # Save learner and its weight
            self.estimators_.append(weak_learner)
            self.estimator_weights_.append(alpha)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # Compute weighted sum of weak learner predictions
        score = self.decision_function(X)
        # Convert score to class labels
        pred_class_idx = (score > 0).astype(int)
        return self.classes_[pred_class_idx]
    
    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        n_samples = X.shape[0]
        score = np.zeros(n_samples)
        
        for weak_learner, alpha in zip(self.estimators_, self.estimator_weights_):
            y_pred = weak_learner.predict(X)
            y_pred_mapped = pd.Series(y_pred).map(self.class_map_).values.astype(float)
            score += alpha * y_pred_mapped
        
        return score
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities of the input samples.
            Column order matches self.classes_.
        """
        score = self.decision_function(X)
        # Convert score to probability using sigmoid
        prob_positive = 1 / (1 + np.exp(-2 * score))  # Platt scaling approximation
        prob_negative = 1 - prob_positive
        # Stack probabilities according to class order
        if self.classes_[0] == 1:  # mapping check
            proba = np.column_stack([prob_positive, prob_negative])
        else:
            proba = np.column_stack([prob_negative, prob_positive])
        return proba
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# ----------------------------------------------------------------------
# Обучение и оценка AdaBoost на датасете Mushroom
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("ADABOOST КЛАССИФИКАЦИЯ - МUSHROOM DATASET")
    print("=" * 60)
    
    # Загрузка датасета
    print("\n1. Загрузка датасета Mushroom (UCI ML Repository, ID=73)...")
    mushroom = fetch_ucirepo(id=73)
    X = mushroom.data.features
    y = mushroom.data.targets.iloc[:, 0]  # столбец 'poisonous'
    
    print(f"   Размерность данных: {X.shape}")
    print(f"   Количество классов: {len(np.unique(y))} ({np.unique(y)})")
    
    # Разделение на обучающую и тестовую выборки (80/20)
    print("\n2. Разделение данных на обучающую (80%) и тестовую (20%) выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"   Тестовая выборка:  {X_test.shape[0]} образцов")
    
    # ------------------------------------------------------------------
    # 3. Обучение одиночного дерева (для сравнения)
    # ------------------------------------------------------------------
    print("\n3. Обучение одиночного дерева решений (max_depth=5)...")
    single_tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    single_tree.learn(X_train, y_train)
    y_pred_single = single_tree.predict(X_test)
    acc_single = accuracy_score(y_test, y_pred_single)
    print(f"   Accuracy одиночного дерева на тесте: {acc_single:.4f}")
    
    # ------------------------------------------------------------------
    # 4. Обучение случайного леса (из task2)
    # ------------------------------------------------------------------
    print("\n4. Обучение случайного леса (n_estimators=30, max_depth=5)...")
    from task2 import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"   Accuracy случайного леса на тесте: {acc_rf:.4f}")
    
    # ------------------------------------------------------------------
    # 5. Обучение AdaBoost
    # ------------------------------------------------------------------
    print("\n5. Обучение AdaBoost (n_estimators=50, weak learner: stump)...")
    ada = AdaBoostClassifier(
        n_estimators=50,
        random_state=42
    )
    ada.fit(X_train, y_train)
    y_pred_ada = ada.predict(X_test)
    acc_ada = accuracy_score(y_test, y_pred_ada)
    print(f"   Accuracy AdaBoost на тесте: {acc_ada:.4f}")
    
    # ------------------------------------------------------------------
    # 6. Сравнение моделей
    # ------------------------------------------------------------------
    print("\n6. Сравнение качества классификации:")
    print(f"   Одиночное дерево (max_depth=5): {acc_single:.4f}")
    print(f"   Случайный лес (30 деревьев):    {acc_rf:.4f}")
    print(f"   AdaBoost (50 stump):            {acc_ada:.4f}")
    
    # ------------------------------------------------------------------
    # 7. График зависимости точности от количества деревьев в AdaBoost
    # ------------------------------------------------------------------
    print("\n7. Построение графика зависимости точности от количества деревьев...")
    n_estimators_range = [1, 5]
    train_scores = []
    test_scores = []
    
    for n in n_estimators_range:
        ada_temp = AdaBoostClassifier(
            n_estimators=n,
            weak_learner_depth=3,
            random_state=42
        )
        ada_temp.fit(X_train, y_train)
        train_scores.append(ada_temp.score(X_train, y_train))
        test_scores.append(ada_temp.score(X_test, y_test))
        print(f"   n_estimators={n:3d}: train_acc={train_scores[-1]:.4f}, test_acc={test_scores[-1]:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_scores, 'b-', label='Обучающая выборка', marker='o')
    plt.plot(n_estimators_range, test_scores, 'r-', label='Тестовая выборка', marker='s')
    plt.xlabel('Количество деревьев (stump) в ансамбле')
    plt.ylabel('Accuracy')
    plt.title('Зависимость точности AdaBoost от количества деревьев')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('task3_1.png', dpi=150)
    print("   График сохранен как 'task3_1.png'")
    
    # ------------------------------------------------------------------
    # 8. Объяснение графика
    # ------------------------------------------------------------------
    print("\n8. Объяснение графика:")
    print("   - С ростом количества деревьев точность на обучающей выборке")
    print("     монотонно увеличивается, так как ансамбль становится сложнее.")
    print("   - Точность на тестовой выборке сначала быстро растёт, затем")
    print("     стабилизируется или может немного снижаться из-за переобучения.")
    print("   - AdaBoost хорошо комбинирует слабые классификаторы, уменьшая")
    print("     ошибку за счёт фокусировки на сложных примерах.")
    
    # ------------------------------------------------------------------
    # 9. Матрица ошибок AdaBoost
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred_ada, labels=['e', 'p'])
    print(f"\n9. Матрица ошибок AdaBoost:")
    print(f"       [[TN FP]   [[{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f"        [FN TP]] =  [{cm[1,0]:4d} {cm[1,1]:4d}]]")
    print(f"       (e='edible', p='poisonous')")
    
    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)