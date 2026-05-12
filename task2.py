import numpy as np
import pandas as pd
from typing import Optional, Union
import matplotlib.pyplot as plt
from task1 import DecisionTreeClassifier


class RandomForestClassifier:
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = 'sqrt'
        self.bootstrap = True
        self.random_state = random_state
        
        self.estimators_ = []
        self.feature_indices_ = []  # stores which features were used for each tree
        self.classes_ = None
        self.n_features_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'RandomForestClassifier':
        # Convert to pandas for consistency with DecisionTreeClassifier
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_ = np.unique(y)
        
        # Build trees
        self.estimators_ = []
        self.feature_indices_ = []
        
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X.iloc[indices].reset_index(drop=True)
            y_boot = y.iloc[indices].reset_index(drop=True)
            
            n_features_tree = int(np.sqrt(n_features))
            
            n_features_tree = max(1, min(n_features_tree, n_features))
            feature_idx = np.random.choice(n_features, size=n_features_tree, replace=False)
            self.feature_indices_.append(feature_idx)
            
            # Subset of features for this tree
            X_tree = X_boot.iloc[:, feature_idx]
            
            # Create and train tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i if self.random_state is not None else None
            )
            tree.learn(X_tree, y_boot)
            self.estimators_.append((tree, feature_idx))
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        proba = self.predict_with_probability(X)
        # Choose class with highest probability
        class_idx = np.argmax(proba, axis=1)
        return self.classes_[class_idx]
    
    def predict_with_probability(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize array for collecting votes
        votes = np.zeros((n_samples, n_classes))
        
        for (tree, feature_idx) in self.estimators_:
            # Use only the features that this tree was trained on
            X_tree = X.iloc[:, feature_idx]
            pred = tree.predict(X_tree)  # shape (n_samples,)
            
            # Convert predictions to indices
            for i, class_label in enumerate(self.classes_):
                votes[:, i] += (pred == class_label)
        
        # Convert votes to probabilities
        proba = votes / len(self.estimators_)
        return proba
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("СЛУЧАЙНЫЙ ЛЕС КЛАССИФИКАЦИИ - МUSHROOM DATASET")
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
    print("\n3. Обучение одиночного дерева решений (для сравнения)...")
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
    # 4. Обучение случайного леса
    # ------------------------------------------------------------------
    print("\n4. Обучение случайного леса (n_estimators=30)...")
    rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"   Accuracy случайного леса на тесте: {acc_rf:.4f}")
    
    # Сравнение
    print(f"\n5. Сравнение качества классификации:")
    print(f"   Одиночное дерево: {acc_single:.4f}")
    print(f"   Случайный лес:    {acc_rf:.4f}")
    print(f"   Улучшение:        {acc_rf - acc_single:.4f}")
    
    # ------------------------------------------------------------------
    # 6. График зависимости точности от количества деревьев в ансамбле
    # ------------------------------------------------------------------
    print("\n6. Построение графика зависимости точности от количества деревьев...")
    n_trees_range = [1, 5, 10, 20, 30]
    train_scores = []
    test_scores = []
    
    for n in n_trees_range:
        rf_temp = RandomForestClassifier(
            n_estimators=n,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        rf_temp.fit(X_train, y_train)
        train_scores.append(rf_temp.score(X_train, y_train))
        test_scores.append(rf_temp.score(X_test, y_test))
        print(f"   n_trees={n:3d}: train_acc={train_scores[-1]:.4f}, test_acc={test_scores[-1]:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_trees_range, train_scores, 'b-', label='Обучающая выборка', marker='o')
    plt.plot(n_trees_range, test_scores, 'r-', label='Тестовая выборка', marker='s')
    plt.xlabel('Количество деревьев в ансамбле')
    plt.ylabel('Accuracy')
    plt.title('Зависимость точности случайного леса от количества деревьев')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('task2_1.png', dpi=150)
    print("   График сохранён как 'task2_1.png'")
    
    # ------------------------------------------------------------------
    # 7. Визуализация уверенности модели в ответах
    # ------------------------------------------------------------------
    print("\n7. Визуализация уверенности модели в ответах...")
    
    # Получаем вероятности для тестовой выборки
    proba = rf.predict_with_probability(X_test)
    # Уверенность = максимальная вероятность среди классов
    confidence = np.max(proba, axis=1)
    
    # Определяем, какие объекты классифицированы неверно
    incorrect = (y_pred_rf != y_test)
    correct = ~incorrect
    
    # Создаём график
    plt.figure(figsize=(12, 5))
    
    # Гистограмма уверенности для правильно классифицированных объектов
    plt.subplot(1, 2, 1)
    plt.hist(confidence[correct], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Уверенность модели')
    plt.ylabel('Количество объектов')
    plt.title('Правильно классифицированные объекты')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Гистограмма уверенности для неверно классифицированных объектов
    plt.subplot(1, 2, 2)
    plt.hist(confidence[incorrect], bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Уверенность модели')
    plt.ylabel('Количество объектов')
    plt.title('Неверно классифицированные объекты')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('task2_2.png', dpi=150)
    print("   Гистограммы сохранены как 'task2_2.png'")
    
    # ------------------------------------------------------------------
    # 8. Дополнительная информация
    # ------------------------------------------------------------------
    print("\n8. Дополнительная информация:")
    print(f"   Количество неверно классифицированных объектов: {incorrect.sum()} из {len(y_test)}")
    print(f"   Средняя уверенность на правильных: {confidence[correct].mean():.4f}")
    print(f"   Средняя уверенность на ошибках:    {confidence[incorrect].mean():.4f}")
    
    # Матрица ошибок для случайного леса
    cm = confusion_matrix(y_test, y_pred_rf, labels=['e', 'p'])
    print(f"\n   Матрица ошибок случайного леса:")
    print(f"       [[TN FP]   [[{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f"        [FN TP]] =  [{cm[1,0]:4d} {cm[1,1]:4d}]]")
    print(f"       (e='edible', p='poisonous')")
    
    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)