import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

# Import our implementations
from task1 import DecisionTreeClassifier as OurDecisionTreeClassifier
from task2 import RandomForestClassifier as OurRandomForestClassifier
from task3 import AdaBoostClassifier as OurAdaBoostClassifier


def load_and_preprocess_data():
    """Load Mushroom dataset and preprocess for both our and sklearn models."""
    print("=" * 60)
    print("ЗАДАНИЕ 5: СРАВНЕНИЕ С БИБЛИОТЕЧНЫМИ РЕАЛИЗАЦИЯМИ")
    print("=" * 60)
    
    print("\n1. Загрузка датасета Mushroom (UCI ML Repository, ID=73)...")
    mushroom = fetch_ucirepo(id=73)
    X = mushroom.data.features
    y = mushroom.data.targets.iloc[:, 0]  # столбец 'poisonous'
    
    print(f"   Размерность данных: {X.shape}")
    print(f"   Количество классов: {len(np.unique(y))} ({np.unique(y)})")
    
    # Split data
    print("\n2. Разделение данных на обучающую (80%) и тестовую (20%) выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"   Тестовая выборка:  {X_test.shape[0]} образцов")
    
    # For sklearn models, we need to encode categorical features to numeric
    print("\n3. Кодирование категориальных признаков для sklearn...")
    label_encoders = {}
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    for col in X_train.columns:
        le = LabelEncoder()
        # Fit on combined data to avoid unseen labels
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train_encoded[col] = le.transform(X_train[col])
        X_test_encoded[col] = le.transform(X_test[col])
        label_encoders[col] = le
    
    # Encode target labels
    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train)
    y_test_encoded = le_target.transform(y_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_encoded': X_train_encoded,
        'X_test_encoded': X_test_encoded,
        'y_train_encoded': y_train_encoded,
        'y_test_encoded': y_test_encoded,
        'label_encoders': label_encoders,
        'le_target': le_target
    }


def train_and_evaluate_sklearn(models_config, data):
    """Train and evaluate sklearn models."""
    results = {}
    
    for name, config in models_config.items():
        print(f"\n--- Обучение sklearn {name} ---")
        model = config['model'](**config['params'])
        
        start_time = time.time()
        model.fit(data['X_train_encoded'], data['y_train_encoded'])
        train_time = time.time() - start_time
        
        y_pred = model.predict(data['X_test_encoded'])
        y_pred_proba = model.predict_proba(data['X_test_encoded']) if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(data['y_test_encoded'], y_pred)
        precision = precision_score(data['y_test_encoded'], y_pred, average='weighted', zero_division=0)
        recall = recall_score(data['y_test_encoded'], y_pred, average='weighted', zero_division=0)
        f1 = f1_score(data['y_test_encoded'], y_pred, average='weighted', zero_division=0)
        
        results[name] = {
            'model': model,
            'train_time': train_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"   Время обучения: {train_time:.4f} сек")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-score: {f1:.4f}")
    
    return results


def train_and_evaluate_our(models_config, data):
    """Train and evaluate our implementations."""
    results = {}
    
    for name, config in models_config.items():
        print(f"\n--- Обучение нашей реализации {name} ---")
        model = config['model'](**config['params'])
        
        start_time = time.time()
        if name == 'DecisionTreeClassifier':
            model.learn(data['X_train'], data['y_train'])
        else:
            model.fit(data['X_train'], data['y_train'])
        train_time = time.time() - start_time
        
        y_pred = model.predict(data['X_test'])
        y_pred_proba = model.predict_proba(data['X_test']) if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(data['y_test'], y_pred)
        # Use binary classification metrics with 'e' as positive class
        precision = precision_score(data['y_test'], y_pred, pos_label='e', zero_division=0)
        recall = recall_score(data['y_test'], y_pred, pos_label='e', zero_division=0)
        f1 = f1_score(data['y_test'], y_pred, pos_label='e', zero_division=0)
        
        results[name] = {
            'model': model,
            'train_time': train_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"   Время обучения: {train_time:.4f} сек")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-score: {f1:.4f}")
    
    return results


def compare_interpretability(sklearn_results, our_results):
    """Compare interpretability of models."""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ИНТЕРПРЕТИРУЕМОСТИ")
    print("=" * 60)
    
    interpretability_scores = {
        'DecisionTreeClassifier': {
            'sklearn': 9,
            'our': 8,
            'reason': 'Деревья решений легко интерпретируемы, можно визуализировать правила.'
        },
        'RandomForestClassifier': {
            'sklearn': 5,
            'our': 5,
            'reason': 'Ансамбли деревьев сложнее интерпретировать, но можно анализировать важность признаков.'
        },
        'AdaBoostClassifier': {
            'sklearn': 4,
            'our': 4,
            'reason': 'Бустинг создаёт сложные композиции, интерпретируемость низкая.'
        }
    }
    
    for model_name in interpretability_scores.keys():
        score = interpretability_scores[model_name]
        print(f"\n{model_name}:")
        print(f"  sklearn: {score['sklearn']}/10 - {score['reason']}")
        print(f"  Наша реализация: {score['our']}/10")


def compare_supported_features(sklearn_results, our_results):
    """Compare supported features and capabilities."""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПОДДЕРЖИВАЕМЫХ ФИЧ")
    print("=" * 60)
    
    features = {
        'DecisionTreeClassifier': {
            'sklearn': ['Многоклассовая классификация', 'Регрессия', 'Важность признаков', 
                       'Визуализация дерева', 'Предобработка пропусков', 'Веса образцов',
                       'Критерии (gini, entropy, log_loss)', 'Предсказание вероятностей'],
            'our': ['Бинарная/многоклассовая классификация', 'Обработка пропусков (замена маркером)',
                   'Только критерий gini', 'Предсказание классов', 'Ограниченная глубина',
                   'Минимальное количество образцов для разделения/листа']
        },
        'RandomForestClassifier': {
            'sklearn': ['Ансамбль деревьев', 'Бутстрап выборки', 'Случайный выбор признаков',
                       'Важность признаков (Gini, permutation)', 'Out-of-bag оценка',
                       'Параллельное обучение', 'Предсказание вероятностей'],
            'our': ['Ансамбль деревьев', 'Бутстрап выборки', 'Случайный выбор признаков (sqrt, log2, int, float)',
                   'Голосование за классы', 'Предсказание вероятностей (нормализация голосов)',
                   'Однопоточное обучение']
        },
        'AdaBoostClassifier': {
            'sklearn': ['Адаптивный бустинг', 'Разные базовые классификаторы', 'SAMME.R алгоритм',
                       'Веса образцов', 'Предсказание вероятностей', 'Многоклассовая классификация'],
            'our': ['Адаптивный бустинг', 'Деревья решений как слабые классификаторы',
                   'Бинарная классификация', 'Обновление весов образцов', 'Предсказание вероятностей (сигмоид)',
                   'Ранняя остановка при error >= 0.5']
        }
    }
    
    for model_name, feat_dict in features.items():
        print(f"\n{model_name}:")
        print("  sklearn поддерживает:")
        for f in feat_dict['sklearn']:
            print(f"    • {f}")
        print("  Наша реализация поддерживает:")
        for f in feat_dict['our']:
            print(f"    • {f}")


def plot_comparison(sklearn_results, our_results):
    """Create comparison plots."""
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ")
    print("=" * 60)
    
    models = ['DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier']
    
    # Accuracy comparison
    sklearn_acc = [sklearn_results[m]['accuracy'] for m in models]
    our_acc = [our_results[m]['accuracy'] for m in models]
    
    # Training time comparison
    sklearn_time = [sklearn_results[m]['train_time'] for m in models]
    our_time = [our_results[m]['train_time'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy bar plot
    x = np.arange(len(models))
    width = 0.35
    axes[0].bar(x - width/2, sklearn_acc, width, label='sklearn', color='skyblue', edgecolor='black')
    axes[0].bar(x + width/2, our_acc, width, label='Наша реализация', color='lightcoral', edgecolor='black')
    axes[0].set_xlabel('Модель')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Сравнение точности (Accuracy)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (sk, our) in enumerate(zip(sklearn_acc, our_acc)):
        axes[0].text(i - width/2, sk + 0.001, f'{sk:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, our + 0.001, f'{our:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Training time bar plot (log scale for better visualization)
    axes[1].bar(x - width/2, sklearn_time, width, label='sklearn', color='skyblue', edgecolor='black')
    axes[1].bar(x + width/2, our_time, width, label='Наша реализация', color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Модель')
    axes[1].set_ylabel('Время обучения (сек)')
    axes[1].set_title('Сравнение времени обучения')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Use log scale if times vary greatly
    if max(max(sklearn_time), max(our_time)) / min(min(sklearn_time), min(our_time)) > 100:
        axes[1].set_yscale('log')
    
    # Add value labels on bars
    for i, (sk, our) in enumerate(zip(sklearn_time, our_time)):
        axes[1].text(i - width/2, sk + 0.001, f'{sk:.3f}', ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width/2, our + 0.001, f'{our:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('comparison_sklearn_vs_our.png', dpi=150)
    print("   График сохранён как 'comparison_sklearn_vs_our.png'")


def print_summary_table(sklearn_results, our_results):
    """Print comprehensive comparison table."""
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ")
    print("=" * 80)
    
    models = ['DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier']
    
    # Table header
    print(f"{'Модель':<25} {'Метрика':<15} {'sklearn':<12} {'Наша реализация':<18} {'Разница':<10}")
    print("-" * 80)
    
    for model in models:
        sk = sklearn_results[model]
        our = our_results[model]
        
        # Accuracy row
        diff_acc = sk['accuracy'] - our['accuracy']
        print(f"{model:<25} {'Accuracy':<15} {sk['accuracy']:<12.4f} {our['accuracy']:<18.4f} {diff_acc:+.4f}")
        
        # Training time row
        diff_time = sk['train_time'] - our['train_time']
        print(f"{'':<25} {'Time (s)':<15} {sk['train_time']:<12.4f} {our['train_time']:<18.4f} {diff_time:+.4f}")
        
        # F1-score row
        diff_f1 = sk['f1'] - our['f1']
        print(f"{'':<25} {'F1-score':<15} {sk['f1']:<12.4f} {our['f1']:<18.4f} {diff_f1:+.4f}")
        
        print("-" * 80)


def main():
    """Main execution function."""
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Define sklearn models configuration
    sklearn_models = {
        'DecisionTreeClassifier': {
            'model': SklearnDecisionTreeClassifier,
            'params': {
                'criterion': 'gini',
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42
            }
        },
        'RandomForestClassifier': {
            'model': SklearnRandomForestClassifier,
            'params': {
                'n_estimators': 30,
                'criterion': 'gini',
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'AdaBoostClassifier': {
            'model': SklearnAdaBoostClassifier,
            'params': {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': 42
            }
        }
    }
    
    # Define our models configuration
    our_models = {
        'DecisionTreeClassifier': {
            'model': OurDecisionTreeClassifier,
            'params': {
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42
            }
        },
        'RandomForestClassifier': {
            'model': OurRandomForestClassifier,
            'params': {
                'n_estimators': 30,
                'max_depth': 5,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42
            }
        },
        'AdaBoostClassifier': {
            'model': OurAdaBoostClassifier,
            'params': {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'weak_learner_depth': 3,
                'random_state': 42
            }
        }
    }
    
    # Train and evaluate sklearn models
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ И ОЦЕНКА SKLEARN МОДЕЛЕЙ")
    print("=" * 60)
    sklearn_results = train_and_evaluate_sklearn(sklearn_models, data)
    
    # Train and evaluate our models
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ И ОЦЕНКА НАШИХ РЕАЛИЗАЦИЙ")
    print("=" * 60)
    our_results = train_and_evaluate_our(our_models, data)
    
    # Generate comparison plots
    plot_comparison(sklearn_results, our_results)
    
    # Print summary table
    print_summary_table(sklearn_results, our_results)
    
    # Compare interpretability
    compare_interpretability(sklearn_results, our_results)
    
    # Compare supported features
    compare_supported_features(sklearn_results, our_results)
    
    # Final conclusions
    print("\n" + "=" * 60)
    print("ВЫВОДЫ")
    print("=" * 60)
    print("1. Точность: sklearn модели обычно показывают сравнимую или немного лучшую точность")
    print("   благодаря оптимизированным алгоритмам и дополнительным эвристикам.")
    print("2. Скорость: sklearn значительно быстрее благодаря реализации на C++ и параллелизации.")
    print("3. Интерпретируемость: Обе реализации деревьев интерпретируемы, но sklearn")
    print("   предоставляет больше инструментов для визуализации и анализа.")
    print("4. Функциональность: sklearn поддерживает больше возможностей (регрессия,")
    print("   многоклассовая классификация, различные критерии, веса образцов и т.д.).")
    print("5. Наши реализации демонстрируют понимание алгоритмов и могут быть полезны")
    print("   для образовательных целей, но для production рекомендуется использовать sklearn.")
    
    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАДАНИЯ 5 ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    # Activate virtual environment implicitly (assumed already activated)
    main()