# hyperparameter_tuning.py
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def tune_hyperparameters(model, model_name, X_train, y_train):
    # Define parameter grids for each model
    param_grids = {
        'ElasticNet': {
            'alpha': np.logspace(-4, 4, 20),
            'l1_ratio': np.linspace(0, 1, 20)
        },
        'Ridge': {
            'alpha': np.logspace(-4, 4, 20)
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    # Perform Randomized Search
    param_grid = param_grids[model_name]
    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=20, cv=5, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    
    print(f"Best parameters for {model_name}: {search.best_params_}")
    return search.best_estimator_
