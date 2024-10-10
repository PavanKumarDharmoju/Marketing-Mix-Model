import numpy as np
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
from modules.data_preprocessing import load_and_preprocess_data

# Load the preprocessed data
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data('./data/marketing_campaign.csv')

# Define models and hyperparameter grids
models = {
    'BayesianRidge': BayesianRidge(),
    'ElasticNet': ElasticNet(),
    'Ridge': Ridge()
}

param_grids = {
    'BayesianRidge': {},
    'ElasticNet': {
        'alpha': [0.1, 1.0, 10],
        'l1_ratio': [0.1, 0.5, 0.9]
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10, 100]
    }
}

# Train models and tune hyperparameters
best_models = {}
for model_name, model in models.items():
    print(f'Training {model_name}...')
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    # Save the best model
    joblib.dump(best_model, f'{model_name}_best_model.pkl')

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{model_name} Test MSE: {mse}')

# Save the preprocessor
joblib.dump(preprocessor, 'data_preprocessor.pkl')
