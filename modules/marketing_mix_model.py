from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_preprocessing import load_and_preprocess_data
import joblib

def train_marketing_mix_models():
    # Load preprocessed data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data('marketing_campaign.csv')

    # Define models
    models = {
        'BayesianRidge': BayesianRidge(),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Ridge': Ridge(alpha=1.0)
    }

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f'Training {model_name}...')
        model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(model, f'{model_name}_marketing_mix_model.pkl')

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'{model_name} Test MSE: {mse}')

    # Save the preprocessor
    joblib.dump(preprocessor, 'data_preprocessor.pkl')

if __name__ == "__main__":
    train_marketing_mix_models()
