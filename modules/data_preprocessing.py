import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path, delimiter=';')

    # Drop irrelevant columns and handle missing data
    data.drop(['ID', 'Dt_Customer'], axis=1, inplace=True)
    data['Income'] = data['Income'].fillna(data['Income'].median())  # Impute missing Income values
    
    # Convert categorical variables using OneHotEncoding
    categorical_features = ['Education', 'Marital_Status']
    numerical_features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
                          'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                          'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                          'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                          'NumWebVisitsMonth']

    # Define transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Apply transformations and return preprocessed data
    X = data.drop(['Response'], axis=1)
    y = data['Response']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor
