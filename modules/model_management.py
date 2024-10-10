import joblib

def save_model(model, file_name):
    '''Save a model to a file using joblib'''
    joblib.dump(model, file_name)
    print(f'Model saved to {file_name}')

def load_model(file_name):
    '''Load a model from a file using joblib'''
    model = joblib.load(file_name)
    print(f'Model loaded from {file_name}')
    return model
