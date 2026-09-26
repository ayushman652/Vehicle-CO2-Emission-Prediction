import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score

def evaluate_model(model, x_test, y_test):
    """ Evaluate the model on test set"""
    predictions = model.predict(x_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return mae,mse,rmse,r2
    