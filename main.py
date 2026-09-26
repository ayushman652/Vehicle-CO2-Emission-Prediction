from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.visualizer import plot_predictions 


def main():
    df = load_data()
    
    x_train, x_test, y_train, y_test, scaler = preprocess_data(df)
    
    model = train_model(x_train, y_train)
    
    original_coefficients = model.coef_ / scaler.scale_
    original_intercept = model.intercept_ - sum(original_coefficients * scaler.mean_)
    
    print("\noriginal unit coefficiants:", original_coefficients)
    print("original intercept:", original_intercept)
    
    mae, mse, rmse, r2 = evaluate_model(
    model,
    x_test,
    y_test
    )
    
    print("\nModel Evaluation")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")    
    
    predictions = model.predict(x_test)

    plot_predictions(y_test, predictions)
    


if __name__ == "__main__":
    main()    