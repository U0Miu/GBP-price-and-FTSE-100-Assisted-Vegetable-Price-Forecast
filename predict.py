# predict.py
import argparse
import pickle
import os
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.models import TransformerModel, TCNModel, NBEATSModel
from darts.metrics import mape, smape, mae

MODEL_MAP = {
    'tcn': TCNModel,
    'transformer': TransformerModel,
    'nbeats': NBEATSModel
}

_DEFAULT_PATH = "/GBP-price-and-FTSE-100-Assisted-Vegetable-Price-Forecast-main/"

def main(vegetable_name, model_type='transformer', week = 24):
    # Create a dedicated result directory for the model.
    os.makedirs(f"results/{model_type}", exist_ok=True)
    
    # Load data
    market_data = load_market_data()
    raw_series = load_vegetable_data(vegetable_name)
    
    # Load model and scaler
    model, scalers = load_model_and_scalers(model_type)
    
    scaled_series = scalers[vegetable_name].transform(raw_series)
    historical_series = scaled_series.split_after(0.8)[0]
    
    prediction = model.predict(
        n = 64 + week,
        series=historical_series,
        past_covariates=scalers['market'].transform(market_data)
    )
    
    prediction_unscaled = scalers[vegetable_name].inverse_transform(prediction)
    
    # Result save path
    visualize_results(raw_series, prediction_unscaled, vegetable_name, model_type)
    calculate_metrics(raw_series, prediction_unscaled, vegetable_name, model_type)
    save_predictions(prediction_unscaled, vegetable_name, model_type)


def load_market_data():
    """Load and merge market data."""
    series_gbp = TimeSeries.from_csv(_DEFAULT_PATH +"datasets/market/GBP2USD.csv", time_col="week")
    series_ftse = TimeSeries.from_csv(_DEFAULT_PATH +"datasets/market/FTSE100.csv", time_col="week")
    return concatenate([series_gbp, series_ftse], axis=1)

def load_vegetable_data(vegetable_name):
    """Load the corresponding data based on the vegetable name."""
    file_map = {
        'carrots': _DEFAULT_PATH +'datasets/vegetables/carrots_prices.csv',
        'onions': _DEFAULT_PATH +'datasets/vegetables/onions_prices.csv',
        'cabbage': _DEFAULT_PATH +'datasets/vegetables/cabbage_prices.csv',
        'lettuce': _DEFAULT_PATH +'datasets/vegetables/lettuce_prices.csv'
    }
    return TimeSeries.from_csv(file_map[vegetable_name], time_col="week")

def load_model_and_scalers(model_type):
    """Load the model and scaler based on the model type."""
    model_path = f"models/{model_type}/vegetable_{model_type}.pth"
    scaler_path = f"models/{model_type}/scalers.pkl"
    
    # Model selection
    model = MODEL_MAP[model_type].load(model_path)
    
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
        
    return model, scalers

# Visualization
def visualize_results(raw_series, prediction, vegetable_name, model_type):
    plt.figure(figsize=(10, 6))
    raw_series.split_after(0.8)[1].plot(label="Actual")
    prediction.plot(label="Forecast")
    plt.title(f"{vegetable_name.capitalize()} Price Forecast ({model_type.upper()})")
    plt.legend()
    plt.savefig(f"results/{model_type}/{vegetable_name}_forecast.png")
    plt.close()

# Evaluation
def calculate_metrics(raw_series, prediction, vegetable_name, model_type):
    actual = raw_series.split_after(0.8)[1]
    mape_val = mape(actual, prediction)
    smape_val = smape(actual, prediction)
    mae_val = mae(actual, prediction)
    print(f"{model_type.upper()} - MAPE for {vegetable_name}: {mape_val:.2f}%")
    print(f"{model_type.upper()} - SMAPE for {vegetable_name}: {smape_val:.2f}%")
    print(f"{model_type.upper()} - MAE for {vegetable_name}: {mae_val:.2f}")
    
    with open(f"results/{model_type}/{vegetable_name}_metrics.txt", "w") as f:
        f.write(f"MAPE: {mape_val:.2f}%")
        f.write(f"SMAPE: {smape_val:.2f}%\n")
        f.write(f"MAE: {mae_val:.2f}\n")

def save_predictions(prediction_series, vegetable_name, model_type):
    """Save the forecast results as a CSV file."""
    # Convert to a Pandas DataFrame.
    df = prediction_series.pd_dataframe().reset_index()
    df.columns = ['Date', 'Predicted_Price']
    
    # Set the save path.
    save_dir = f"results/{model_type}"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/{vegetable_name}_predictions.csv"
    
    # Save csv file
    df.to_csv(csv_path, index=False)
    print(f"The forecast results have been saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vegetable Price Forecasting')
    parser.add_argument('--vegetable', type=str, required=True,
                        choices=['carrots', 'onions', 'cabbage', 'lettuce'])
    
    parser.add_argument('--model', type=str, default='transformer',
                        choices=['tcn', 'transformer','nbeats'],
                        help='Model type (tcn, transformer or nbeats)')
    
    parser.add_argument('--week', type=int, default=24,
                        choices=range(1,25),
                        help='Week 1 to 24'
                        )
    args = parser.parse_args()
    
    main(args.vegetable, args.model.lower(), args.week)
