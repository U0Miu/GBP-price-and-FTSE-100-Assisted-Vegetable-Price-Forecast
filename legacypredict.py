# predict.py
import argparse
import pickle
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.models import TransformerModel
from darts.metrics import mape

def main(vegetable_name):
    # Load data
    market_data = load_market_data()
    raw_series = load_vegetable_data(vegetable_name)
    
    # Load model and scaler
    model, scalers = load_model_and_scalers()
    
    # Prepare data for prediction
    scaled_series = scalers[vegetable_name].transform(raw_series)
    historical_series = scaled_series.split_after(0.8)[0]
    
    # Predict
    prediction = model.predict(
        n=76,
        series=historical_series,
        past_covariates=scalers['market'].transform(market_data)
    )
    
    # Post-processing and visualization
    prediction_unscaled = scalers[vegetable_name].inverse_transform(prediction)
    visualize_results(raw_series, prediction_unscaled, vegetable_name)
    calculate_metrics(raw_series, prediction_unscaled, vegetable_name)

def load_market_data():
    """Load market data and merge"""
    series_gbp = TimeSeries.from_csv("datasets/market/GBP2USD.csv", time_col="week")
    series_ftse = TimeSeries.from_csv("datasets/market/FTSE100.csv", time_col="week")
    return concatenate([series_gbp, series_ftse], axis=1)

def load_vegetable_data(vegetable_name):
    """Load vegetable data based on their names"""
    file_map = {
        'carrots': 'datasets/vegetables/carrots_prices.csv',
        'onions': 'datasets/vegetables/onions_prices.csv',
        'cabbage': 'datasets/vegetables/cabbage_prices.csv',
        'lettuce': 'datasets/vegetables/lettuce_prices.csv'
    }
    return TimeSeries.from_csv(file_map[vegetable_name], time_col="week")

def load_model_and_scalers():
    """Load trained model and scaler"""
    model = TransformerModel.load("models/vegetable_transformer.pth")
    with open("models/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    return model, scalers

def visualize_results(raw_series, prediction, vegetable_name):
    """result visualization"""
    plt.figure(figsize=(10, 6))
    raw_series.split_after(0.8)[1].plot(label="Actual")
    prediction.plot(label="Forecast")
    plt.title(f"{vegetable_name.capitalize()} Price Forecast")
    plt.legend()
    plt.savefig(f"results/{vegetable_name}_forecast.png")
    plt.close()

def calculate_metrics(raw_series, prediction, vegetable_name):
    """Calculate evaluation metrics"""
    actual = raw_series.split_after(0.8)[1]
    mape_val = mape(actual, prediction)
    print(f"MAPE for {vegetable_name}: {mape_val:.2f}%")
    with open(f"results/{vegetable_name}_metrics.txt", "w") as f:
        f.write(f"MAPE: {mape_val:.2f}%")

# python predict.py --vegetable carrots
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vegetable Price Forecasting')
    parser.add_argument('--vegetable', type=str, required=True,
                        choices=['carrots', 'onions', 'cabbage', 'lettuce'],
                        help='Name of the vegetable to forecast')
    args = parser.parse_args()
    
    main(args.vegetable)
