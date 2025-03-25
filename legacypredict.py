# predict.py
import argparse
import pickle
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.models import TransformerModel
from darts.metrics import mape

def main(vegetable_name):
    # 加载数据
    market_data = load_market_data()
    raw_series = load_vegetable_data(vegetable_name)
    
    # 加载模型和scaler
    model, scalers = load_model_and_scalers()
    
    # 准备预测数据
    scaled_series = scalers[vegetable_name].transform(raw_series)
    historical_series = scaled_series.split_after(0.8)[0]
    
    # 生成预测
    prediction = model.predict(
        n=76,
        series=historical_series,
        past_covariates=scalers['market'].transform(market_data)
    )
    
    # 后处理与可视化
    prediction_unscaled = scalers[vegetable_name].inverse_transform(prediction)
    visualize_results(raw_series, prediction_unscaled, vegetable_name)
    calculate_metrics(raw_series, prediction_unscaled, vegetable_name)

def load_market_data():
    """加载市场数据并合并"""
    series_gbp = TimeSeries.from_csv("datasets/market/GBP2USD.csv", time_col="week")
    series_ftse = TimeSeries.from_csv("datasets/market/FTSE100.csv", time_col="week")
    return concatenate([series_gbp, series_ftse], axis=1)

def load_vegetable_data(vegetable_name):
    """根据蔬菜名称加载对应数据"""
    file_map = {
        'carrots': 'datasets/vegetables/carrots_prices.csv',
        'onions': 'datasets/vegetables/onions_prices.csv',
        'cabbage': 'datasets/vegetables/cabbage_prices.csv',
        'lettuce': 'datasets/vegetables/lettuce_prices.csv'
    }
    return TimeSeries.from_csv(file_map[vegetable_name], time_col="week")

def load_model_and_scalers():
    """加载训练好的模型和scaler"""
    model = TransformerModel.load("models/vegetable_transformer.pth")
    with open("models/scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    return model, scalers

def visualize_results(raw_series, prediction, vegetable_name):
    """结果可视化"""
    plt.figure(figsize=(10, 6))
    raw_series.split_after(0.8)[1].plot(label="Actual")
    prediction.plot(label="Forecast")
    plt.title(f"{vegetable_name.capitalize()} Price Forecast")
    plt.legend()
    plt.savefig(f"results/{vegetable_name}_forecast.png")
    plt.close()

def calculate_metrics(raw_series, prediction, vegetable_name):
    """计算评估指标"""
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