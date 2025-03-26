# fix python path if working locally
from utils import fix_pythonpath_if_working_locally

fix_pythonpath_if_working_locally()
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.metrics import mape
from darts.utils.callbacks import TFMProgressBar

logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")

def generate_torch_kwargs():
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }

# Load data and create the time series

series_gbp = TimeSeries.from_csv("datasets/market/GBP2USD.csv", time_col="week")
series_ftse = TimeSeries.from_csv("datasets/market/FTSE100.csv", time_col="week")
series_carrots = TimeSeries.from_csv("datasets/vegetables/carrots_prices.csv", time_col="week")
series_onions = TimeSeries.from_csv("datasets/vegetables/onions_prices.csv", time_col="week")
series_cabbage = TimeSeries.from_csv("datasets/vegetables/cabbage_prices.csv", time_col="week")
series_lettuce = TimeSeries.from_csv("datasets/vegetables/lettuce_prices.csv", time_col="week")

# Data standardization
# Initialize the Scaler for each vegetable
scaler_carrots = Scaler()
scaler_onions = Scaler()
scaler_cabbage = Scaler()
scaler_lettuce = Scaler()

# Standardized vegetable price
series_carrots_scaled = scaler_carrots.fit_transform(series_carrots)
series_onions_scaled = scaler_onions.fit_transform(series_onions)
series_cabbage_scaled = scaler_cabbage.fit_transform(series_cabbage)
series_lettuce_scaled = scaler_lettuce.fit_transform(series_lettuce)

# Merge and standardize market data.
series_market = concatenate([series_gbp, series_ftse], axis=1)
scaler_market = Scaler()
series_market_scaled = scaler_market.fit_transform(series_market)

# Create a training set and a validation set
target_scaled = [
    series_carrots_scaled,
    series_onions_scaled,
    series_cabbage_scaled,
    series_lettuce_scaled
]

# Split the data in an 80-20 ratio.
# Split each vegetable series along time dimension
train_series = [ts[:-64] for ts in target_scaled]
val_series = [ts[-64:] for ts in target_scaled]

# Split covariates along time dimension
train_cov = [series_market_scaled[:-64]] * 4
val_cov = [series_market_scaled[-64:]] * 4

# 参数网格搜索和可视化部分
param_grid = {
    'd_model': [32, 64, 128],  # 测试不同的模型维度
    'nhead': [2, 4]            # 测试不同的注意力头数
    
}
results = []

# 遍历所有参数组合
for d_model in param_grid['d_model']:
    for nhead in param_grid['nhead']:
        print(f"\nTraining with d_model={d_model}, nhead={nhead}")
        
        # 初始化模型（使用不同参数）
        model = TransformerModel(
            input_chunk_length=12,
            output_chunk_length=24,
            batch_size=32,
            n_epochs=10,  # 减少训练轮次以加快演示速度
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=2,
            num_decoder_layers=3,
            dim_feedforward=256,
            dropout=0.3,
            activation="gelu",
            optimizer_kwargs={"lr": 0.001},
            model_name=f"transformer_d{d_model}_h{nhead}",
            save_checkpoints=False,  # 禁用检查点保存以节省空间
            force_reset=True,
            **generate_torch_kwargs()
        )

        
        model.fit(
            series=train_series,
            past_covariates=train_cov,
            val_series=val_series,
            val_past_covariates=val_cov,
            verbose=False
        )

       
        predictions = []
        for train_ts, val_ts, cov in zip(train_series, val_series, [series_market_scaled]*4):
        
            pred = model.predict(
               n=88, 
               series=train_ts,  
               past_covariates=cov
            )
            predictions.append(pred)

        
        mape_values = [mape(ts, pred) for ts, pred in zip(val_series, predictions)]
        avg_mape = np.mean(mape_values)
        results.append({
            'd_model': d_model,
            'nhead': nhead,
            'mape': avg_mape
        })

results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 8))

heatmap_data = results_df.pivot(index="d_model", columns="nhead", values="mape")

sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'MAPE (%)'})
plt.title("MAPE under Different Model Parameters")
plt.xlabel("Number of Attention Heads")
plt.ylabel("Model Dimension (d_model)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for nhead in param_grid['nhead']:
    subset = results_df[results_df['nhead'] == nhead]
    plt.plot(subset['d_model'], subset['mape'], 
             marker='o', linestyle='--', 
             label=f'nhead={nhead}')

plt.title("MAPE vs Model Dimension with Different Heads")
plt.xlabel("Model Dimension (d_model)")
plt.ylabel("MAPE (%)")
plt.legend()
plt.grid(True)
plt.show()

results_df.to_csv("parameter_tuning_results.csv", index=False)