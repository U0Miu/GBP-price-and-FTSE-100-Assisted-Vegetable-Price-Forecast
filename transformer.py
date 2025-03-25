# fix python path if working locally
from utils import fix_pythonpath_if_working_locally

fix_pythonpath_if_working_locally()
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
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

# 设置随机种子
torch.manual_seed(1)
np.random.seed(1)

def generate_torch_kwargs():
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }

# 加载数据并创建时间序列

series_gbp = TimeSeries.from_csv("datasets/market/GBP2USD.csv", time_col="week")
series_ftse = TimeSeries.from_csv("datasets/market/FTSE100.csv", time_col="week")
series_carrots = TimeSeries.from_csv("datasets/vegetables/carrots_prices.csv", time_col="week")
series_onions = TimeSeries.from_csv("datasets/vegetables/onions_prices.csv", time_col="week")
series_cabbage = TimeSeries.from_csv("datasets/vegetables/cabbage_prices.csv", time_col="week")
series_lettuce = TimeSeries.from_csv("datasets/vegetables/lettuce_prices.csv", time_col="week")

# 数据标准化
# 初始化每个蔬菜的Scaler
scaler_carrots = Scaler()
scaler_onions = Scaler()
scaler_cabbage = Scaler()
scaler_lettuce = Scaler()

# 标准化蔬菜价格数据
series_carrots_scaled = scaler_carrots.fit_transform(series_carrots)
series_onions_scaled = scaler_onions.fit_transform(series_onions)
series_cabbage_scaled = scaler_cabbage.fit_transform(series_cabbage)
series_lettuce_scaled = scaler_lettuce.fit_transform(series_lettuce)

# 合并并标准化市场数据
series_market = concatenate([series_gbp, series_ftse], axis=1)
scaler_market = Scaler()
series_market_scaled = scaler_market.fit_transform(series_market)

# 创建训练集和验证集
target_scaled = [
    series_carrots_scaled,
    series_onions_scaled,
    series_cabbage_scaled,
    series_lettuce_scaled
]

# 按80-20比例分割数据
# Split each vegetable series along time dimension
train_series = [ts[:-64] for ts in target_scaled]
val_series = [ts[-64:] for ts in target_scaled]

# Split covariates along time dimension
train_cov = [series_market_scaled[:-64]] * 4
val_cov = [series_market_scaled[-64:]] * 4

# 初始化Transformer模型
model = TransformerModel(
    input_chunk_length=24,
    output_chunk_length=12,
    batch_size=32,
    n_epochs=10,
    d_model=64,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dropout=0.1,
    optimizer_kwargs={"lr": 1e-3},
    model_name="vegetable_transformer",
    save_checkpoints=True,
    force_reset=True,
    **generate_torch_kwargs()
)

# 训练模型
model.fit(
    series=train_series,
    past_covariates=train_cov,
    val_series=val_series,
    val_past_covariates=val_cov,
    verbose=True
)

# 保存模型和scaler
model.save("models/transformer/vegetable_transformer.pth")
with open("models/transformer/scalers.pkl", "wb") as f:
    pickle.dump({
        'carrots': scaler_carrots,
        'onions': scaler_onions,
        'cabbage': scaler_cabbage,
        'lettuce': scaler_lettuce,
        'market': scaler_market
    }, f)

