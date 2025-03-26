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
from darts.models import NBEATSModel
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

# Initialize the Nbeats model
model = NBEATSModel(
    input_chunk_length=20,  # Number of past time steps
    output_chunk_length=10,  # Number of future time steps to predict
    n_epochs=10,
    random_state=0,
    **generate_torch_kwargs(),
)

# Training model
model.fit(
    series=train_series,
    past_covariates=train_cov,
    val_series=val_series,
    val_past_covariates=val_cov,
    verbose=True
)

# Save the model and scaler.
model.save("models/nbeats/vegetable_nbeats.pth")
with open("models/nbeats/scalers.pkl", "wb") as f:
    pickle.dump({
        'carrots': scaler_carrots,
        'onions': scaler_onions,
        'cabbage': scaler_cabbage,
        'lettuce': scaler_lettuce,
        'market': scaler_market
    }, f)

