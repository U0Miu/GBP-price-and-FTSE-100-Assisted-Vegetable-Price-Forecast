# fix python path if working locally
from utils import fix_pythonpath_if_working_locally

fix_pythonpath_if_working_locally()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse, mape, r2_score
from darts.models import (
    VARIMA,
    BlockRNNModel,
    NBEATSModel,
    RNNModel,
)
from darts.utils.callbacks import TFMProgressBar
from darts.utils.timeseries_generation import (
    datetime_attribute_timeseries,
    sine_timeseries,
)

# Disable logging and warnings
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

# Generate PyTorch training parameters
def generate_torch_kwargs():
    # Run torch models on CPU and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }

def display_forecast(pred_series, ts_transformed, forecast_type, start_date=None):
    plt.figure(figsize=(8, 5))
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    plt.title(f"R2: {r2_score(ts_transformed.univariate_component(0), pred_series)}")
    plt.legend()


_DEFAULT_PATH = "/GBP-price-and-FTSE-100-Assisted-Vegetable-Price-Forecast-main/"

# Load and Visualize datasets
series_pound = TimeSeries.from_csv(_DEFAULT_PATH +"datasets/market/FTSE100.csv",time_col="week",value_cols="closing price")
series_stock = TimeSeries.from_csv(_DEFAULT_PATH +"datasets/market/GBP2USD.csv",time_col="week")
series_veg1 = TimeSeries.from_csv(_DEFAULT_PATH +"datasets/vegetables/cabbage_prices.csv",time_col="week")
series_veg2 = TimeSeries.from_csv(_DEFAULT_PATH +"datasets/vegetables/carrots_prices.csv",time_col="week")
series_veg3 = TimeSeries.from_csv(_DEFAULT_PATH +"datasets/vegetables/lettuce_prices.csv",time_col="week")
series_veg4 = TimeSeries.from_csv(_DEFAULT_PATH +"datasets/vegetables/onions_prices.csv",time_col="week")


# Normalize (scale) data
scaler_pound, scaler_stock, scaler_veg1, scaler_veg2, scaler_veg3, scaler_veg4 = Scaler(), Scaler(), Scaler(), Scaler(), Scaler(), Scaler()
series_pound_scaled = scaler_pound.fit_transform(series_pound)
series_stock_scaled = scaler_stock.fit_transform(series_stock)
series_veg1_scaled = scaler_veg1.fit_transform(series_veg1)
series_veg2_scaled = scaler_veg1.fit_transform(series_veg2)
series_veg3_scaled = scaler_veg1.fit_transform(series_veg3)
series_veg4_scaled = scaler_veg1.fit_transform(series_veg4)

# Split into training and validation sets
# The dataset contains 320 weeks in total
train_pound, val_pound = series_pound_scaled[:-24], series_pound_scaled[-24:] # Take the last 30 as validation set, and the rest as training set
train_stock, val_stock = series_stock_scaled[:-24], series_stock_scaled[-24:]
train_veg1, val_veg1 = series_veg1_scaled[:-24], series_veg1_scaled[-24:]
train_veg2, val_veg2 = series_veg2_scaled[:-24], series_veg2_scaled[-24:]
train_veg3, val_veg3 = series_veg3_scaled[:-24], series_veg3_scaled[-24:]
train_veg4, val_veg4 = series_veg4_scaled[:-24], series_veg4_scaled[-24:]


# Train the model

# Ensure all time series have matching time indices
train_veg1 = train_veg1.slice_intersect(train_pound).slice_intersect(train_stock)
val_veg1 = val_veg1.slice_intersect(val_pound).slice_intersect(val_stock)
train_veg2 = train_veg2.slice_intersect(train_pound).slice_intersect(train_stock)
val_veg2 = val_veg2.slice_intersect(val_pound).slice_intersect(val_stock)
train_veg3 = train_veg3.slice_intersect(train_pound).slice_intersect(train_stock)
val_veg3 = val_veg3.slice_intersect(val_pound).slice_intersect(val_stock)
train_veg4 = train_veg4.slice_intersect(train_pound).slice_intersect(train_stock)
val_veg4 = val_veg4.slice_intersect(val_pound).slice_intersect(val_stock)

# Instead of using past data from "series_veg", we will stack "series_pound" and "series_stock"
past_train = train_pound.stack(train_stock)
past_val = val_pound.stack(val_stock)

# Model Hyperparameters
input_chunk_length = 48
output_chunk_length = 24

# Model Saving Path
model_dir = "models/nbeats/"
os.makedirs(model_dir, exist_ok=True)

# Define, train and save a model for each vegetable
# Initialize NBEATS multivariate model
model_veg1 = NBEATSModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=100)
# Fit the model using only "past_covariates"(train_pound and train_stock)
model_veg1.fit(series=train_veg1, past_covariates=past_train)
# Save the model
model_veg1.save(model_dir + "cabbage_nbeats.pth")
# Make predictions with a prediction period of 24
pred_veg1 = model_veg1.predict(n=24, past_covariates=past_train)

model_veg2 = NBEATSModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=100)
model_veg2.fit(series=train_veg2, past_covariates=past_train)
model_veg2.save(model_dir + "carrot_nbeats.pth")
pred_veg2 = model_veg2.predict(n=24, past_covariates=past_train)

model_veg3 = NBEATSModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=100)
model_veg3.fit(series=train_veg3, past_covariates=past_train)
model_veg3.save(model_dir + "lettuce_nbeats.pth")
pred_veg3 = model_veg3.predict(n=24, past_covariates=past_train)

model_veg4 = NBEATSModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length, n_epochs=100)
model_veg4.fit(series=train_veg4, past_covariates=past_train)
model_veg4.save(model_dir + "onion_nbeats.pth")
pred_veg4 = model_veg4.predict(n=24, past_covariates=past_train)

