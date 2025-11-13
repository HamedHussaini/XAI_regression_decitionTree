# isolated_run.py
from train import train
from predict import predict

# === Local test run ===
train("example_data/historic_brazil.csv", "output/brazil_model.bin")

# Use real historic and future data
predict(
    "output/brazil_model.bin",
    "example_data/historic_brazil.csv",
    "example_data/future_brazil.csv",
    "output/predictions.csv"
)
