import requests
from flask import Flask, request, jsonify
import subprocess
import os
import pandas as pd
import re

# ==========================
# Flask Web App
# Oracle.py
# ├── predict_future()       # Main dispatch function: calls predict.py and reads the result
# ├── Flask endpoint /oracle # Receives user queries and returns forecast results
# ├── run_server()           # Launches the Flask server (for API access)
# ├── run_client()           # Terminal-based chatbot client
# └── __main__               # Program entry point: choose between server/client modes
# ==========================

app = Flask(__name__)

# Vegetable price prediction function (calls predict.py and reads the forecast result)
def predict_future(query):
    vegetable_map = {
        "carrot price": "carrots",
        "onion price": "onions",
        "cabbage price": "cabbage",
        "lettuce price": "lettuce"
    }

    # Extract the target forecast week: default is week 24
    week_pattern = re.search(r"week\s*(\d{1,2})", query.lower())
    target_week = int(week_pattern.group(1)) if week_pattern else 24

    # Limit the maximum forecast horizon to 24 weeks
    if target_week > 24:
        return {
            "message": "⚠️ Sorry, I can only forecast up to 24 weeks.",
            "tip": "Please ask for a week between 1 and 24."
        }

    # Match vegetable keyword
    for key, vegetable in vegetable_map.items():
        if key in query.lower():
            model_type = "transformer"

            command = [
                "python", "predict.py",
                "--vegetable", vegetable,
                "--model", model_type,
                "--week", str(target_week)
            ]

            try:
                subprocess.run(command, capture_output=True, text=True, check=True)

                # Load prediction result
                prediction_file = f"results/{model_type}/{vegetable}_predictions.csv"
                if not os.path.exists(prediction_file):
                    return {
                        "message": "Prediction completed, but result file not found."
                    }

                df = pd.read_csv(prediction_file)

                # Index starts from 0, so week N corresponds to index N-1
                idx = target_week - 1
                if idx >= len(df):
                    return {
                        "message": f"Prediction only covers {len(df)} weeks, but week {target_week} was requested."
                    }

                forecast_price = float(df.loc[idx, 'Predicted_Price'])

                # Load actual value for comparison
                actual_price = None
                try:
                    veg_data = pd.read_csv(f"datasets/vegetables/{vegetable}_prices.csv")
                    if target_week - 1 < len(veg_data):
                        actual_price = float(veg_data.iloc[target_week - 1]['price'])  # Assuming column name is 'price'
                except Exception as e:
                    actual_price = None

                result = {
                    "message": f"✅ Prediction for {vegetable.capitalize()} - Week {target_week}.",
                    "predicted_price": round(forecast_price, 2)
                }
                if actual_price:
                    result["actual_price"] = round(actual_price, 2)
                return result

            except subprocess.CalledProcessError as e:
                return {
                    "message": "❌ Error during prediction.",
                    "details": e.output
                }

    return {
        "message": "⚠️ Sorry, I cannot predict this. Try asking about carrot, onion, cabbage or lettuce prices."
    }


# Flask endpoint: handles incoming prediction requests
@app.route("/oracle", methods=["POST"])
def oracle():
    data = request.json
    question = data.get("question", "")
    result = predict_future(question)
    return jsonify(result)

# Run the Flask server
def run_server():
    print("🔌 Server started, listening on port 5000...")
    app.run(host="0.0.0.0", port=5000)

# ==========================
# Client Mode (Terminal)
# ==========================

def run_client():
    print("Oracle: Hello, I'm the Oracle. How can I help you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "bye", "goodbye"]:
            print("Oracle: Goodbye!")
            break

        try:
            response = requests.post("http://127.0.0.1:5000/oracle", json={"question": user_input})
            if response.status_code == 200:
                data = response.json()
                print(f"Oracle: {data['message']}")
                if 'predicted_price' in data and 'message' in data:
                    # Extract vegetable and week from message
                    match = re.search(r"Prediction for (\w+) - Week (\d+)", data['message'])
                    if match:
                        veg = match.group(1).lower()
                        week = match.group(2)
                        price = data['predicted_price']
                        print(f"Oracle: 💰 The predicted price of {veg} in week {week} is {price}.")
                        if 'actual_price' in data:
                            print(f"📊 Actual price for week {week} was {data['actual_price']}.")
                        print("What else would you like to do today?")
                    else:
                        print(f"Oracle: {data['message']}")
            else:
                print("Oracle: Sorry, I cannot retrieve the prediction data. Please try again.")
        except requests.exceptions.ConnectionError:
            print("Oracle: Unable to connect to the server. Please try again and make sure the server is running.")

# ==========================
# Main Entry Point
# ==========================

import multiprocessing
import time
import argparse

# Disable Flask default logging
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vegetable Price Oracle System")
    # Default to mode 3 (server + client auto start); only use mode 1/2 if explicitly specified via --mode
    parser.add_argument("--mode", type=str, choices=["1", "2", "3"], default="3",
                        help="Select mode: 1=Server only, 2=Client only, 3=Both (default)")
    args = parser.parse_args()
    mode = args.mode

    if mode == "1":
        print("🖥️ Starting server only mode...")
        run_server()

    elif mode == "2":
        print("💬 Starting client only mode...")
        run_client()

    elif mode == "3":
        print("🔁 Starting server + client mode (default)...")

        # Launch Flask server in background process
        server_process = multiprocessing.Process(target=run_server)
        server_process.start()

        # Wait for server to finish initializing
        time.sleep(2)

        try:
            run_client()
        finally:
            print("🔌 Shutting down server...")
            server_process.terminate()
            server_process.join()



