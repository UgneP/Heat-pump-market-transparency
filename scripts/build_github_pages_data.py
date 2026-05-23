import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app


OUTPUT = ROOT / "docs" / "data" / "app-data.json"


def clean_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def numeric_grid(series, count=16):
    values = np.linspace(float(series.min()), float(series.max()), count)
    return sorted({round(float(value), 2) for value in values})


def main():
    df = app.load_data()
    model_info = app.train_price_model(df)
    model = model_info["model"]
    training = app.model_training_data(df)

    power_values = numeric_grid(training["Rated Power medium T [kW]"], 13)
    storage_values = sorted({round(float(value), 1) for value in training["Storage Size (L)"].dropna().unique()})
    noise_values = numeric_grid(training["SPL outdoor high Power [dBA]"], 10)
    standby_values = numeric_grid(training["Poff [W]"], 8)
    config_values = app.option_list(training["Config"])
    tank_values = app.option_list(training["Storage"])
    refrigerant_values = app.option_list(training["Refrigerant"])

    scenarios = []
    for config in config_values:
        for tank in tank_values:
            for refrigerant in refrigerant_values:
                for power in power_values:
                    for storage in storage_values:
                        for noise in noise_values:
                            for standby in standby_values:
                                scenarios.append(
                                    {
                                        "Rated Power medium T [kW]": power,
                                        "Storage Size (L)": storage,
                                        "SPL outdoor high Power [dBA]": noise,
                                        "Poff [W]": standby,
                                        "Config": config,
                                        "Storage": tank,
                                        "Refrigerant": refrigerant,
                                    }
                                )

    scenario_df = app.pd.DataFrame(scenarios)
    predictions = model.predict(scenario_df)
    prediction_rows = [
        [
            row["Rated Power medium T [kW]"],
            row["Storage Size (L)"],
            row["SPL outdoor high Power [dBA]"],
            row["Poff [W]"],
            row["Config"],
            row["Storage"],
            row["Refrigerant"],
            round(float(prediction), 0),
        ]
        for row, prediction in zip(scenarios, predictions)
    ]

    table_columns = [
        "Manufacturer display",
        "Model/Type",
        "Titel",
        "Model",
        "Price",
        "Rated Power low T [kW]",
        "SCOP",
        "Configuration",
        "Refrigerant",
        "Website",
        "Date_scraped",
    ]
    dashboard_rows = [
        {column: clean_value(row[column]) for column in table_columns}
        for _, row in df[table_columns].sort_values("Price").iterrows()
    ]

    payload = {
        "generatedFrom": "RandomForest_200t precomputed grid",
        "expectedPriceBand": app.EXPECTED_PRICE_BAND,
        "modelMetrics": {
            "r2": round(float(model_info["r2"]), 3),
            "mae": round(float(model_info["mae"]), 0),
            "mape": round(float(model_info["mape"]), 1),
        },
        "limits": {
            "power": [min(power_values), max(power_values)],
            "storage": [min(storage_values), max(storage_values)],
            "noise": [min(noise_values), max(noise_values)],
            "standby": [min(standby_values), max(standby_values)],
        },
        "options": {
            "config": config_values,
            "tank": tank_values,
            "refrigerant": refrigerant_values,
        },
        "predictionColumns": ["power", "storage", "noise", "standby", "config", "tank", "refrigerant", "prediction"],
        "predictionRows": prediction_rows,
        "dashboardRows": dashboard_rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(prediction_rows):,} predictions and {len(dashboard_rows):,} table rows.")


if __name__ == "__main__":
    main()
