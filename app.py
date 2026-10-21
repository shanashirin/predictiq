
import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your trained model and preprocessing files
model = joblib.load(
    os.path.join(BASE_DIR, "predictive_maintenance_model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "minmax_scaler.pkl")
)

feature_names = joblib.load(
    os.path.join(BASE_DIR, "feature_names.pkl")
)

# Sensor fields displayed on the webpage
SENSOR_FIELDS = [
    {
        "name": "air_temperature",
        "label": "Air Temperature (K)",
        "default": 298.1,
        "min": 250,
        "max": 350,
        "step": 0.1,
    },
    {
        "name": "process_temperature",
        "label": "Process Temperature (K)",
        "default": 308.6,
        "min": 250,
        "max": 400,
        "step": 0.1,
    },
    {
        "name": "rotational_speed",
        "label": "Rotational Speed (rpm)",
        "default": 1551,
        "min": 0,
        "max": 10000,
        "step": 1,
    },
    {
        "name": "torque",
        "label": "Torque (Nm)",
        "default": 42.8,
        "min": 0,
        "max": 500,
        "step": 0.1,
    },
    {
        "name": "tool_wear",
        "label": "Tool Wear (min)",
        "default": 20,
        "min": 0,
        "max": 1000,
        "step": 1,
    },
]


def normalize_name(name):
    """Normalize column names for matching."""
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def prepare_features(form_data):
    """Create input in the exact order used during training."""

    sensor_values = {
        "airtemperaturek": float(form_data["air_temperature"]),
        "processtemperaturek": float(form_data["process_temperature"]),
        "rotationalspeedrpm": float(form_data["rotational_speed"]),
        "torquenm": float(form_data["torque"]),
        "toolwearmin": float(form_data["tool_wear"]),
    }

    # Support common alternative column names
    sensor_values.update({
        "airtemperature": float(form_data["air_temperature"]),
        "processtemperature": float(form_data["process_temperature"]),
        "rotationalspeed": float(form_data["rotational_speed"]),
        "torque": float(form_data["torque"]),
        "toolwear": float(form_data["tool_wear"]),
    })

    names = list(feature_names)

    values = []
    for name in names:
        normalized = normalize_name(name)

        if normalized in sensor_values:
            values.append(sensor_values[normalized])
        else:
            # Unknown/extra training features default to zero.
            # If your model uses categorical or other features,
            # replace this with the correct input values.
            values.append(0.0)

    X = np.asarray(values, dtype=float).reshape(1, -1)

    # Apply the same scaler used during training
    X_scaled = scaler.transform(X)

    return X_scaled


@app.route("/")
def home():
    return render_template(
        "index.html",
        sensor_fields=SENSOR_FIELDS,
        result=None,
        error=None,
        values={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    values = request.form.to_dict()

    try:
        # Validate the five user-entered values
        for field in SENSOR_FIELDS:
            value = float(values[field["name"]])

            if not np.isfinite(value):
                raise ValueError("Please enter valid numeric values.")

            if value < field["min"] or value > field["max"]:
                raise ValueError(
                    f'{field["label"]} must be between '
                    f'{field["min"]} and {field["max"]}.'
                )

        X_scaled = prepare_features(values)

        prediction = int(model.predict(X_scaled)[0])

        if prediction == 1:
            status = "Machine Failure Detected"
            description = (
                "The model predicts a failure. "
                "Inspect the machine and its operating conditions."
            )
            result_class = "danger"
        else:
            status = "No Failure Predicted"
            description = (
                "The model predicts no failure for these input readings. "
                "Continue routine monitoring and maintenance."
            )
            result_class = "success"

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = round(
                    float(probabilities[classes.index(1)]) * 100, 2
                )

        result = {
            "status": status,
            "description": description,
            "result_class": result_class,
            "probability": probability,
        }

        return render_template(
            "index.html",
            sensor_fields=SENSOR_FIELDS,
            result=result,
            error=None,
            values=values,
        )

    except (ValueError, TypeError, KeyError) as exc:
        return render_template(
            "index.html",
            sensor_fields=SENSOR_FIELDS,
            result=None,
            error=str(exc),
            values=values,
        )

    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            sensor_fields=SENSOR_FIELDS,
            result=None,
            error=(
                "Prediction failed. Check the model, scaler, feature "
                "names, and the terminal error."
            ),
            values=values,
        )


if __name__ == "__main__":
    app.run(debug=True)
