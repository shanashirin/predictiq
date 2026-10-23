import io
import os
import secrets
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("PREDICTIQ_SECRET_KEY", "change-this-demo-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB CSV upload limit

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


def find_artifact(*names):
    """Find a saved artifact either beside app.py or in ./artifacts."""
    for folder in (BASE_DIR, ARTIFACT_DIR):
        for name in names:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


MODEL_PATH = find_artifact("predictive_maintenance_model.pkl", "predictive_maintenance_model.joblib")
SCALER_PATH = find_artifact("minmax_scaler.pkl", "minmax_scaler.joblib")
FEATURES_PATH = find_artifact("feature_names.pkl", "feature_columns.joblib", "feature_names.joblib")
COMPARISON_PATH = find_artifact("model_comparison.csv")

MODEL = SCALER = FEATURE_NAMES = None
ARTIFACT_ERROR = None
try:
    missing = [label for label, path in (("trained model", MODEL_PATH), ("scaler", SCALER_PATH), ("feature names", FEATURES_PATH)) if path is None]
    if missing:
        raise FileNotFoundError(
            "Missing " + ", ".join(missing) + ". Copy the three saved artifacts from your notebook into this app's folder or its artifacts/ folder."
        )
    MODEL = joblib.load(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
    FEATURE_NAMES = list(joblib.load(FEATURES_PATH))
except Exception as exc:
    ARTIFACT_ERROR = str(exc)

SENSOR_FIELDS = [
    {"name": "air_temperature", "label": "Air Temperature (K)", "default": 298.1, "min": 250, "max": 350, "step": 0.1},
    {"name": "process_temperature", "label": "Process Temperature (K)", "default": 308.6, "min": 250, "max": 400, "step": 0.1},
    {"name": "rotational_speed", "label": "Rotational Speed (rpm)", "default": 1551, "min": 0, "max": 10000, "step": 1},
    {"name": "torque", "label": "Torque (Nm)", "default": 42.8, "min": 0, "max": 500, "step": 0.1},
    {"name": "tool_wear", "label": "Tool Wear (min)", "default": 20, "min": 0, "max": 1000, "step": 1},
]

# In-memory download cache is sufficient for a classroom/demo app. It clears on restart.
BATCH_DOWNLOADS = {}
MAX_BATCH_ROWS = 10000

# Stores the latest batch prediction summary in memory.
# This resets when the Flask application restarts.
LATEST_BATCH_STATS = None


def normalize_name(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def value_from_mapping(mapping, aliases, default=None):
    """Look up a field in a form/CSV row regardless of spaces, case, or units punctuation."""
    normalized = {normalize_name(k): v for k, v in mapping.items()}
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized and pd.notna(normalized[key]) and str(normalized[key]).strip() != "":
            return normalized[key]
    return default


def parse_machine_type(raw_type):
    if raw_type is None or str(raw_type).strip() == "":
        return "H"  # AI4I reference category when Type is omitted
    machine_type = str(raw_type).strip().upper()
    if machine_type not in {"L", "M", "H"}:
        raise ValueError("Machine type must be L, M, or H.")
    return machine_type


def build_feature_row(row):
    """Build one row in the exact feature order used when the model was trained."""
    if FEATURE_NAMES is None:
        raise RuntimeError(ARTIFACT_ERROR or "Model artifacts could not be loaded.")

    sensor_aliases = {
        "airtemperaturek": ("air_temperature", "Air temperature [K]", "Air temperature", "airtemperaturek"),
        "airtemperature": ("air_temperature", "Air temperature [K]", "Air temperature", "airtemperaturek"),
        "processtemperaturek": ("process_temperature", "Process temperature [K]", "Process temperature", "processtemperaturek"),
        "processtemperature": ("process_temperature", "Process temperature [K]", "Process temperature", "processtemperaturek"),
        "rotationalspeedrpm": ("rotational_speed", "Rotational speed [rpm]", "Rotational speed", "rotationalspeedrpm"),
        "rotationalspeed": ("rotational_speed", "Rotational speed [rpm]", "Rotational speed", "rotationalspeedrpm"),
        "torquenm": ("torque", "Torque [Nm]", "Torque", "torquenm"),
        "torque": ("torque", "Torque [Nm]", "Torque", "torquenm"),
        "toolwearmin": ("tool_wear", "Tool wear [min]", "Tool wear", "toolwearmin"),
        "toolwear": ("tool_wear", "Tool wear [min]", "Tool wear", "toolwearmin"),
    }
    machine_type = parse_machine_type(value_from_mapping(row, ("Type", "machine_type", "Machine Type"), default="H"))
    feature_values = []

    for feature in FEATURE_NAMES:
        normalized = normalize_name(feature)
        if normalized in sensor_aliases:
            raw_value = value_from_mapping(row, sensor_aliases[normalized])
            if raw_value is None:
                raise ValueError(f"Missing required sensor value for '{feature}'.")
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                raise ValueError(f"'{feature}' must be numeric.")
            if not np.isfinite(numeric_value):
                raise ValueError(f"'{feature}' must be a finite number.")
            feature_values.append(numeric_value)
        elif normalized == "typel":
            feature_values.append(1.0 if machine_type == "L" else 0.0)
        elif normalized == "typem":
            feature_values.append(1.0 if machine_type == "M" else 0.0)
        elif normalized == "typeh":
            feature_values.append(1.0 if machine_type == "H" else 0.0)
        else:
            raise ValueError(
                f"The saved model expects feature '{feature}', but the app has no mapping for it. "
                "Update the input mapping to match the notebook's training features."
            )

    return pd.DataFrame([feature_values], columns=FEATURE_NAMES)


def validate_form_values(form_values):
    cleaned = dict(form_values)
    for field in SENSOR_FIELDS:
        raw = cleaned.get(field["name"])
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"{field['label']} is required.")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{field['label']} must be numeric.")
        if not np.isfinite(number):
            raise ValueError(f"{field['label']} must be a finite number.")
        if number < field["min"] or number > field["max"]:
            raise ValueError(f"{field['label']} must be between {field['min']} and {field['max']}.")
        cleaned[field["name"]] = number
    # parse_machine_type(cleaned.get("machine_type", "H"))
    return cleaned


def predict_row(row):
    feature_frame = build_feature_row(row)
    scaled = SCALER.transform(feature_frame)
    predicted_class = int(MODEL.predict(scaled)[0])
    failure_probability = None
    if hasattr(MODEL, "predict_proba"):
        probabilities = MODEL.predict_proba(scaled)[0]
        classes = list(MODEL.classes_)
        if 1 in classes:
            failure_probability = round(float(probabilities[classes.index(1)]) * 100, 2)

    if predicted_class == 1:
        status = "Machine Failure Predicted"
        result_class = "danger"
    else:
        status = "No Failure Predicted"
        result_class = "success"

    # Display-only probability bands; validate thresholds against operational data before real use.
    if failure_probability is None:
        risk = "Unavailable"
    elif failure_probability < 30:
        risk = "Low"
    elif failure_probability < 70:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "prediction": predicted_class,
        "status": status,
        "result_class": result_class,
        "probability": failure_probability,
        "risk": risk,
    }



def get_model_comparison():
    """Load saved model evaluation results."""
    comparison_path = find_artifact("model_comparison.csv")

    if comparison_path is None:
        return [], (
            "Model evaluation results are unavailable. "
            "Generate model_comparison.csv from your notebook "
            "using held-out test data."
        )

    try:
        comparison_df = pd.read_csv(comparison_path)

        if comparison_df.empty:
            return [], "The model comparison file contains no evaluation rows."

        # Avoid invalid JSON-like values in template rendering.
        comparison_df = comparison_df.replace(
            [np.inf, -np.inf], np.nan
        )

        comparison_df = comparison_df.astype(object).where(
            pd.notna(comparison_df), None
        )

        return comparison_df.to_dict(orient="records"), None

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        app.logger.exception("Unable to load model comparison data")
        return [], f"Could not read model evaluation results: {exc}"




def get_dataset_stats():
    """Load actual historical statistics from the AI4I dataset."""
    candidates = [
        BASE_DIR / "ai4i2020.csv",
        BASE_DIR / "AI4I 2020 Predictive Maintenance Dataset.csv",
        ARTIFACT_DIR / "ai4i2020.csv",
        ARTIFACT_DIR / "AI4I 2020 Predictive Maintenance Dataset.csv",
    ]

    for path in candidates:
        if not path.is_file():
            continue

        try:
            data = pd.read_csv(path)

            target = next(
                (
                    col for col in data.columns
                    if normalize_name(col) == "machinefailure"
                ),
                None,
            )

            if target is None:
                return {
                    "records": len(data),
                    "failures": None,
                    "failure_rate": None,
                    "source": path.name,
                    "error": "Dataset has no Machine failure target column.",
                }

            labels = pd.to_numeric(data[target], errors="coerce")
            valid_labels = labels.dropna()

            if not valid_labels.isin([0, 1]).all():
                return {
                    "records": len(data),
                    "failures": None,
                    "failure_rate": None,
                    "source": path.name,
                    "error": "Machine failure labels must be 0 or 1.",
                }

            failures = int((valid_labels == 1).sum())
            total_labelled = len(valid_labels)

            return {
                "records": len(data),
                "failures": failures,
                "failure_rate": (
                    round(failures / total_labelled * 100, 2)
                    if total_labelled else None
                ),
                "source": path.name,
                "error": None,
            }

        except (OSError, ValueError, pd.errors.ParserError) as exc:
            app.logger.warning(
                "Could not read dataset %s: %s", path, exc
            )

    return {
        "records": None,
        "failures": None,
        "failure_rate": None,
        "source": None,
        "error": "AI4I dataset CSV was not found or could not be read.",
    }


@app.context_processor
def inject_artifact_status():
    return {"artifact_error": ARTIFACT_ERROR}


@app.route("/")
def landing():
    return render_template("home.html")


@app.route("/predict-page")
def home():
    return render_template(
        "index.html",
        sensor_fields=SENSOR_FIELDS,
        result=None,
        error=None,
        values={}
    )


@app.route("/predict", methods=["POST"])
def predict():
    values = request.form.to_dict()
    try:
        if ARTIFACT_ERROR:
            raise RuntimeError(ARTIFACT_ERROR)
        values = validate_form_values(values)
        result = predict_row(values)
        result["description"] = (
            "Review the machine and its operating conditions." if result["prediction"] == 1
            else "Continue routine monitoring and follow your maintenance schedule."
        )
        return render_template("index.html", sensor_fields=SENSOR_FIELDS, result=result, error=None, values=values)
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        return render_template("index.html", sensor_fields=SENSOR_FIELDS, result=None, error=str(exc), values=values), 400 if isinstance(exc, ValueError) else 500
    except Exception:
        app.logger.exception("Single prediction failed")
        return render_template("index.html", sensor_fields=SENSOR_FIELDS, result=None, error="Prediction failed. Check the model artifacts and server log.", values=values), 500



@app.route("/dashboard")
def dashboard():
    comparison, comparison_error = get_model_comparison()
    stats = get_dataset_stats()

    return render_template(
        "dashboard.html",
        comparison=comparison,
        comparison_error=comparison_error,
        stats=stats,
        latest_batch=LATEST_BATCH_STATS,
        model_name=(
            type(MODEL).__name__
            if MODEL is not None
            else "Not loaded"
        ),
        feature_count=(
            len(FEATURE_NAMES)
            if FEATURE_NAMES is not None
            else 0
        ),
        model_ready=(
            MODEL is not None
            and SCALER is not None
            and FEATURE_NAMES is not None
        ),
    )


@app.route("/batch", methods=["GET"])
def batch_upload():
    return render_template("batch.html", error=None)


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    if ARTIFACT_ERROR:
        return render_template("batch.html", error=ARTIFACT_ERROR), 500
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return render_template("batch.html", error="Choose a CSV file to upload."), 400
    if Path(secure_filename(upload.filename)).suffix.lower() != ".csv":
        return render_template("batch.html", error="Only .csv files are supported."), 400
    try:
        raw = upload.read()
        if not raw:
            raise ValueError("The uploaded CSV is empty.")
        frame = pd.read_csv(io.BytesIO(raw))
        if frame.empty:
            raise ValueError("The uploaded CSV has no data rows.")
        if len(frame) > MAX_BATCH_ROWS:
            raise ValueError(f"Upload at most {MAX_BATCH_ROWS} rows at a time.")
        normalized_columns = {normalize_name(col) for col in frame.columns}
        required_groups = [
            {normalize_name("Air temperature [K]"), normalize_name("air_temperature"), normalize_name("airtemperature")},
            {normalize_name("Process temperature [K]"), normalize_name("process_temperature"), normalize_name("processtemperature")},
            {normalize_name("Rotational speed [rpm]"), normalize_name("rotational_speed"), normalize_name("rotationalspeed")},
            {normalize_name("Torque [Nm]"), normalize_name("torque")},
            {normalize_name("Tool wear [min]"), normalize_name("tool_wear"), normalize_name("toolwear")},
        ]
        if any(not (group & normalized_columns) for group in required_groups):
            raise ValueError("CSV must include Air temperature, Process temperature, Rotational speed, Torque, and Tool wear columns. A Type column (L/M/H) is optional; if omitted, Type H is assumed.")


        output_rows = []
        errors = []
        feature_frames = []
        valid_records = []

        # Validate records and prepare model inputs.
        for index, (_, row) in enumerate(frame.iterrows(), start=2):
            row_data = row.to_dict()

            try:
                for field in SENSOR_FIELDS:
                    aliases = {
                        "air_temperature": (
                            "air_temperature", "Air temperature [K]",
                            "Air temperature", "airtemperature"
                        ),
                        "process_temperature": (
                            "process_temperature", "Process temperature [K]",
                            "Process temperature", "processtemperature"
                        ),
                        "rotational_speed": (
                            "rotational_speed", "Rotational speed [rpm]",
                            "Rotational speed", "rotationalspeed"
                        ),
                        "torque": ("torque", "Torque [Nm]"),
                        "tool_wear": (
                            "tool_wear", "Tool wear [min]",
                            "Tool wear", "toolwear"
                        ),
                    }[field["name"]]

                    raw_value = value_from_mapping(row_data, aliases)

                    if raw_value is None:
                        raise ValueError(f"Missing {field['label']}.")

                    number = float(raw_value)

                    if (
                        not np.isfinite(number)
                        or number < field["min"]
                        or number > field["max"]
                    ):
                        raise ValueError(
                            f"{field['label']} must be finite and "
                            f"between {field['min']} and {field['max']}."
                        )

                machine_type = parse_machine_type(
                    value_from_mapping(
                        row_data,
                        ("Type", "machine_type", "Machine Type"),
                        default="H"
                    )
                )

                row_data["Type"] = machine_type

                # Prepare features without predicting yet.
                feature_frame = build_feature_row(row_data)
                feature_frames.append(feature_frame)

                valid_records.append({
                    "record": index - 1,
                    "machine_type": machine_type,
                    "csv_row": index,
                })

            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f"CSV row {index}: {exc}")

        if not feature_frames:
            raise ValueError(
                "No rows could be predicted. "
                + (errors[0] if errors else "Check your CSV values.")
            )

        # Combine all valid records into one feature matrix.
        all_features = pd.concat(feature_frames, ignore_index=True)

        # Scale and predict in batches rather than once per row.
        scaled_features = SCALER.transform(all_features)
        predictions = MODEL.predict(scaled_features)

        probabilities = None
        positive_class_index = None

        if hasattr(MODEL, "predict_proba"):
            class_labels = list(MODEL.classes_)

            if 1 in class_labels:
                positive_class_index = class_labels.index(1)
                probabilities = MODEL.predict_proba(scaled_features)

        # Format the results.
        for i, record in enumerate(valid_records):
            predicted_class = int(predictions[i])

            if probabilities is not None:
                probability = round(
                    float(probabilities[i, positive_class_index]) * 100,
                    2
                )
                if probability < 30:
                    risk = "Low"
                elif probability < 70:
                    risk = "Medium"
                else:
                    risk = "High"
            else:
                probability = None
                risk = "Unavailable"

            output_rows.append({
                "Record": record["record"],
                "Machine Type": record["machine_type"],
                "Predicted Class": predicted_class,
                "Prediction": (
                    "Machine Failure Predicted"
                    if predicted_class == 1
                    else "No Failure Predicted"
                ),
                "Failure Probability (%)": probability,
                "Risk Category": risk,
            })

        # Calculate summary from the actual batch predictions.
        total_records = len(output_rows)

        predicted_failures = sum(
            1 for row in output_rows
            if row["Predicted Class"] == 1
        )

        predicted_no_failures = total_records - predicted_failures

        predicted_failure_rate = (
            round((predicted_failures / total_records) * 100, 2)
            if total_records > 0
            else 0
        )

        # Save the latest summary for the dashboard.
        global LATEST_BATCH_STATS

        LATEST_BATCH_STATS = {
            "total_records": total_records,
            "predicted_failures": predicted_failures,
            "predicted_no_failures": predicted_no_failures,
            "predicted_failure_rate": predicted_failure_rate,
        }

        results_frame = pd.DataFrame(output_rows)

        token = secrets.token_urlsafe(16)
        BATCH_DOWNLOADS[token] = results_frame.to_csv(index=False)
        # Bound in-memory demo cache size.
        if len(BATCH_DOWNLOADS) > 10:
            BATCH_DOWNLOADS.pop(next(iter(BATCH_DOWNLOADS)))

        return render_template(
            "results.html",
            results=output_rows,
            download_token=token,
            error=None,
        )

    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return render_template("batch.html", error=str(exc)), 400
    except Exception:
        app.logger.exception("Batch prediction failed")
        return render_template("batch.html", error="Batch prediction failed. Check the CSV columns and server log."), 500


@app.route("/download-results/<token>")
def download_results(token):
    csv_text = BATCH_DOWNLOADS.get(token)
    if csv_text is None:
        abort(404, description="This results download has expired. Upload the CSV again to regenerate it.")
    return send_file(
        io.BytesIO(csv_text.encode("utf-8-sig")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictiq_batch_results.csv",
    )


@app.errorhandler(413)
def upload_too_large(_error):
    return render_template("batch.html", error="File is too large. Maximum upload size is 10 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
