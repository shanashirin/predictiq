PredictIQ — AI-Powered Predictive Maintenance System
PredictIQ is a Machine Learning-based predictive maintenance web application that uses machine operating parameters to predict potential machine failure. Built with Python and Flask, it provides a simple web interface where users can enter sensor readings and view the model's prediction.
Goal: Support proactive maintenance decisions by identifying potential machine failure risks from operating data.

Features
- Machine failure prediction using a trained Machine Learning model.
- Sensor input form for air temperature, process temperature, rotational speed, torque, and tool wear.
- Data preprocessing using a saved scaler.
- Web interface built with HTML and CSS and served by Flask.
- Prediction feedback showing the predicted machine condition.
- Failure probability display when the trained model supports predict_proba.
- Input validation to help prevent invalid sensor values.
- Responsive interface designed for desktop and mobile screens.
Tech Stack
- Language: Python
- Machine Learning: scikit-learn-compatible model
- Web framework: Flask
- Data processing: NumPy
- Model and preprocessing file loading: joblib
- Frontend: HTML, CSS, Jinja templates
- Development tools: PyCharm or another Python IDE
Project Structure
predictiq-predictive-maintenance/
├── app.py
├── predictive_maintenance_model.pkl
├── minmax_scaler.pkl
├── feature_names.pkl
├── templates/
│   └── index.html
├── predictive maintenance system.ipynb
├── .gitignore
└── README.md
Your local project may also contain a virtual environment (venv/) and IDE or notebook checkpoint files. These normally should not be committed to Git.
Input Parameters
The web form accepts the following sensor readings:
Parameter	Description	Example
Air Temperature (K)	Air temperature around the machine	298.1
Process Temperature (K)	Machine process temperature	308.6
Rotational Speed (rpm)	Machine rotational speed	1551
Torque (Nm)	Applied machine torque	42.8
Tool Wear (min)	Tool usage or wear time	20


The values above are examples for testing the interface, not guaranteed safe operating conditions. Use the input ranges and feature definitions appropriate to your dataset and trained model.
Getting Started
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/predictiq-predictive-maintenance.git
cd predictiq-predictive-maintenance
Replace YOUR-USERNAME with your GitHub username and use your actual repository URL.
2. Create and activate a virtual environment
Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1
If PowerShell prevents activation, you can use Command Prompt:
venv\Scripts\activate.bat
3. Install dependencies
python -m pip install flask numpy scikit-learn joblib
If you have a requirements.txt file, install dependencies with:
python -m pip install -r requirements.txt
4. Check required files
Make sure the following files are present in the project root:
- predictive_maintenance_model.pkl
- minmax_scaler.pkl
- feature_names.pkl
The application also requires templates/index.html.

The model, scaler, and feature names must be compatible with one another and with the features used during model training.
5. Run the application
python app.py
6. Open the application
Open this address in your browser:

http://127.0.0.1:5000

Enter sensor values and select Predict Machine Condition to view the model output.
How It Works
1. The user enters machine sensor readings in the web form.
2. Flask receives and validates the submitted values.
3. The application arranges input features in the order expected by the trained model.
4. The saved scaler transforms the input data.
5. The trained model generates a prediction.
6. The webpage displays the predicted class and, if supported, the model-estimated failure probability.
Model and Data Compatibility
For valid predictions, ensure that:
- The model was trained using the same input features.
- Feature order and preprocessing match the training pipeline.
- The saved scaler was fitted on the corresponding training features.
- The predicted class labels are mapped correctly to failure and no-failure outcomes.
- Any additional model features, such as categorical or encoded columns, are supplied correctly.
The application should not substitute arbitrary values for features required by the trained model. If the model uses more than the five sensor inputs listed above, update the input form and preprocessing logic accordingly.
Limitations
- Predictions depend on the quality, representativeness, and features of the training data.
- A model prediction is not a guarantee that a machine will or will not fail.
- Probability output is available only if the model supports predict_proba; it should not automatically be interpreted as a calibrated real-world failure probability.
- The current application is a demonstration and should not be used as the sole basis for safety-critical maintenance decisions.
- The Flask development server is intended for local development, not production deployment.
Future Improvements
- Add model evaluation metrics such as precision, recall, F1-score, and a confusion matrix.
- Display prediction history and sensor trends.
- Add interactive charts and a maintenance dashboard.
- Support CSV uploads for batch predictions.
- Add authentication and user roles if required.
- Deploy with a production-ready WSGI server.
- Add automated tests and input-schema validation.
Contributing
Contributions and suggestions are welcome.
1. Fork the repository.
2. Create a branch for your change.
3. Commit your changes.
4. Open a pull request describing the improvement.
License
No license has been specified yet. Add a LICENSE file before distributing the project if you want to define reuse and contribution permissions.
PredictIQ — Predict Failures. Prevent Downtime. Improve Reliability.
