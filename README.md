# SMG7914.github.io
# SMG7914.github.io



Nigeria Weather Forecasting Model for Smart Farming



**Project Overview**



This project implements a predictive weather forecasting model specifically designed for smart farming applications in Nigeria. The system uses machine learning to predict weather conditions, temperature, humidity, and wind patterns across different regions of Nigeria, helping farmers make data-driven agricultural decisions.



**Features**



&nbsp;   Multi-output Prediction:



&nbsp;       Weather description classification (Clear, Clouds, etc.)



&nbsp;       Temperature prediction (in Kelvin, converted to Celsius)



&nbsp;       Humidity prediction



&nbsp;       Wind vector calculation



&nbsp;   Regional Analysis: Covers all six geopolitical zones of Nigeria



&nbsp;   Smart Feature Engineering:



&nbsp;       Wind vector calculation from speed and direction



&nbsp;       Day length computation from sunrise/sunset times



&nbsp;       Urban/Rural classification based on population density



&nbsp;       Regional mapping for better geographical context





**Technical Implementation**



Data Preprocessing \& Feature Engineering



&nbsp;   Wind vector feature combining speed and direction



&nbsp;   Regional classification (North-central, North-east, North-west, South-east, South-south, South-west)



&nbsp;   Urban/Rural classification based on population thresholds



&nbsp;   Day length calculation from sunrise and sunset times



&nbsp;   Comprehensive data cleaning and normalization



**Machine Learning Models**



&nbsp;   Random Forest for both classification and regression tasks



&nbsp;   XGBoost implementation (with error handling for missing dependencies)



&nbsp;   Cross-validation and hyperparameter tuning capabilities



&nbsp;   Model evaluation with accuracy and R² scores



**Model Pipeline**



\# Classification: Weather description

\# Regression: Temperature, Humidity, Wind Vector

Features: \['latitude', 'longitude', 'pressure', 'wind\_speed', 

&nbsp;          'wind\_degree', 'day\_length', 'urban\_rural', 'region\_new']





**Data Sources**



    Primary Dataset: nigeria\_cities\_weather\_data.csv



&nbsp;   Features Included:



&nbsp;       Geographical: latitude, longitude



        **Meteorological: temperature, pressure, humidity, wind speed/direction, cloud cover**



        **Temporal: date, sunrise, sunset, timezone**



        **Demographic: population, region, urban/rural classification**





**Installation \& Setup**



Prerequisites

pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit



Optional Dependencies

pip install xgboost  # For XGBoost models



Running the Application

streamlit run app.py



Project Structure

├── nigeria\_cities\_weather\_data.csv    # Raw weather data

├── predictive\_weather\_model.py        # Main modeling script

├── app.py                            # Streamlit web application

├── rf\_classification\_pipeline.pkl    # Saved classification model

├── rf\_regression\_pipeline.pkl        # Saved regression model

├── label\_encoder.pkl                 # Target label encoder

└── README.md                         # Project documentation



**Usage**

Web Application



The Streamlit app provides an intuitive interface for weather predictions:



&nbsp;   Select Geopolitical Zone: Choose from Nigeria's six regions



&nbsp;   Automatic Coordinates: Latitude/longitude auto-populated based on region



&nbsp;   Input Parameters:



&nbsp;       Urban/Rural classification



&nbsp;       Pressure, wind speed, wind degree



&nbsp;       Day length



&nbsp;   Real-time Predictions: Get instant weather forecasts



**Model Training**

\# Train classification model for weather description

clf\_pipeline.fit(X\_train\_class, y\_train\_class)



\# Train regression model for temp, humidity, wind

reg\_pipeline.fit(X\_train\_reg, y\_train\_reg)



**Model Performance**

Evaluation Metrics



&nbsp;   Classification: Accuracy score for weather description prediction



&nbsp;   Regression: R² score for **continuous variables (temperature, humidity, wind** vector)



&nbsp;   Cross-validation: 5-fold CV for robust performance assessment



Feature Importance



&nbsp;   Geographical features (latitude/longitude) show high importance



&nbsp;   Meteorological parameters (pressure, wind) contribute significantly



&nbsp;   Regional and urban/rural context provides valuable contextual information





**Smart Farming Applications**

&nbsp;   1. Crop Planning: Optimal planting times based on weather forecasts



&nbsp;   2. Irrigation Management: Humidity and temperature-based water scheduling



&nbsp;   3. Pest Control: Weather-dependent pest management strategies



&nbsp;   4. Harvest Planning: Weather-optimized harvest timing



&nbsp;   5. Risk Mitigation: Early warnings for adverse weather conditions



**Customization**



Adding New Regions

state\_to\_region = {

&nbsp;   'New\_State': 'Region\_Name',

&nbsp;   # Add new mappings here

}



Model Tuning

param\_grid = {

&nbsp;   'n\_estimators': \[50, 100, 200],

&nbsp;   'max\_depth': \[None, 10, 20, 30],

&nbsp;   # Add more parameters for optimization

}



**Contributing**

We welcome contributions to improve the model:



&nbsp;   Additional weather features



&nbsp;   More sophisticated model architectures



&nbsp;   Enhanced regional mapping



&nbsp;   Integration with real-time weather APIs



**Known Issues \& Solutions**

&nbsp;   XGBoost Dependency: Install via pip install xgboost



&nbsp;   Model File Not Found: Ensure pipelines are saved after training



&nbsp;   Coordinate Accuracy: Regional coordinates are approximations



**License**

This project is intended for educational and research purposes in smart farming applications.



**Acknowledgments**

&nbsp;   Nigerian Meteorological Agency for weather data



&nbsp;   Open-source machine learning communities



&nbsp;   Agricultural research institutions supporting smart farming initiatives



**Note:** This model is designed for educational and research purposes. For actual farming decisions, always consult with agricultural experts and use official weather services.

