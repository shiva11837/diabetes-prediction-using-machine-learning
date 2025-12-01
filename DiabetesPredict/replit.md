# Diabetes Risk Prediction App

## Overview
A comprehensive web application for predicting diabetes risk using multiple machine learning models. Users can input their health metrics and receive an instant risk assessment with probability percentage, risk category classification, and personalized recommendations.

## Current State
Fully functional diabetes prediction application with:
- Patient data input form with 8 health metrics
- Multiple machine learning models (RandomForest, SVM, Neural Network, Logistic Regression, Ensemble)
- Interactive visualizations and model comparison
- Prediction history tracking with PostgreSQL database
- PDF health report generation
- Population statistics benchmarking
- Model retraining with custom datasets
- Educational content about diabetes

## Project Architecture

### Files
- `app.py` - Main Streamlit application with UI, tabs, and visualizations
- `model.py` - Machine learning model training, prediction, and ensemble logic
- `database.py` - PostgreSQL database models and CRUD operations
- `pdf_report.py` - PDF health report generation with recommendations
- `diabetes_model.pkl` - Saved trained models (auto-generated on first run)
- `.streamlit/config.toml` - Streamlit server configuration

### Technology Stack
- **Frontend**: Streamlit
- **ML Framework**: scikit-learn (RandomForest, SVM, MLP Neural Network, Logistic Regression, Voting Ensemble)
- **Data Processing**: pandas, numpy
- **Visualizations**: Plotly
- **Database**: PostgreSQL with SQLAlchemy ORM
- **PDF Generation**: ReportLab

### Features

#### Prediction Tab
- Enter 8 health metrics to get diabetes risk assessment
- Choose from 5 ML models (Ensemble, RandomForest, SVM, NeuralNetwork, LogisticRegression)
- View risk level with color-coded visualization
- Gauge chart and pie chart showing prediction confidence

#### Analytics Tab
- View model accuracy metrics for all 5 models
- Feature importance visualization
- Reference ranges table for health metrics
- Model retraining with CSV upload

#### History Tab
- View prediction history with trend chart
- Track risk level changes over time
- Delete individual predictions

#### Compare Models Tab
- Bar chart comparing predictions from all 5 models
- Model agreement statistics
- Population benchmarking (your metrics vs population average)

#### Generate Report Tab
- Create downloadable PDF health reports
- Personalized recommendations based on health metrics
- Professional formatting with disclaimer

#### Learn More Tab
- Educational content about diabetes types
- Risk factors and prevention tips
- Warning signs and symptoms

### Health Metrics Analyzed
- Pregnancies
- Glucose Level (mg/dL)
- Blood Pressure (mm Hg)
- Skin Thickness (mm)
- Insulin Level (mu U/ml)
- BMI (Body Mass Index, kg/m²)
- Family History Score (Diabetes Pedigree Function)
- Age (years)

### Risk Levels
- **Low** (< 30%): Green indicator
- **Medium** (30-60%): Orange indicator
- **High** (> 60%): Red indicator

### Database Schema

#### prediction_history table
- id: Primary key
- session_id: Browser session identifier
- patient_id: Optional patient reference
- pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age: Health metrics
- prediction: Binary outcome (0/1)
- probability: Risk probability percentage
- risk_level: Low/Medium/High
- model_used: Name of ML model used
- ensemble_prediction: Boolean flag
- created_at: Timestamp
- notes: Optional notes

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Environment Variables
- `DATABASE_URL`: PostgreSQL connection string (auto-configured)

## Recent Changes
- December 1, 2025: Added multiple ML models (SVM, Neural Network, Logistic Regression, Ensemble)
- December 1, 2025: Implemented prediction history tracking with PostgreSQL
- December 1, 2025: Added PDF health report generation
- December 1, 2025: Created population comparison and benchmarking tool
- December 1, 2025: Added model retraining with user-uploaded datasets
- December 1, 2025: Initial implementation with full ML pipeline and Streamlit UI
