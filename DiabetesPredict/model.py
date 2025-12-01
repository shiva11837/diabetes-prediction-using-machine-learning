import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

def create_diabetes_dataset():
    """
    Create a synthetic dataset based on the Pima Indians Diabetes Dataset structure.
    Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    """
    np.random.seed(42)
    n_samples = 768
    
    pregnancies = np.random.randint(0, 17, n_samples)
    glucose = np.random.normal(120, 32, n_samples).clip(0, 200)
    blood_pressure = np.random.normal(69, 19, n_samples).clip(0, 122)
    skin_thickness = np.random.normal(20, 16, n_samples).clip(0, 99)
    insulin = np.random.normal(80, 115, n_samples).clip(0, 846)
    bmi = np.random.normal(32, 8, n_samples).clip(0, 67)
    diabetes_pedigree = np.random.exponential(0.47, n_samples).clip(0.078, 2.42)
    age = np.random.randint(21, 81, n_samples)
    
    risk_score = (
        0.02 * pregnancies +
        0.015 * glucose +
        0.005 * blood_pressure +
        0.003 * skin_thickness +
        0.002 * insulin +
        0.03 * bmi +
        0.5 * diabetes_pedigree +
        0.02 * age
    )
    
    threshold = np.percentile(risk_score, 65)
    noise = np.random.normal(0, 0.5, n_samples)
    outcome = (risk_score + noise > threshold).astype(int)
    
    data = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age,
        'Outcome': outcome
    })
    
    return data

def train_all_models():
    """Train all models including ensemble."""
    data = create_diabetes_dataset()
    
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_accuracy = rf_model.score(X_test_scaled, y_test)
    
    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train_scaled, y_train)
    svm_accuracy = svm_model.score(X_test_scaled, y_test)
    
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    nn_model.fit(X_train_scaled, y_train)
    nn_accuracy = nn_model.score(X_test_scaled, y_test)
    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_accuracy = lr_model.score(X_test_scaled, y_test)
    
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('svm', svm_model),
            ('nn', nn_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train_scaled, y_train)
    ensemble_accuracy = ensemble_model.score(X_test_scaled, y_test)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    model_data = {
        'models': {
            'RandomForest': rf_model,
            'SVM': svm_model,
            'NeuralNetwork': nn_model,
            'LogisticRegression': lr_model,
            'Ensemble': ensemble_model
        },
        'accuracies': {
            'RandomForest': rf_accuracy,
            'SVM': svm_accuracy,
            'NeuralNetwork': nn_accuracy,
            'LogisticRegression': lr_accuracy,
            'Ensemble': ensemble_accuracy
        },
        'scaler': scaler,
        'feature_names': list(X.columns),
        'feature_importance': feature_importance,
        'accuracy': ensemble_accuracy,
        'model': rf_model
    }
    
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("All models trained and saved successfully!")
    print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
    print(f"SVM Accuracy: {svm_accuracy:.2%}")
    print(f"Neural Network Accuracy: {nn_accuracy:.2%}")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")
    print(f"Ensemble Accuracy: {ensemble_accuracy:.2%}")
    
    return model_data

def train_model():
    """Train and save all models (wrapper for compatibility)."""
    return train_all_models()

def load_model():
    """Load the trained models, training if necessary."""
    if not os.path.exists('diabetes_model.pkl'):
        return train_all_models()
    
    with open('diabetes_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    if 'models' not in model_data:
        return train_all_models()
    
    return model_data

def predict_diabetes(patient_data, model_data, model_name='RandomForest'):
    """
    Predict diabetes risk for a patient using specified model.
    
    Args:
        patient_data: dict with keys matching feature names
        model_data: loaded model data from load_model()
        model_name: which model to use ('RandomForest', 'SVM', 'NeuralNetwork', 'LogisticRegression', 'Ensemble')
    
    Returns:
        dict with prediction, probability, and risk level
    """
    if 'models' in model_data:
        model = model_data['models'].get(model_name, model_data['models']['RandomForest'])
    else:
        model = model_data['model']
    
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    input_df = pd.DataFrame([patient_data])[feature_names]
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    diabetes_prob = probability[1] * 100
    
    if diabetes_prob < 30:
        risk_level = "Low"
        risk_color = "green"
    elif diabetes_prob < 60:
        risk_level = "Medium"
        risk_color = "orange"
    else:
        risk_level = "High"
        risk_color = "red"
    
    return {
        'prediction': int(prediction),
        'probability': diabetes_prob,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'no_diabetes_prob': probability[0] * 100,
        'diabetes_prob': probability[1] * 100,
        'model_used': model_name
    }

def predict_with_all_models(patient_data, model_data):
    """
    Get predictions from all models for comparison.
    
    Returns:
        dict with predictions from each model
    """
    if 'models' not in model_data:
        result = predict_diabetes(patient_data, model_data)
        return {'RandomForest': result}
    
    results = {}
    for model_name in model_data['models'].keys():
        results[model_name] = predict_diabetes(patient_data, model_data, model_name)
    
    return results

def retrain_with_data(df, target_column='Outcome'):
    """
    Retrain models with user-provided data.
    
    Args:
        df: pandas DataFrame with features and target
        target_column: name of the target column
    
    Returns:
        model_data dict with trained models
    """
    required_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df[required_features]
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_accuracy = rf_model.score(X_test_scaled, y_test)
    
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_accuracy = svm_model.score(X_test_scaled, y_test)
    
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    nn_model.fit(X_train_scaled, y_train)
    nn_accuracy = nn_model.score(X_test_scaled, y_test)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_accuracy = lr_model.score(X_test_scaled, y_test)
    
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('svm', svm_model), ('nn', nn_model), ('lr', lr_model)],
        voting='soft'
    )
    ensemble_model.fit(X_train_scaled, y_train)
    ensemble_accuracy = ensemble_model.score(X_test_scaled, y_test)
    
    feature_importance = pd.DataFrame({
        'feature': required_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    model_data = {
        'models': {
            'RandomForest': rf_model,
            'SVM': svm_model,
            'NeuralNetwork': nn_model,
            'LogisticRegression': lr_model,
            'Ensemble': ensemble_model
        },
        'accuracies': {
            'RandomForest': rf_accuracy,
            'SVM': svm_accuracy,
            'NeuralNetwork': nn_accuracy,
            'LogisticRegression': lr_accuracy,
            'Ensemble': ensemble_accuracy
        },
        'scaler': scaler,
        'feature_names': required_features,
        'feature_importance': feature_importance,
        'accuracy': ensemble_accuracy,
        'model': rf_model,
        'training_samples': len(df),
        'retrained': True
    }
    
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

if __name__ == "__main__":
    train_all_models()
