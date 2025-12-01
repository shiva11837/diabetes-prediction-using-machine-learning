import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import load_model, predict_diabetes, predict_with_all_models, retrain_with_data
from database import init_db, save_prediction, get_prediction_history, get_population_statistics, delete_prediction, DB_AVAILABLE
from pdf_report import create_health_report, generate_report_filename
import uuid
from datetime import datetime

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fa;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
    }
    .risk-high {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        color: #1e3a5f;
    }
    .info-box h4 {
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .info-box p, .info-box ol, .info-box li {
        color: #2d3748;
    }
    .history-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        color: #1e3a5f;
    }
    .comparison-better {
        color: #28a745;
        font-weight: bold;
    }
    .comparison-worse {
        color: #dc3545;
        font-weight: bold;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1e3a5f !important;
    }
    p, span, label, .stMarkdown {
        color: #2d3748;
    }
    .stSelectbox label, .stNumberInput label {
        color: #1e3a5f !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    """Load and cache the ML model."""
    return load_model()

def create_gauge_chart(probability, risk_level):
    """Create a gauge chart for diabetes risk visualization."""
    if risk_level == "Low":
        color = "#38ef7d"
    elif risk_level == "Medium":
        color = "#f5576c"
    else:
        color = "#eb3349"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Diabetes Risk Score", 'font': {'size': 24, 'color': '#1e3a5f'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#1e3a5f'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1e3a5f"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 60], 'color': '#fff3e0'},
                {'range': [60, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "#1e3a5f", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1e3a5f"}
    )
    
    return fig

def create_feature_importance_chart(model_data):
    """Create a horizontal bar chart for feature importance."""
    importance_df = model_data['feature_importance'].copy()
    importance_df['importance_pct'] = importance_df['importance'] * 100
    
    feature_labels = {
        'Pregnancies': 'Pregnancies',
        'Glucose': 'Glucose Level',
        'BloodPressure': 'Blood Pressure',
        'SkinThickness': 'Skin Thickness',
        'Insulin': 'Insulin Level',
        'BMI': 'Body Mass Index (BMI)',
        'DiabetesPedigreeFunction': 'Family History Score',
        'Age': 'Age'
    }
    importance_df['feature_label'] = importance_df['feature'].map(feature_labels)
    
    fig = px.bar(
        importance_df,
        x='importance_pct',
        y='feature_label',
        orientation='h',
        title='Feature Importance in Prediction',
        labels={'importance_pct': 'Importance (%)', 'feature_label': 'Health Factor'},
        color='importance_pct',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e3a5f'}
    )
    
    return fig

def create_probability_pie(result):
    """Create a pie chart showing prediction probabilities."""
    fig = go.Figure(data=[go.Pie(
        labels=['No Diabetes', 'Diabetes'],
        values=[result['no_diabetes_prob'], result['diabetes_prob']],
        hole=0.4,
        marker_colors=['#38ef7d', '#eb3349'],
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title={'text': 'Prediction Confidence', 'font': {'size': 18, 'color': '#1e3a5f'}},
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e3a5f'},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_model_comparison_chart(all_results):
    """Create a bar chart comparing predictions from all models."""
    models = list(all_results.keys())
    probabilities = [all_results[m]['probability'] for m in models]
    
    colors = []
    for m in models:
        if all_results[m]['risk_level'] == 'Low':
            colors.append('#38ef7d')
        elif all_results[m]['risk_level'] == 'Medium':
            colors.append('#f5576c')
        else:
            colors.append('#eb3349')
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=probabilities,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probabilities],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title={'text': 'Model Prediction Comparison', 'font': {'size': 18, 'color': '#1e3a5f'}},
        xaxis_title="Model",
        yaxis_title="Diabetes Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e3a5f'}
    )
    
    return fig

def create_history_trend_chart(history):
    """Create a line chart showing prediction trends over time."""
    if not history:
        return None
    
    dates = [h.created_at for h in history]
    probabilities = [h.probability for h in history]
    risk_levels = [h.risk_level for h in history]
    
    colors = ['#38ef7d' if r == 'Low' else '#f5576c' if r == 'Medium' else '#eb3349' for r in risk_levels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=probabilities,
        mode='lines+markers',
        name='Risk Probability',
        line=dict(color='#1e3a5f', width=2),
        marker=dict(size=10, color=colors)
    ))
    
    fig.add_hline(y=30, line_dash="dash", line_color="#28a745", annotation_text="Low Risk Threshold")
    fig.add_hline(y=60, line_dash="dash", line_color="#dc3545", annotation_text="High Risk Threshold")
    
    fig.update_layout(
        title={'text': 'Your Risk Trend Over Time', 'font': {'size': 18, 'color': '#1e3a5f'}},
        xaxis_title="Date",
        yaxis_title="Diabetes Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e3a5f'}
    )
    
    return fig

def create_population_comparison_chart(patient_data, pop_stats):
    """Create a comparison chart between patient metrics and population averages."""
    metrics = ['Glucose', 'BMI', 'Blood Pressure', 'Age']
    patient_values = [
        patient_data['Glucose'],
        patient_data['BMI'],
        patient_data['BloodPressure'],
        patient_data['Age']
    ]
    pop_values = [
        pop_stats['avg_glucose'],
        pop_stats['avg_bmi'],
        pop_stats['avg_blood_pressure'],
        pop_stats['avg_age']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Your Values',
        x=metrics,
        y=patient_values,
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Bar(
        name='Population Average',
        x=metrics,
        y=pop_values,
        marker_color='#e0e0e0'
    ))
    
    fig.update_layout(
        title={'text': 'Your Metrics vs Population Average', 'font': {'size': 18, 'color': '#1e3a5f'}},
        barmode='group',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e3a5f'},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Health Assessment Tool with Multiple ML Models</p>', unsafe_allow_html=True)
    
    model_data = get_model()
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Prediction", 
        "📊 Analytics", 
        "📈 History",
        "🔬 Compare Models",
        "📄 Generate Report",
        "📚 Learn More"
    ])
    
    with tab1:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("### Patient Information")
            st.markdown("Enter your health metrics below for diabetes risk assessment.")
            
            model_choice = st.selectbox(
                "Select Prediction Model",
                options=['Ensemble', 'RandomForest', 'SVM', 'NeuralNetwork', 'LogisticRegression'],
                help="Choose which machine learning model to use for prediction"
            )
            
            with st.form("prediction_form"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    pregnancies = st.number_input(
                        "Number of Pregnancies",
                        min_value=0, max_value=20, value=1,
                        help="Number of times pregnant (enter 0 if not applicable)"
                    )
                    
                    glucose = st.number_input(
                        "Glucose Level (mg/dL)",
                        min_value=0, max_value=300, value=120,
                        help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test"
                    )
                    
                    blood_pressure = st.number_input(
                        "Blood Pressure (mm Hg)",
                        min_value=0, max_value=200, value=70,
                        help="Diastolic blood pressure"
                    )
                    
                    skin_thickness = st.number_input(
                        "Skin Thickness (mm)",
                        min_value=0, max_value=100, value=20,
                        help="Triceps skin fold thickness"
                    )
                
                with col_b:
                    insulin = st.number_input(
                        "Insulin Level (mu U/ml)",
                        min_value=0, max_value=900, value=80,
                        help="2-Hour serum insulin"
                    )
                    
                    bmi = st.number_input(
                        "BMI (kg/m²)",
                        min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                        help="Body mass index (weight in kg / height in m²)"
                    )
                    
                    diabetes_pedigree = st.number_input(
                        "Family History Score",
                        min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                        help="Diabetes pedigree function (family history influence)"
                    )
                    
                    age = st.number_input(
                        "Age (years)",
                        min_value=1, max_value=120, value=30,
                        help="Patient's age in years"
                    )
                
                submit_button = st.form_submit_button("🔍 Analyze Risk", use_container_width=True)
        
        with col2:
            if submit_button:
                patient_data = {
                    'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'BloodPressure': blood_pressure,
                    'SkinThickness': skin_thickness,
                    'Insulin': insulin,
                    'BMI': bmi,
                    'DiabetesPedigreeFunction': diabetes_pedigree,
                    'Age': age
                }
                
                result = predict_diabetes(patient_data, model_data, model_choice)
                
                try:
                    save_prediction(
                        st.session_state.session_id,
                        patient_data,
                        result,
                        model_used=model_choice,
                        ensemble=(model_choice == 'Ensemble')
                    )
                except Exception as e:
                    st.warning(f"Could not save to history: {e}")
                
                st.session_state.last_prediction = {
                    'patient_data': patient_data,
                    'result': result
                }
                
                st.markdown("### Prediction Results")
                
                risk_class = f"risk-{result['risk_level'].lower()}"
                st.markdown(f"""
                <div class="{risk_class}">
                    <h2 style="margin: 0; font-size: 1.5rem;">Risk Level: {result['risk_level']}</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: bold;">{result['probability']:.1f}%</p>
                    <p style="margin: 0;">probability of diabetes (using {model_choice})</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(create_gauge_chart(result['probability'], result['risk_level']), use_container_width=True)
                
                st.plotly_chart(create_probability_pie(result), use_container_width=True)
                
                if result['risk_level'] == "Low":
                    st.success("✅ Your risk factors indicate a low probability of diabetes. Continue maintaining a healthy lifestyle!")
                elif result['risk_level'] == "Medium":
                    st.warning("⚠️ Your risk factors show moderate concern. Consider consulting a healthcare provider for further evaluation.")
                else:
                    st.error("🚨 Your risk factors indicate elevated concern. Please consult a healthcare professional for proper diagnosis and guidance.")
            else:
                st.markdown("### Results will appear here")
                st.info("👈 Fill in your health information and click 'Analyze Risk' to get your diabetes risk prediction.")
                
                st.markdown("""
                <div class="info-box">
                    <h4>How it works:</h4>
                    <ol>
                        <li>Enter your health metrics in the form</li>
                        <li>Choose your preferred ML model</li>
                        <li>Our AI analyzes your data</li>
                        <li>Receive instant risk assessment</li>
                        <li>Download a detailed PDF report</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Model Analytics & Feature Importance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'accuracies' in model_data:
                st.metric(
                    label="Ensemble Accuracy",
                    value=f"{model_data['accuracies'].get('Ensemble', model_data['accuracy']):.1%}",
                    delta="Best Combined"
                )
            else:
                st.metric(label="Model Accuracy", value=f"{model_data['accuracy']:.1%}")
        
        with col2:
            st.metric(
                label="Features Analyzed",
                value="8",
                delta="Health Factors"
            )
        
        with col3:
            st.metric(
                label="Models Available",
                value="5",
                delta="ML Algorithms"
            )
        
        with col4:
            training_samples = model_data.get('training_samples', 768)
            st.metric(
                label="Training Samples",
                value=str(training_samples),
                delta="Data Points"
            )
        
        st.markdown("---")
        
        if 'accuracies' in model_data:
            st.markdown("### Individual Model Accuracy")
            accuracy_df = pd.DataFrame([
                {'Model': name, 'Accuracy': f"{acc:.1%}"} 
                for name, acc in model_data['accuracies'].items()
            ])
            st.dataframe(accuracy_df, use_container_width=True, hide_index=True)
        
        st.plotly_chart(create_feature_importance_chart(model_data), use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Understanding Feature Importance</h4>
            <p>The chart above shows how much each health factor contributes to the prediction. 
            Higher values indicate stronger influence on diabetes risk assessment. 
            This helps you understand which factors are most critical for your health.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Reference Ranges")
        
        reference_data = pd.DataFrame({
            'Health Metric': ['Glucose (fasting)', 'Blood Pressure', 'BMI', 'Age Factor'],
            'Normal Range': ['70-100 mg/dL', '<120/80 mm Hg', '18.5-24.9 kg/m²', 'Risk increases with age'],
            'Pre-Diabetic': ['100-125 mg/dL', '120-139/80-89 mm Hg', '25-29.9 kg/m²', 'Higher risk >45 years'],
            'Diabetic/High Risk': ['>126 mg/dL', '>140/90 mm Hg', '>30 kg/m²', 'Significant risk >60 years']
        })
        
        st.dataframe(reference_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Retrain Model with Custom Data")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file with diabetes data to retrain the model",
            type=['csv'],
            help="CSV must contain columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🔄 Retrain Models with This Data"):
                    with st.spinner("Retraining all models... This may take a moment."):
                        try:
                            new_model_data = retrain_with_data(df)
                            st.cache_resource.clear()
                            st.success(f"✅ Models retrained successfully with {len(df)} samples!")
                            st.info("Refresh the page to use the new models.")
                        except ValueError as e:
                            st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with tab3:
        st.markdown("### Prediction History")
        st.markdown("Track your diabetes risk assessments over time.")
        
        history = get_prediction_history(st.session_state.session_id)
        
        if history:
            trend_chart = create_history_trend_chart(history)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            
            st.markdown("### Recent Predictions")
            
            for h in history[:10]:
                risk_color = "#28a745" if h.risk_level == "Low" else "#ffc107" if h.risk_level == "Medium" else "#dc3545"
                
                with st.expander(f"📊 {h.created_at.strftime('%B %d, %Y at %I:%M %p')} - Risk: {h.risk_level} ({h.probability:.1f}%)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Health Metrics:**")
                        st.write(f"- Glucose: {h.glucose:.1f} mg/dL")
                        st.write(f"- BMI: {h.bmi:.1f} kg/m²")
                        st.write(f"- Blood Pressure: {h.blood_pressure:.1f} mm Hg")
                        st.write(f"- Age: {h.age} years")
                    
                    with col2:
                        st.markdown("**Additional Metrics:**")
                        st.write(f"- Pregnancies: {h.pregnancies}")
                        st.write(f"- Insulin: {h.insulin:.1f} mu U/ml")
                        st.write(f"- Skin Thickness: {h.skin_thickness:.1f} mm")
                        st.write(f"- Family History Score: {h.diabetes_pedigree:.3f}")
                    
                    st.markdown(f"**Model Used:** {h.model_used}")
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{h.id}"):
                        delete_prediction(h.id)
                        st.rerun()
        else:
            st.info("No prediction history yet. Make a prediction in the Prediction tab to start tracking your health journey!")
        
        pop_stats = get_population_statistics()
        if pop_stats['total_predictions'] > 0:
            st.markdown("---")
            st.markdown("### Population Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", pop_stats['total_predictions'])
            with col2:
                st.metric("Low Risk %", f"{pop_stats['low_risk_pct']:.1f}%")
            with col3:
                st.metric("Medium Risk %", f"{pop_stats['medium_risk_pct']:.1f}%")
            with col4:
                st.metric("High Risk %", f"{pop_stats['high_risk_pct']:.1f}%")
    
    with tab4:
        st.markdown("### Compare All ML Models")
        st.markdown("See how different machine learning models predict your diabetes risk.")
        
        if 'last_prediction' in st.session_state:
            patient_data = st.session_state.last_prediction['patient_data']
            
            st.markdown("**Using your last submitted health data:**")
            
            all_results = predict_with_all_models(patient_data, model_data)
            
            st.plotly_chart(create_model_comparison_chart(all_results), use_container_width=True)
            
            st.markdown("### Detailed Model Results")
            
            results_df = pd.DataFrame([
                {
                    'Model': name,
                    'Probability': f"{res['probability']:.1f}%",
                    'Risk Level': res['risk_level'],
                    'Prediction': 'Diabetes' if res['prediction'] == 1 else 'No Diabetes'
                }
                for name, res in all_results.items()
            ])
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            probabilities = [all_results[m]['probability'] for m in all_results]
            avg_prob = sum(probabilities) / len(probabilities)
            agreement = sum(1 for m in all_results if all_results[m]['risk_level'] == all_results['Ensemble']['risk_level'])
            
            st.markdown("### Model Agreement")
            st.metric("Average Probability", f"{avg_prob:.1f}%")
            st.metric("Models Agreeing with Ensemble", f"{agreement}/{len(all_results)}")
            
            pop_stats = get_population_statistics()
            if pop_stats['total_predictions'] > 0:
                st.markdown("---")
                st.markdown("### Your Metrics vs Population")
                
                st.plotly_chart(create_population_comparison_chart(patient_data, pop_stats), use_container_width=True)
                
                st.markdown("### Detailed Comparison")
                
                comparisons = [
                    ('Glucose', patient_data['Glucose'], pop_stats['avg_glucose'], 'mg/dL'),
                    ('BMI', patient_data['BMI'], pop_stats['avg_bmi'], 'kg/m²'),
                    ('Blood Pressure', patient_data['BloodPressure'], pop_stats['avg_blood_pressure'], 'mm Hg'),
                    ('Age', patient_data['Age'], pop_stats['avg_age'], 'years')
                ]
                
                for metric, your_val, pop_val, unit in comparisons:
                    diff = your_val - pop_val
                    diff_pct = (diff / pop_val) * 100 if pop_val > 0 else 0
                    
                    if metric in ['Glucose', 'BMI', 'Blood Pressure']:
                        status = "higher" if diff > 0 else "lower"
                        color_class = "comparison-worse" if diff > 0 else "comparison-better"
                    else:
                        status = "higher" if diff > 0 else "lower"
                        color_class = ""
                    
                    st.markdown(f"**{metric}:** Your value ({your_val:.1f} {unit}) is <span class='{color_class}'>{abs(diff_pct):.1f}% {status}</span> than average ({pop_val:.1f} {unit})", unsafe_allow_html=True)
        else:
            st.info("👈 Make a prediction first in the Prediction tab to compare all models.")
    
    with tab5:
        st.markdown("### Generate PDF Health Report")
        st.markdown("Download a comprehensive health report with your prediction results and personalized recommendations.")
        
        if 'last_prediction' in st.session_state:
            patient_data = st.session_state.last_prediction['patient_data']
            result = st.session_state.last_prediction['result']
            
            st.markdown("**Report Preview:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Your Health Metrics:**")
                for key, value in patient_data.items():
                    label = key.replace('DiabetesPedigreeFunction', 'Family History Score')
                    if isinstance(value, float):
                        st.write(f"- {label}: {value:.2f}")
                    else:
                        st.write(f"- {label}: {value}")
            
            with col2:
                st.markdown("**Prediction Results:**")
                st.write(f"- Risk Level: **{result['risk_level']}**")
                st.write(f"- Probability: **{result['probability']:.1f}%**")
                st.write(f"- Model Used: **{result.get('model_used', 'RandomForest')}**")
            
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating your personalized health report..."):
                    model_accuracies = model_data.get('accuracies', None)
                    pdf_buffer = create_health_report(patient_data, result, model_accuracies)
                    
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_buffer,
                        file_name=generate_report_filename(),
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ Report generated successfully! Click the button above to download.")
        else:
            st.info("👈 Make a prediction first in the Prediction tab to generate a report.")
    
    with tab6:
        st.markdown("### Understanding Diabetes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### What is Diabetes?
            
            Diabetes is a chronic health condition that affects how your body turns food into energy. 
            When you have diabetes, your body either doesn't make enough insulin or can't use the 
            insulin it makes as well as it should.
            
            **Types of Diabetes:**
            - **Type 1**: The body doesn't produce insulin
            - **Type 2**: The body doesn't use insulin well (most common)
            - **Gestational**: Develops during pregnancy
            """)
            
            st.markdown("""
            #### Risk Factors
            
            Several factors can increase your risk of developing diabetes:
            
            - **Family History**: Having a parent or sibling with diabetes
            - **Weight**: Being overweight or obese
            - **Age**: Risk increases with age, especially after 45
            - **Physical Inactivity**: Sedentary lifestyle
            - **High Blood Pressure**: Blood pressure over 140/90 mm Hg
            - **Abnormal Cholesterol**: Low HDL or high triglycerides
            """)
        
        with col2:
            st.markdown("""
            #### Prevention Tips
            
            You can take steps to prevent or delay type 2 diabetes:
            
            1. **Maintain Healthy Weight**
               - Aim for a BMI between 18.5 and 24.9
               - Even modest weight loss can help
            
            2. **Stay Active**
               - At least 150 minutes of moderate activity per week
               - Include strength training 2+ times per week
            
            3. **Eat Healthy**
               - Choose whole grains over refined carbs
               - Include vegetables in every meal
               - Limit sugary drinks and processed foods
            
            4. **Regular Check-ups**
               - Monitor blood sugar levels
               - Annual health screenings
               - Discuss family history with your doctor
            """)
            
            st.markdown("""
            #### Warning Signs
            
            Early signs of diabetes include:
            - Increased thirst and urination
            - Unexplained weight loss
            - Fatigue and weakness
            - Blurred vision
            - Slow-healing cuts or bruises
            - Tingling in hands or feet
            """)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ Important Disclaimer</h4>
            <p>This tool is for educational purposes only and should not be used as a substitute for 
            professional medical advice, diagnosis, or treatment. Always consult with a qualified 
            healthcare provider for proper evaluation and guidance regarding your health.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This Tool")
    st.sidebar.info("""
    This diabetes prediction tool uses multiple machine learning algorithms to assess your risk 
    based on key health indicators.
    
    **Available Models:**
    - Random Forest
    - Support Vector Machine (SVM)
    - Neural Network
    - Logistic Regression
    - Ensemble (Combined)
    
    **Features:**
    - Instant Risk Assessment
    - Multiple Model Comparison
    - Prediction History Tracking
    - PDF Report Generation
    - Population Benchmarking
    """)
    
    st.sidebar.markdown("### Session Info")
    st.sidebar.text(f"Session ID: {st.session_state.session_id[:8]}...")

if __name__ == "__main__":
    main()
