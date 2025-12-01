import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    DB_AVAILABLE = True
else:
    engine = None
    SessionLocal = None
    DB_AVAILABLE = False

Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, nullable=True)
    session_id = Column(String(100), nullable=False)
    
    pregnancies = Column(Integer)
    glucose = Column(Float)
    blood_pressure = Column(Float)
    skin_thickness = Column(Float)
    insulin = Column(Float)
    bmi = Column(Float)
    diabetes_pedigree = Column(Float)
    age = Column(Integer)
    
    prediction = Column(Integer)
    probability = Column(Float)
    risk_level = Column(String(20))
    
    model_used = Column(String(50), default='RandomForest')
    ensemble_prediction = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

def init_db():
    """Initialize database tables."""
    if not DB_AVAILABLE:
        return
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session."""
    if not DB_AVAILABLE:
        return None
    db = SessionLocal()
    try:
        return db
    except:
        db.close()
        raise

def save_prediction(session_id, patient_data, result, model_used='RandomForest', ensemble=False, patient_id=None):
    """Save a prediction to history."""
    if not DB_AVAILABLE:
        return None
    db = get_db()
    try:
        prediction = PredictionHistory(
            patient_id=patient_id,
            session_id=session_id,
            pregnancies=int(patient_data['Pregnancies']),
            glucose=float(patient_data['Glucose']),
            blood_pressure=float(patient_data['BloodPressure']),
            skin_thickness=float(patient_data['SkinThickness']),
            insulin=float(patient_data['Insulin']),
            bmi=float(patient_data['BMI']),
            diabetes_pedigree=float(patient_data['DiabetesPedigreeFunction']),
            age=int(patient_data['Age']),
            prediction=int(result['prediction']),
            probability=float(result['probability']),
            risk_level=str(result['risk_level']),
            model_used=model_used,
            ensemble_prediction=ensemble
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction.id
    finally:
        db.close()

def get_prediction_history(session_id=None, limit=50):
    """Get prediction history."""
    if not DB_AVAILABLE:
        return []
    db = get_db()
    try:
        query = db.query(PredictionHistory)
        if session_id:
            query = query.filter(PredictionHistory.session_id == session_id)
        return query.order_by(PredictionHistory.created_at.desc()).limit(limit).all()
    finally:
        db.close()

def get_prediction_by_id(prediction_id):
    """Get a specific prediction by ID."""
    if not DB_AVAILABLE:
        return None
    db = get_db()
    try:
        return db.query(PredictionHistory).filter(PredictionHistory.id == prediction_id).first()
    finally:
        db.close()

def get_population_statistics():
    """Get population statistics from all predictions."""
    default_stats = {
        'total_predictions': 0,
        'avg_glucose': 120,
        'avg_bmi': 25,
        'avg_blood_pressure': 70,
        'avg_age': 35,
        'high_risk_pct': 0,
        'medium_risk_pct': 0,
        'low_risk_pct': 0
    }
    
    if not DB_AVAILABLE:
        return default_stats
    
    db = get_db()
    try:
        predictions = db.query(PredictionHistory).all()
        
        if not predictions:
            return default_stats
        
        total = len(predictions)
        avg_glucose = sum(p.glucose for p in predictions) / total
        avg_bmi = sum(p.bmi for p in predictions) / total
        avg_bp = sum(p.blood_pressure for p in predictions) / total
        avg_age = sum(p.age for p in predictions) / total
        
        high_risk = sum(1 for p in predictions if p.risk_level == 'High')
        medium_risk = sum(1 for p in predictions if p.risk_level == 'Medium')
        low_risk = sum(1 for p in predictions if p.risk_level == 'Low')
        
        return {
            'total_predictions': total,
            'avg_glucose': avg_glucose,
            'avg_bmi': avg_bmi,
            'avg_blood_pressure': avg_bp,
            'avg_age': avg_age,
            'high_risk_pct': (high_risk / total) * 100 if total > 0 else 0,
            'medium_risk_pct': (medium_risk / total) * 100 if total > 0 else 0,
            'low_risk_pct': (low_risk / total) * 100 if total > 0 else 0
        }
    finally:
        db.close()

def delete_prediction(prediction_id):
    """Delete a prediction from history."""
    if not DB_AVAILABLE:
        return False
    db = get_db()
    try:
        prediction = db.query(PredictionHistory).filter(PredictionHistory.id == prediction_id).first()
        if prediction:
            db.delete(prediction)
            db.commit()
            return True
        return False
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    print("Database tables created successfully!")
