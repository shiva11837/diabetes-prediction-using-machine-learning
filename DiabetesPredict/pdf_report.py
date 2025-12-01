from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.piecharts import Pie
from io import BytesIO
from datetime import datetime

def create_health_report(patient_data, result, model_accuracies=None):
    """
    Generate a PDF health report with prediction results and recommendations.
    
    Args:
        patient_data: dict with patient health metrics
        result: dict with prediction results
        model_accuracies: dict with model accuracy scores (optional)
    
    Returns:
        BytesIO object containing the PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e3a5f'),
        spaceAfter=20,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1e3a5f'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8
    )
    
    elements = []
    
    elements.append(Paragraph("Diabetes Risk Assessment Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                             ParagraphStyle('Date', parent=styles['Normal'], alignment=1, textColor=colors.gray)))
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Patient Health Metrics", heading_style))
    
    metrics_data = [
        ['Health Metric', 'Value', 'Reference Range'],
        ['Pregnancies', str(int(patient_data['Pregnancies'])), '0-17'],
        ['Glucose Level', f"{patient_data['Glucose']:.1f} mg/dL", '70-140 mg/dL (normal)'],
        ['Blood Pressure', f"{patient_data['BloodPressure']:.1f} mm Hg", '< 120 mm Hg (normal)'],
        ['Skin Thickness', f"{patient_data['SkinThickness']:.1f} mm", '10-50 mm'],
        ['Insulin Level', f"{patient_data['Insulin']:.1f} mu U/ml", '16-166 mu U/ml'],
        ['BMI', f"{patient_data['BMI']:.1f} kg/m²", '18.5-24.9 (healthy)'],
        ['Family History Score', f"{patient_data['DiabetesPedigreeFunction']:.3f}", '0.078-2.42'],
        ['Age', f"{int(patient_data['Age'])} years", 'N/A']
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Risk Assessment Results", heading_style))
    
    if result['risk_level'] == 'Low':
        risk_color = colors.HexColor('#28a745')
    elif result['risk_level'] == 'Medium':
        risk_color = colors.HexColor('#ffc107')
    else:
        risk_color = colors.HexColor('#dc3545')
    
    risk_data = [
        ['Risk Level', 'Probability', 'Model Used'],
        [result['risk_level'], f"{result['probability']:.1f}%", result.get('model_used', 'RandomForest')]
    ]
    
    risk_table = Table(risk_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 1), (-1, 1), 12),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 12),
        ('BACKGROUND', (0, 1), (0, 1), risk_color),
        ('TEXTCOLOR', (0, 1), (0, 1), colors.white),
        ('BACKGROUND', (1, 1), (-1, 1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 20))
    
    if model_accuracies:
        elements.append(Paragraph("Model Performance", heading_style))
        
        model_data = [['Model', 'Accuracy']]
        for model_name, accuracy in model_accuracies.items():
            model_data.append([model_name, f"{accuracy:.1%}"])
        
        model_table = Table(model_data, colWidths=[3.3*inch, 3.3*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        elements.append(model_table)
        elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Personalized Recommendations", heading_style))
    
    recommendations = []
    
    if patient_data['Glucose'] > 125:
        recommendations.append("• <b>Blood Sugar Management:</b> Your glucose level is elevated. Consider reducing sugar intake, eating more fiber-rich foods, and monitoring your blood sugar regularly.")
    elif patient_data['Glucose'] > 100:
        recommendations.append("• <b>Blood Sugar Awareness:</b> Your glucose is in the pre-diabetic range. Focus on balanced meals with low glycemic index foods.")
    
    if patient_data['BMI'] > 30:
        recommendations.append("• <b>Weight Management:</b> Your BMI indicates obesity. A combination of balanced nutrition and regular physical activity can help reduce diabetes risk significantly.")
    elif patient_data['BMI'] > 25:
        recommendations.append("• <b>Healthy Weight:</b> Your BMI indicates you're overweight. Aim for gradual weight loss through sustainable lifestyle changes.")
    
    if patient_data['BloodPressure'] > 90:
        recommendations.append("• <b>Blood Pressure Control:</b> Your diastolic blood pressure is elevated. Reduce sodium intake, manage stress, and consider regular monitoring.")
    
    if patient_data['Age'] > 45:
        recommendations.append("• <b>Regular Screening:</b> Given your age, annual diabetes screening is recommended. Early detection is key to effective management.")
    
    if patient_data['DiabetesPedigreeFunction'] > 0.5:
        recommendations.append("• <b>Family History Alert:</b> Your family history score suggests genetic predisposition. Be extra vigilant about lifestyle factors you can control.")
    
    recommendations.append("• <b>Physical Activity:</b> Aim for at least 150 minutes of moderate exercise per week. This improves insulin sensitivity and overall health.")
    recommendations.append("• <b>Regular Check-ups:</b> Schedule regular appointments with your healthcare provider to monitor your health metrics.")
    
    for rec in recommendations:
        elements.append(Paragraph(rec, normal_style))
    
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Understanding Your Risk Level", heading_style))
    
    risk_explanation = {
        'Low': "Your current health metrics suggest a <b>low probability</b> of developing diabetes. Continue maintaining a healthy lifestyle with balanced nutrition, regular exercise, and routine health check-ups.",
        'Medium': "Your health metrics indicate a <b>moderate risk</b> for diabetes. This is an important time to make lifestyle adjustments. Focus on diet, exercise, and regular monitoring to prevent progression.",
        'High': "Your assessment indicates an <b>elevated risk</b> for diabetes. We strongly recommend consulting with a healthcare professional for a comprehensive evaluation and personalized management plan."
    }
    
    elements.append(Paragraph(risk_explanation[result['risk_level']], normal_style))
    
    elements.append(Spacer(1, 30))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.gray,
        alignment=1
    )
    
    elements.append(Paragraph("⚠️ IMPORTANT DISCLAIMER", 
                             ParagraphStyle('DisclaimerTitle', parent=disclaimer_style, fontName='Helvetica-Bold', fontSize=10)))
    elements.append(Paragraph(
        "This report is generated by an AI-powered risk assessment tool and is intended for educational purposes only. "
        "It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult with a qualified healthcare provider for proper evaluation and guidance regarding your health.",
        disclaimer_style
    ))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_report_filename():
    """Generate a unique filename for the report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"diabetes_risk_report_{timestamp}.pdf"
