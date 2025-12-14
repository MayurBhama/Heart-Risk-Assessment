"""
CardioPredict - Heart Disease Risk Assessment
Industry-Grade Clinical UI with Safety, Trust, and Actionable Insights
"""

import streamlit as st
import requests
from datetime import datetime
from io import BytesIO

# Configuration
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="CardioPredict - Heart Risk Assessment",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d1f3d 50%, #4a2040 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid #3d2a50;
    }
    .main-header {
        font-size: 2.8rem !important; 
        font-weight: 700 !important; 
        color: #ffffff !important;
        margin: 0 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        line-height: 1.2 !important;
    }
    .sub-header {
        font-size: 1.2rem; 
        color: rgba(255,255,255,0.8); 
        margin-top: 1rem;
        margin-bottom: 0;
    }
    .disclaimer-box {
        background: #1a1a2e; 
        border: 1px solid #444;
        padding: 0.8rem 1rem; 
        border-radius: 6px; 
        font-size: 0.85rem; 
        color: #bbb; 
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .priority-high {border-left: 4px solid #ff4b4b; padding-left: 1rem; margin: 0.5rem 0; background: rgba(255,75,75,0.1); padding: 0.5rem; border-radius: 0 4px 4px 0;}
    .priority-medium {border-left: 4px solid #ffa726; padding-left: 1rem; margin: 0.5rem 0; background: rgba(255,167,38,0.1); padding: 0.5rem; border-radius: 0 4px 4px 0;}
    .priority-low {border-left: 4px solid #66bb6a; padding-left: 1rem; margin: 0.5rem 0; background: rgba(102,187,106,0.1); padding: 0.5rem; border-radius: 0 4px 4px 0;}
    .calibration-note {font-size: 0.8rem; color: #888; font-style: italic; margin-top: 0.3rem;}
    
    /* Action plan cards */
    .action-card {
        background: rgba(30, 40, 60, 0.5);
        border-left: 3px solid #4a9eff;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
    .action-card.high { border-left-color: #ff6b6b; }
    .action-card.medium { border-left-color: #ffa726; }
    .action-card.long { border-left-color: #66bb6a; }
    .action-card .action-text { font-size: 1rem; color: #eee; margin-bottom: 0.3rem; }
    .action-card .target-text { font-size: 0.85rem; color: #888; font-style: italic; }
    
    /* Form labels - Streamlit specific selectors */
    [data-testid="stNumberInput"] label p,
    [data-testid="stSelectbox"] label p,
    div[data-baseweb="select"] ~ div,
    .stNumberInput > label > div > p,
    .stSelectbox > label > div > p {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
    }
    
    /* Also target the label text directly */
    label[data-testid="stWidgetLabel"] p {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
    }
    
    /* Dropdown - clean transparent white theme */
    div[data-baseweb="select"] > div {
        border-color: rgba(255,255,255,0.3) !important;
    }
    div[data-baseweb="select"]:focus-within > div {
        border-color: rgba(255,255,255,0.5) !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.2) !important;
    }
    div[data-baseweb="popover"] {
        border-color: rgba(255,255,255,0.2) !important;
    }
    li[role="option"]:hover {
        background-color: rgba(255,255,255,0.1) !important;
    }
    li[role="option"][aria-selected="true"] {
        background-color: rgba(255,255,255,0.15) !important;
    }
    
    /* Fix red helper text to gray */
    [data-testid="InputInstructions"] {
        color: #888 !important;
    }
    
    /* Button - subtle gray */
    .stButton > button {
        background-color: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.3) !important;
        color: #fff !important;
    }
    .stButton > button:hover {
        background-color: rgba(255,255,255,0.2) !important;
        border-color: rgba(255,255,255,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


def send_request(payload):
    """Send prediction request to FastAPI backend."""
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        if response.status_code != 200:
            return {"error": response.text}
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server. Make sure FastAPI is running on port 8000."}
    except Exception as e:
        return {"error": str(e)}


def get_risk_contributors(bmi, bp_cat, chol, gluc, smoke, alco, active, age):
    """Analyze and rank risk contributors with modifiability."""
    contributors = []
    
    # Age - Non-modifiable
    if age >= 55:
        contributors.append(("MEDIUM", "Age", f"{age} years", "Non-modifiable"))
    
    # Blood Pressure - Treatable (NHS: 0=Normal, 1=High Normal, 2+=Hypertension)
    if bp_cat >= 3:
        contributors.append(("HIGH", "Blood Pressure", f"High blood pressure (Stage {bp_cat-1})", "Treatable"))
    elif bp_cat == 2:
        contributors.append(("MEDIUM", "Blood Pressure", "High blood pressure (Stage 1)", "Treatable"))
    # bp_cat 0-1 (Normal/High Normal) - no risk contributor shown
    
    # Cholesterol - Modifiable
    if chol == 3:
        contributors.append(("HIGH", "Cholesterol", "Very high levels", "Modifiable"))
    elif chol == 2:
        contributors.append(("MEDIUM", "Cholesterol", "Above normal", "Modifiable"))
    
    # BMI - Modifiable
    if bmi >= 30:
        contributors.append(("HIGH", "BMI", f"Obese ({int(bmi)})", "Modifiable"))
    elif bmi >= 25:
        contributors.append(("MEDIUM", "BMI", f"Overweight ({int(bmi)})", "Modifiable"))
    
    # Glucose - Modifiable
    if gluc == 3:
        contributors.append(("HIGH", "Glucose", "Very high levels", "Modifiable"))
    elif gluc == 2:
        contributors.append(("MEDIUM", "Glucose", "Above normal", "Modifiable"))
    
    # Smoking - Modifiable (Very High Impact)
    if smoke == 1:
        contributors.append(("HIGH", "Smoking", "Active smoker", "Modifiable - Very High Impact"))
    
    # Alcohol - Modifiable
    if alco == 1:
        contributors.append(("MEDIUM", "Alcohol", "Regular consumption", "Modifiable"))
    
    # Physical Activity - Modifiable
    if active == 0:
        contributors.append(("LOW", "Physical Activity", "Sedentary lifestyle", "Modifiable"))
    
    # Sort by severity
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    contributors.sort(key=lambda x: severity_order[x[0]])
    
    return contributors


def get_prioritized_recommendations(bmi, bp_cat, chol, gluc, smoke, alco, active, ap_hi):
    """Generate prioritized recommendations with measurable targets."""
    high_priority = []
    medium_priority = []
    long_term = []
    
    # High Priority with targets
    if bp_cat >= 3:
        high_priority.append(("Consult a healthcare provider about blood pressure management", "Target: < 130/80 mmHg"))
    if chol == 3:
        high_priority.append(("Discuss cholesterol management with your doctor", "Target: LDL < 100 mg/dL"))
    if smoke == 1:
        high_priority.append(("Consider a smoking cessation program", "Goal: Smoke-free within 3-6 months"))
    
    # Medium Priority with targets
    # BP recommendations only for actual hypertension (bp_cat >= 2)
    if bp_cat >= 2:
        medium_priority.append(("Monitor blood pressure and reduce sodium intake", "Target: < 2g sodium/day"))
    if bmi >= 25:
        target_loss = max(3, int((bmi - 24) * 1.5))
        medium_priority.append(("Work on gradual weight reduction", f"Target: -{target_loss} kg in 3 months"))
    if chol == 2:
        medium_priority.append(("Increase dietary fiber and reduce saturated fats", "Target: 25-30g fiber/day"))
    if gluc >= 2:
        medium_priority.append(("Reduce sugar and refined carbohydrates", "Target: < 25g added sugar/day"))
    if alco == 1:
        medium_priority.append(("Limit alcohol consumption", "Target: < 2 drinks/week"))
    
    # Long Term with targets
    if active == 0:
        long_term.append(("Begin regular physical activity", "Target: 30 min walking, 5x/week"))
    else:
        long_term.append(("Maintain current activity, consider adding variety", "Target: 150 min moderate activity/week"))
    long_term.append(("Schedule regular health monitoring", "Target: Check-up every 6 months"))
    long_term.append(("Track key health metrics at home", "Monitor: BP and weight weekly"))
    
    # Ensure at least one item in each category
    if not high_priority:
        high_priority.append(("Continue current healthy practices", "Maintain regular preventive care"))
    if not medium_priority:
        medium_priority.append(("Focus on balanced nutrition", "Target: 5 servings fruits/vegetables daily"))
    
    return high_priority, medium_priority, long_term


def get_age_percentile(age, probability):
    """Estimate risk percentile compared to age group."""
    base_percentile = probability * 100
    
    if age < 40:
        percentile = min(95, base_percentile + 15)
    elif age < 50:
        percentile = min(95, base_percentile + 10)
    elif age < 60:
        percentile = base_percentile + 5
    else:
        percentile = max(50, base_percentile - 5)
    
    return int(min(99, max(1, percentile)))


def get_risk_context(risk_level):
    """Get contextual explanation for risk level with safe language."""
    contexts = {
        "Very High": {
            "meaning": "This score indicates a significantly elevated cardiovascular risk profile compared to the general population.",
            "action": "A medical evaluation is recommended in the near future to discuss risk factors and preventive options.",
            "note": "This is a risk assessment tool, not a diagnosis. Many individuals with elevated risk profiles maintain good heart health through appropriate lifestyle choices and medical guidance."
        },
        "High": {
            "meaning": "Your cardiovascular risk profile is above average for your demographic group.",
            "action": "Consider discussing these results with your healthcare provider at your next visit.",
            "note": "Lifestyle modifications can significantly improve your risk profile over time."
        },
        "Moderate": {
            "meaning": "Your risk profile is within the expected range but shows some areas for potential improvement.",
            "action": "Focus on the suggested lifestyle adjustments and continue monitoring your health metrics.",
            "note": "Regular monitoring and healthy habits can help maintain or improve your current profile."
        },
        "Low": {
            "meaning": "Your cardiovascular risk profile appears to be favorable.",
            "action": "Continue your current healthy practices.",
            "note": "Maintain regular check-ups to support continued heart health."
        }
    }
    return contexts.get(risk_level, contexts["Moderate"])


def calculate_what_if(base_payload, smoke_change=None, active_change=None, bp_reduction=None):
    """Calculate risk with hypothetical changes."""
    modified = base_payload.copy()
    
    if smoke_change is not None:
        modified["smoke"] = smoke_change
    if active_change is not None:
        modified["active"] = active_change
    if bp_reduction is not None:
        modified["ap_hi"] = max(90, base_payload["ap_hi"] - bp_reduction)
        modified["ap_lo"] = max(60, base_payload["ap_lo"] - int(bp_reduction * 0.6))
    
    result = send_request(modified)
    if "error" not in result:
        return result.get("probability", 0) * 100
    return None


def generate_pdf_report(inputs, output, contributors, recommendations):
    """Generate PDF report using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width/2, height - 0.8*inch, "CardioPredict - Risk Assessment Report")
        
        c.setFont("Helvetica", 10)
        c.drawCentredString(width/2, height - 1.1*inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Patient Info
        y = height - 1.7*inch
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, y, "Patient Information")
        y -= 0.25*inch
        
        c.setFont("Helvetica", 10)
        info_text = f"Age: {inputs['age']} years  |  Gender: {inputs['gender'].title()}  |  Height: {inputs['height']} cm  |  Weight: {inputs['weight']} kg"
        c.drawString(1*inch, y, info_text)
        y -= 0.2*inch
        c.drawString(1*inch, y, f"Blood Pressure: {inputs['ap_hi']}/{inputs['ap_lo']} mmHg")
        y -= 0.4*inch
        
        # Risk Assessment
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, y, "Risk Assessment")
        y -= 0.25*inch
        
        c.setFont("Helvetica", 10)
        c.drawString(1*inch, y, f"Risk Score: {int(output['probability']*100)}%  |  Classification: {output['risk_level']}")
        y -= 0.2*inch
        c.drawString(1*inch, y, f"BMI: {output['risk_factors'].get('bmi', 'N/A')}")
        y -= 0.4*inch
        
        # Risk Contributors
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, y, "Key Risk Contributors")
        y -= 0.25*inch
        
        c.setFont("Helvetica", 10)
        for level, factor, desc, mod in contributors[:5]:
            text = f"[{level}] {factor}: {desc} ({mod})"
            c.drawString(1*inch, y, text)
            y -= 0.2*inch
        y -= 0.2*inch
        
        # Recommendations
        high_p, med_p, long_p = recommendations
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, y, "Recommendations")
        y -= 0.25*inch
        
        c.setFont("Helvetica-Bold", 10)
        c.drawString(1*inch, y, "High Priority:")
        y -= 0.2*inch
        c.setFont("Helvetica", 9)
        for rec, target in high_p[:3]:
            # Wrap long text
            if len(rec) > 65:
                c.drawString(1.1*inch, y, f"- {rec[:65]}...")
            else:
                c.drawString(1.1*inch, y, f"- {rec}")
            y -= 0.15*inch
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(1.2*inch, y, f"  {target}")
            c.setFont("Helvetica", 9)
            y -= 0.2*inch
        
        y -= 0.1*inch
        c.setFont("Helvetica-Bold", 10)
        c.drawString(1*inch, y, "Medium Priority:")
        y -= 0.2*inch
        c.setFont("Helvetica", 9)
        for rec, target in med_p[:3]:
            if len(rec) > 65:
                c.drawString(1.1*inch, y, f"- {rec[:65]}...")
            else:
                c.drawString(1.1*inch, y, f"- {rec}")
            y -= 0.15*inch
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(1.2*inch, y, f"  {target}")
            c.setFont("Helvetica", 9)
            y -= 0.2*inch
        
        # Footer - Disclaimer
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.gray)
        c.drawCentredString(width/2, 1*inch, "This assessment is for educational and informational purposes only.")
        c.drawCentredString(width/2, 0.8*inch, "It is not a substitute for professional medical advice, diagnosis, or treatment.")
        c.drawCentredString(width/2, 0.6*inch, "Risk categories aligned with standard cardiovascular prevention guidelines (educational use only).")
        
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        return generate_text_report(inputs, output, contributors, recommendations)


def generate_text_report(inputs, output, contributors, recommendations):
    """Fallback text report."""
    report = []
    report.append("=" * 60)
    report.append("CARDIOPREDICT - HEART DISEASE RISK ASSESSMENT REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("PATIENT INFORMATION")
    report.append(f"Age: {inputs['age']} | Gender: {inputs['gender'].title()}")
    report.append(f"Height: {inputs['height']} cm | Weight: {inputs['weight']} kg")
    report.append(f"Blood Pressure: {inputs['ap_hi']}/{inputs['ap_lo']} mmHg")
    report.append("")
    report.append("RISK ASSESSMENT")
    report.append(f"Risk Score: {int(output['probability']*100)}%")
    report.append(f"Classification: {output['risk_level']}")
    report.append("")
    report.append("KEY RISK CONTRIBUTORS")
    for level, factor, desc, mod in contributors:
        report.append(f"[{level}] {factor}: {desc} ({mod})")
    report.append("")
    high_p, med_p, long_p = recommendations
    report.append("RECOMMENDATIONS")
    report.append("High Priority:")
    for rec, target in high_p:
        report.append(f"  - {rec}")
        report.append(f"    {target}")
    report.append("Medium Priority:")
    for rec, target in med_p:
        report.append(f"  - {rec}")
        report.append(f"    {target}")
    report.append("")
    report.append("=" * 60)
    report.append("DISCLAIMER")
    report.append("This assessment is for educational purposes only.")
    report.append("It is not a substitute for professional medical advice.")
    report.append("Risk categories aligned with standard cardiovascular")
    report.append("prevention guidelines (educational use only).")
    report.append("=" * 60)
    return "\n".join(report)


def main():
    # GRADIENT HEADER BOX - Like Multi-Disease Detection System
    st.markdown("""
    <div class="header-box">
        <p class="main-header">CardioPredict - Heart Disease Risk Assessment</p>
        <p class="sub-header">Cardiovascular risk analysis with personalized recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # MEDICAL DISCLAIMER - Below subtitle
    st.markdown("""
    <div class="disclaimer-box">
        <strong>Medical Disclaimer:</strong> This tool is for educational and informational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult a qualified healthcare provider for medical concerns.
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form with HEALTHY DEFAULTS
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age (years)", 18, 100, 35, step=1)
            gender = st.selectbox("Gender", ["male", "female"])
            height = st.number_input("Height (cm)", 120, 220, 170, step=1)
            weight = st.number_input("Weight (kg)", 30, 200, 70, step=1)
        
        with col2:
            st.subheader("Health Metrics")
            ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", 80, 250, 120, step=1)
            ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 200, 80, step=1)
            
            cholesterol = st.selectbox(
                "Cholesterol Level",
                ["Normal (< 200 mg/dL)", "Above Normal (200-239 mg/dL)", "High (>= 240 mg/dL)"],
                index=0
            )
            gluc = st.selectbox(
                "Glucose Level (Fasting)",
                ["Normal (< 100 mg/dL)", "Above Normal (100-125 mg/dL)", "High (>= 126 mg/dL)"],
                index=0
            )
        
        st.subheader("Lifestyle Factors")
        col3, col4, col5 = st.columns(3)
        with col3:
            smoke = st.selectbox("Do you smoke?", ["No", "Yes"], index=0)
        with col4:
            alco = st.selectbox("Do you consume alcohol regularly?", ["No", "Yes"], index=0)
        with col5:
            active = st.selectbox("Are you physically active?", ["Yes", "No"], index=0)
        
        submitted = st.form_submit_button("Analyze Risk Profile", use_container_width=True)
    
    # MODEL OVERVIEW AND LIMITATIONS (Always visible on main page)
    st.write("---")
    
    # MODEL OVERVIEW (For Transparency)
    with st.expander("Model Overview (For Transparency)"):
        st.markdown('''
        <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
            <strong>Dataset Source</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">Cardiovascular Disease dataset from Kaggle (70,000 patient records)</em>
        </div>
        <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
            <strong>Model Type</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">XGBoost Classifier with hyperparameter tuning via RandomizedSearchCV. RandomizedSearchCV was chosen for efficient hyperparameter exploration under compute constraints.</em>
        </div>
        <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
            <strong>Features Used</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">Age, gender, height, weight, blood pressure (systolic/diastolic), cholesterol level, glucose level, smoking status, alcohol consumption, physical activity</em>
        </div>
        <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
            <strong>Feature Engineering</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">BMI derived from height & weight • Blood pressure categorized using clinical thresholds • Lifestyle factors encoded with medical relevance</em>
        </div>
        <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
            <strong>Evaluation Metrics</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">ROC-AUC, Recall (Sensitivity), Precision, F1-Score. Evaluation performed using stratified cross-validation to preserve class distribution.</em>
        </div>
        <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
            <strong>Baseline Comparison</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">Baseline (Logistic Regression) ROC-AUC: 0.78 → XGBoost ROC-AUC: 0.86 → Selected for deployment due to superior performance on imbalanced healthcare data.</em>
        </div>
        <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
            <strong>Why Recall Matters in Healthcare</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">In healthcare, missing a true positive (failing to identify someone at risk) is more costly than a false alarm. High recall ensures we minimize missed cases of cardiovascular risk, prioritizing patient safety over specificity.</em>
        </div>
        ''', unsafe_allow_html=True)
    
    # LIMITATIONS (Interviewers love this)
    with st.expander("Known Limitations"):
        st.markdown('''
        <div style="border-left: 4px solid #ffa726; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,167,38,0.1); border-radius: 0 4px 4px 0;">
            <strong>Self-Reported Data</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">Lifestyle factors (smoking, alcohol, activity) are self-reported and may not reflect actual behavior</em>
        </div>
        <div style="border-left: 4px solid #ffa726; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,167,38,0.1); border-radius: 0 4px 4px 0;">
            <strong>Non-Laboratory Values</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">Cholesterol and glucose are categorical (Normal/Above Normal/High), not precise lab measurements</em>
        </div>
        <div style="border-left: 4px solid #ffa726; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,167,38,0.1); border-radius: 0 4px 4px 0;">
            <strong>Population-Level Model</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">Trained on a specific population; may not generalize perfectly to all demographics or ethnicities</em>
        </div>
        <div style="border-left: 4px solid #ffa726; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,167,38,0.1); border-radius: 0 4px 4px 0;">
            <strong>Not Age-Adjusted Survival Analysis</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">Does not account for time-to-event or survival curves; provides point-in-time risk classification</em>
        </div>
        <div style="border-left: 4px solid #ffa726; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,167,38,0.1); border-radius: 0 4px 4px 0;">
            <strong>Not Calibrated for Absolute Probability</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">Output represents relative risk classification, not an exact probability of a cardiovascular event occurring</em>
        </div>
        ''', unsafe_allow_html=True)
    
    if submitted:
        chol_map = {"Normal (< 200 mg/dL)": 1, "Above Normal (200-239 mg/dL)": 2, "High (>= 240 mg/dL)": 3}
        gluc_map = {"Normal (< 100 mg/dL)": 1, "Above Normal (100-125 mg/dL)": 2, "High (>= 126 mg/dL)": 3}
        
        payload = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": chol_map[cholesterol],
            "gluc": gluc_map[gluc],
            "smoke": 1 if smoke == "Yes" else 0,
            "alco": 1 if alco == "Yes" else 0,
            "active": 1 if active == "Yes" else 0,
        }
        
        st.session_state["base_payload"] = payload
        
        # HYPOTENSION CHECK - Data not trained on low BP
        if ap_hi < 90 or ap_lo < 60:
            # Add anchor first, then warning
            st.markdown('<div id="hypotension-anchor"></div>', unsafe_allow_html=True)
            st.warning("⚠️ **Low Blood Pressure Detected (Hypotension)**\n\nThis model was trained on data with blood pressure readings in the normal-to-high range. Predictions for hypotensive readings (systolic < 90 or diastolic < 60) may not be reliable. Please consult a healthcare provider for low blood pressure concerns.")
            
            # Auto-scroll to the warning
            import time
            import streamlit.components.v1 as components
            components.html(
                f"""
                <script>
                    // Unique: {time.time()}
                    setTimeout(function() {{
                        const anchor = parent.document.getElementById('hypotension-anchor');
                        if (anchor) {{
                            anchor.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                        }}
                    }}, 300);
                </script>
                """,
                height=0
            )
            return
        
        with st.spinner("Analyzing your cardiovascular risk profile..."):
            output = send_request(payload)
        
        if "error" in output:
            st.error(f"Error: {output['error']}")
            return
        
        if output.get("warning"):
            st.warning(output["warning"])
            return
        
        bmi = weight / ((height / 100) ** 2)
        bp_cat = output.get("risk_factors", {}).get("blood_pressure_category", 0)
        if isinstance(bp_cat, str):
            bp_cat = 0
        
        probability = output["probability"]
        risk_level = output["risk_level"]
        
        st.write("---")
        
        # Auto-scroll anchor and script
        import time
        st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
        import streamlit.components.v1 as components
        components.html(
            f"""
            <script>
                // Unique timestamp: {time.time()}
                setTimeout(function() {{
                    const anchor = parent.document.getElementById('results-anchor');
                    if (anchor) {{
                        anchor.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                    }}
                }}, 500);
            </script>
            """,
            height=0
        )
        
        # JUMP NAVIGATION
        st.markdown("""
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <a href="#risk-assessment-results" style="background: rgba(255,255,255,0.08); padding: 0.4rem 0.8rem; border-radius: 4px; color: #aaa; text-decoration: none; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.1);">Results</a>
            <a href="#risk-contributors" style="background: rgba(255,255,255,0.08); padding: 0.4rem 0.8rem; border-radius: 4px; color: #aaa; text-decoration: none; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.1);">Contributors</a>
            <a href="#personalized-action-plan" style="background: rgba(255,255,255,0.08); padding: 0.4rem 0.8rem; border-radius: 4px; color: #aaa; text-decoration: none; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.1);">Action Plan</a>
            <a href="#potential-impact-of-lifestyle-changes" style="background: rgba(255,255,255,0.08); padding: 0.4rem 0.8rem; border-radius: 4px; color: #aaa; text-decoration: none; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.1);">What-If</a>
            <a href="#export-report" style="background: rgba(255,255,255,0.08); padding: 0.4rem 0.8rem; border-radius: 4px; color: #aaa; text-decoration: none; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.1);">Report</a>
        </div>
        """, unsafe_allow_html=True)
        
        # PREDICTION RESULTS - SAFE LANGUAGE
        st.subheader("Risk Assessment Results")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            # SAFE LANGUAGE: "Identified" not "Detected"
            if output["prediction"] == 1:
                pred_text = "Elevated Cardiovascular Risk Identified"
            else:
                pred_text = "Lower Risk Profile"
            st.metric("Assessment", pred_text)
        
        with c2:
            prob_pct = int(probability * 100)
            if prob_pct >= 70:
                prob_text = f"{prob_pct}% (Significantly Elevated)"
            elif prob_pct >= 50:
                prob_text = f"{prob_pct}% (Moderately Elevated)"
            else:
                prob_text = f"{prob_pct}% (Within Expected Range)"
            st.metric("Relative Risk Index", prob_text)
        
        with c3:
            st.metric("Classification", risk_level)
        
        # RISK CALIBRATION NOTE
        st.markdown('<p class="calibration-note">This score reflects relative cardiovascular risk based on the provided factors, not the probability of an immediate cardiac event.</p>', unsafe_allow_html=True)
        
        # RISK CONTEXT - SAFE LANGUAGE
        context = get_risk_context(risk_level)
        
        st.markdown("##### Understanding Your Risk Profile")
        risk_html = f'''
        <div style="background: rgba(74,158,255,0.1); padding: 1rem; border-radius: 0 6px 6px 0; border-left: 4px solid #4a9eff;">
            <div style="color: #eee; font-size: 0.95rem; margin-bottom: 0.6rem;"><strong>What does "{risk_level}" mean?</strong></div>
            <div style="color: #bbb; font-size: 0.9rem; margin-bottom: 0.6rem;">{context["meaning"]}</div>
            <div style="color: #eee; font-size: 0.9rem;"><strong>Suggested Next Step:</strong> <span style="color: #bbb;">{context["action"]}</span></div>
            <div style="color: #999; font-size: 0.85rem; margin-top: 0.6rem; font-style: italic;">{context["note"]}</div>
        </div>
        '''
        st.markdown(risk_html, unsafe_allow_html=True)
        
        # AGE COMPARISON
        percentile = get_age_percentile(age, probability)
        
        st.markdown("##### Compared to Your Age Group")
        if percentile >= 50:
            age_html = f'''
            <div style="background: rgba(255,167,38,0.1); padding: 0.8rem 1rem; border-radius: 0 6px 6px 0; border-left: 4px solid #ffa726; color: #ddd; font-size: 0.9rem;">
                Your risk profile is higher than approximately {percentile}% of individuals in the {age-5} to {age+5} age range.
            </div>
            '''
        else:
            age_html = f'''
            <div style="background: rgba(102,187,106,0.1); padding: 0.8rem 1rem; border-radius: 0 6px 6px 0; border-left: 4px solid #66bb6a; color: #ddd; font-size: 0.9rem;">
                Your risk profile is lower than approximately {100-percentile}% of individuals in the {age-5} to {age+5} age range.
            </div>
            '''
        st.markdown(age_html, unsafe_allow_html=True)
        
        st.write("---")
        
        # RISK CONTRIBUTORS WITH MODIFIABILITY
        st.subheader("Risk Contributors")
        st.write("Factors influencing your risk profile, with modifiability indicated:")
        
        contributors = get_risk_contributors(
            bmi, bp_cat, chol_map[cholesterol], gluc_map[gluc],
            payload["smoke"], payload["alco"], payload["active"], age
        )
        
        if contributors:
            for level, factor, desc, modifiable in contributors:
                if level == "HIGH":
                    st.markdown(f'<div class="priority-high"><strong>{factor}</strong>: {desc}<br><em style="font-size:0.85rem; color:#888;">{modifiable}</em></div>', unsafe_allow_html=True)
                elif level == "MEDIUM":
                    st.markdown(f'<div class="priority-medium"><strong>{factor}</strong>: {desc}<br><em style="font-size:0.85rem; color:#888;">{modifiable}</em></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="priority-low"><strong>{factor}</strong>: {desc}<br><em style="font-size:0.85rem; color:#888;">{modifiable}</em></div>', unsafe_allow_html=True)
        else:
            st.success("No significant risk factors identified. Maintain your healthy lifestyle!")
        
        st.write("---")
        
        # RECOMMENDATIONS WITH MEASURABLE TARGETS
        st.subheader("Personalized Action Plan")
        
        high_p, med_p, long_p = get_prioritized_recommendations(
            bmi, bp_cat, chol_map[cholesterol], gluc_map[gluc],
            payload["smoke"], payload["alco"], payload["active"], ap_hi
        )
        
        # High Priority Section
        st.markdown("""
        <div style="margin-top: 1rem; margin-bottom: 0.5rem; color: #ff6b6b; font-weight: 600;">High Priority - Start This Week</div>
        """, unsafe_allow_html=True)
        for rec, target in high_p:
            st.markdown(f'''
            <div style="border-left: 4px solid #ff4b4b; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,75,75,0.1); border-radius: 0 4px 4px 0;">
                <strong>{rec}</strong><br>
                <em style="font-size:0.85rem; color:#888;">{target}</em>
            </div>
            ''', unsafe_allow_html=True)
        
        # Medium Priority Section
        st.markdown("""
        <div style="margin-top: 1.5rem; margin-bottom: 0.5rem; color: #ffa726; font-weight: 600;">Medium Priority - Next 30 Days</div>
        """, unsafe_allow_html=True)
        for rec, target in med_p:
            st.markdown(f'''
            <div style="border-left: 4px solid #ffa726; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,167,38,0.1); border-radius: 0 4px 4px 0;">
                <strong>{rec}</strong><br>
                <em style="font-size:0.85rem; color:#888;">{target}</em>
            </div>
            ''', unsafe_allow_html=True)
        
        # Long-term Section
        st.markdown("""
        <div style="margin-top: 1.5rem; margin-bottom: 0.5rem; color: #66bb6a; font-weight: 600;">Long-term Goals - Ongoing</div>
        """, unsafe_allow_html=True)
        for rec, target in long_p:
            st.markdown(f'''
            <div style="border-left: 4px solid #66bb6a; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(102,187,106,0.1); border-radius: 0 4px 4px 0;">
                <strong>{rec}</strong><br>
                <em style="font-size:0.85rem; color:#888;">{target}</em>
            </div>
            ''', unsafe_allow_html=True)
        
        st.write("---")
        
        # WHAT-IF ANALYSIS
        st.subheader("Potential Impact of Lifestyle Changes")
        st.write("Short-term estimated risk with specific modifications:")
        
        scenarios_shown = 0
        
        # Instead of columns, use same card style as risk contributors
        if payload["smoke"] == 1:
            new_prob = calculate_what_if(payload, smoke_change=0)
            if new_prob is not None:
                reduction = probability * 100 - new_prob
                if reduction > 1:
                    st.markdown(f'''
                    <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
                        <strong>If you quit smoking</strong>: {int(new_prob)}% (↓{int(reduction)}% reduction)<br>
                        <em style="font-size:0.85rem; color:#888;">Significant long-term cardiovascular benefit</em>
                    </div>
                    ''', unsafe_allow_html=True)
                    scenarios_shown += 1
                else:
                    st.markdown('''
                    <div style="border-left: 4px solid #66bb6a; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(102,187,106,0.1); border-radius: 0 4px 4px 0;">
                        <strong>Quit Smoking</strong><br>
                        <em style="font-size:0.85rem; color:#888;">Reduces long-term cardiovascular risk by up to 50% within 5 years</em>
                    </div>
                    ''', unsafe_allow_html=True)
                    scenarios_shown += 1
        
        if payload["active"] == 0:
            new_prob = calculate_what_if(payload, active_change=1)
            if new_prob is not None:
                reduction = probability * 100 - new_prob
                if reduction > 1:
                    st.markdown(f'''
                    <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
                        <strong>If you become active</strong>: {int(new_prob)}% (↓{int(reduction)}% reduction)<br>
                        <em style="font-size:0.85rem; color:#888;">Regular physical activity improves cardiovascular function</em>
                    </div>
                    ''', unsafe_allow_html=True)
                    scenarios_shown += 1
                else:
                    st.markdown('''
                    <div style="border-left: 4px solid #66bb6a; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(102,187,106,0.1); border-radius: 0 4px 4px 0;">
                        <strong>Regular Physical Activity</strong><br>
                        <em style="font-size:0.85rem; color:#888;">Reduces heart disease risk by 30-40% through improved cardiovascular function</em>
                    </div>
                    ''', unsafe_allow_html=True)
                    scenarios_shown += 1
        
        if ap_hi > 130:
            new_prob = calculate_what_if(payload, bp_reduction=15)
            if new_prob is not None:
                reduction = probability * 100 - new_prob
                if reduction > 1:
                    st.markdown(f'''
                    <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
                        <strong>If BP reduced by 15 mmHg</strong>: {int(new_prob)}% (↓{int(reduction)}% reduction)<br>
                        <em style="font-size:0.85rem; color:#888;">Blood pressure management is highly effective</em>
                    </div>
                    ''', unsafe_allow_html=True)
                    scenarios_shown += 1
                else:
                    st.markdown('''
                    <div style="border-left: 4px solid #66bb6a; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(102,187,106,0.1); border-radius: 0 4px 4px 0;">
                        <strong>Lower Blood Pressure</strong><br>
                        <em style="font-size:0.85rem; color:#888;">Reducing BP by 10-15 mmHg significantly lowers stroke and heart attack risk</em>
                    </div>
                    ''', unsafe_allow_html=True)
                    scenarios_shown += 1
        
        # COMBINED SCENARIO
        if payload["smoke"] == 1 and ap_hi > 130:
            combined_prob = calculate_what_if(payload, smoke_change=0, bp_reduction=15)
            if combined_prob is not None:
                combined_reduction = probability * 100 - combined_prob
                if combined_reduction > 2:
                    st.markdown(f'''
                    <div style="border-left: 4px solid #ffa726; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,167,38,0.1); border-radius: 0 4px 4px 0;">
                        <strong>Combined: Quit smoking + BP reduction</strong>: {int(combined_prob)}% (↓{int(combined_reduction)}% reduction)<br>
                        <em style="font-size:0.85rem; color:#888;">Combining multiple lifestyle changes produces compounding health benefits</em>
                    </div>
                    ''', unsafe_allow_html=True)
        
        if scenarios_shown == 0 and not (payload["smoke"] == 1 and ap_hi > 130):
            st.markdown('''
            <div style="border-left: 4px solid #66bb6a; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(102,187,106,0.1); border-radius: 0 4px 4px 0;">
                <strong>Optimal Lifestyle</strong><br>
                <em style="font-size:0.85rem; color:#888;">Your current lifestyle choices are already optimal. Keep up the great work!</em>
            </div>
            ''', unsafe_allow_html=True)
        
        # Long-term benefit note
        st.markdown("""
        <div style="background: rgba(100,180,100,0.1); padding: 0.8rem 1rem; border-radius: 0 6px 6px 0; margin-top: 1rem; border-left: 4px solid #66bb6a;">
            <strong style="color: #66bb6a;">About These Estimates:</strong><br>
            <span style="font-size: 0.9rem; color: #aaa;">
            Lifestyle benefits accumulate over months to years and may not be immediately reflected in short-term risk estimates. 
            Smoking cessation, for example, reduces cardiovascular risk by up to 50% within 5 years.
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        
        # CLINICAL INTERPRETATIONS
        st.subheader("Clinical Interpretations")
        
        bp_interp = output.get("bp_interpretation", "N/A")
        chol_interp = output.get("cholesterol_interpretation", "N/A")
        gluc_interp = output.get("glucose_interpretation", "N/A")
        
        st.markdown(f'''
        <div style="border-left: 4px solid #4a9eff; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(74,158,255,0.1); border-radius: 0 4px 4px 0;">
            <strong>Blood Pressure</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">{bp_interp}</em>
        </div>
        <div style="border-left: 4px solid #ffa726; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,167,38,0.1); border-radius: 0 4px 4px 0;">
            <strong>Cholesterol</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">{chol_interp}</em>
        </div>
        <div style="border-left: 4px solid #66bb6a; padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(102,187,106,0.1); border-radius: 0 4px 4px 0;">
            <strong>Glucose</strong><br>
            <em style="font-size:0.9rem; color:#bbb;">{gluc_interp}</em>
        </div>
        ''', unsafe_allow_html=True)
        
        st.write("---")
        
        # DOWNLOAD PDF REPORT
        st.subheader("Export Report")
        
        try:
            pdf_data = generate_pdf_report(
                payload, output, contributors, (high_p, med_p, long_p)
            )
            
            if isinstance(pdf_data, bytes):
                st.download_button(
                    label="Download Risk Assessment Report (PDF)",
                    data=pdf_data,
                    file_name=f"cardiopredict_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.download_button(
                    label="Download Risk Assessment Report",
                    data=pdf_data,
                    file_name=f"cardiopredict_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        except Exception:
            text_report = generate_text_report(payload, output, contributors, (high_p, med_p, long_p))
            st.download_button(
                label="Download Risk Assessment Report",
                data=text_report,
                file_name=f"cardiopredict_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.write("---")
        
        # TRUST SIGNALS - HOW THIS WORKS
        with st.expander("How is this risk estimated?"):
            st.markdown("""
**About This Assessment**

- Trained on clinical cardiovascular health datasets
- Uses age, vitals, laboratory values, and lifestyle factors
- Designed for prevention and awareness, not diagnosis
- Does not replace clinical judgment or professional evaluation

**What This Tool Does:**
- Provides an educational estimate of relative cardiovascular risk
- Identifies modifiable risk factors you can discuss with your doctor
- Offers evidence-based lifestyle recommendations

**What This Tool Does NOT Do:**
- Diagnose any medical condition
- Replace professional medical evaluation
- Provide emergency medical advice
            """)
        
        # BOTTOM DISCLAIMER (always shown)
        st.markdown("""
        <div style="background: #2d2d2d; padding: 1rem; border-radius: 8px; font-size: 0.85rem; color: #aaa; margin-top: 2rem; text-align: center;">
            <strong>Medical Disclaimer</strong><br>
            This tool is for educational and informational purposes only. It is not intended to be a substitute 
            for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
            or other qualified health provider with any questions you may have regarding a medical condition.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
