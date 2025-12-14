"""
CardioPredict - Enhanced Heart Disease Risk Assessment
Industry-Grade Streamlit UI with Trust, Explainability, and Actionable Insights
"""

import streamlit as st
import requests
from datetime import datetime
import io

# Configuration
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="CardioPredict - Heart Risk Assessment",
    page_icon="",
    layout="wide"
)

# Custom CSS for clean professional look
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem;}
    .sub-header {font-size: 1rem; color: #888; margin-bottom: 1.5rem;}
    .risk-context {background: #1e3a5f; padding: 1rem; border-radius: 8px; margin: 1rem 0;}
    .disclaimer {background: #2d2d2d; padding: 1rem; border-radius: 8px; font-size: 0.85rem; color: #aaa; margin-top: 2rem;}
    .priority-high {border-left: 4px solid #ff4b4b; padding-left: 1rem; margin: 0.5rem 0;}
    .priority-medium {border-left: 4px solid #ffa726; padding-left: 1rem; margin: 0.5rem 0;}
    .priority-low {border-left: 4px solid #66bb6a; padding-left: 1rem; margin: 0.5rem 0;}
    .contributor-high {color: #ff4b4b;}
    .contributor-medium {color: #ffa726;}
    .contributor-low {color: #ffeb3b;}
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


def get_risk_contributors(bmi, bp_cat, chol, gluc, smoke, alco, active):
    """
    Analyze and rank risk contributors based on input values.
    Returns list of (level, factor, description) tuples.
    """
    contributors = []
    
    # Blood Pressure
    if bp_cat >= 3:
        contributors.append(("HIGH", "Blood Pressure", f"Stage {bp_cat-1} Hypertension"))
    elif bp_cat == 2:
        contributors.append(("MEDIUM", "Blood Pressure", "Elevated blood pressure"))
    
    # Cholesterol
    if chol == 3:
        contributors.append(("HIGH", "Cholesterol", "Very high cholesterol levels"))
    elif chol == 2:
        contributors.append(("MEDIUM", "Cholesterol", "Above normal cholesterol"))
    
    # BMI
    if bmi >= 30:
        contributors.append(("HIGH", "BMI", f"Obese (BMI: {bmi:.1f})"))
    elif bmi >= 25:
        contributors.append(("MEDIUM", "BMI", f"Overweight (BMI: {bmi:.1f})"))
    
    # Glucose
    if gluc == 3:
        contributors.append(("HIGH", "Glucose", "Very high glucose levels"))
    elif gluc == 2:
        contributors.append(("MEDIUM", "Glucose", "Above normal glucose"))
    
    # Smoking
    if smoke == 1:
        contributors.append(("HIGH", "Smoking", "Active smoker"))
    
    # Alcohol
    if alco == 1:
        contributors.append(("MEDIUM", "Alcohol", "Regular alcohol consumption"))
    
    # Physical Activity
    if active == 0:
        contributors.append(("LOW", "Physical Activity", "Sedentary lifestyle"))
    
    # Sort by severity
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    contributors.sort(key=lambda x: severity_order[x[0]])
    
    return contributors


def get_prioritized_recommendations(contributors, bmi, bp_cat, chol, gluc, smoke, alco, active):
    """Generate prioritized recommendations based on risk contributors."""
    high_priority = []
    medium_priority = []
    long_term = []
    
    # High Priority
    if bp_cat >= 3:
        high_priority.append("Consult a physician immediately for hypertension management")
    if chol == 3:
        high_priority.append("Schedule a lipid panel test and discuss treatment options with your doctor")
    if smoke == 1:
        high_priority.append("Begin a smoking cessation program - this is the fastest way to reduce risk")
    
    # Medium Priority
    if bp_cat == 2:
        medium_priority.append("Monitor blood pressure daily and reduce sodium intake")
    if bmi >= 25:
        medium_priority.append(f"Target weight reduction of {max(5, int((bmi - 24) * 2))} kg through diet and exercise")
    if chol == 2:
        medium_priority.append("Increase dietary fiber and reduce saturated fat intake")
    if gluc >= 2:
        medium_priority.append("Reduce sugar and refined carbohydrate consumption")
    if alco == 1:
        medium_priority.append("Limit alcohol to 1-2 drinks per week maximum")
    
    # Long Term
    if active == 0:
        long_term.append("Begin with 30 minutes of walking daily, gradually increase intensity")
    else:
        long_term.append("Maintain current activity level, consider adding strength training")
    long_term.append("Schedule regular health check-ups every 6 months")
    long_term.append("Monitor blood pressure and blood glucose at home weekly")
    
    # Ensure at least one item in each category
    if not high_priority:
        high_priority.append("Continue current healthy practices and maintain regular check-ups")
    if not medium_priority:
        medium_priority.append("Focus on maintaining a balanced diet rich in vegetables and lean proteins")
    
    return high_priority, medium_priority, long_term


def get_age_percentile(age, probability):
    """Estimate risk percentile compared to age group."""
    # Simplified estimation based on age and probability
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


def get_risk_context(risk_level, probability):
    """Get contextual explanation for risk level."""
    contexts = {
        "Very High": {
            "meaning": "This score indicates a significantly elevated cardiovascular risk compared to the average adult.",
            "action": "Medical evaluation is strongly advised within the next 2 weeks.",
            "note": "This is a risk assessment, not a diagnosis. Many individuals with elevated risk never develop heart disease when they take appropriate action."
        },
        "High": {
            "meaning": "Your cardiovascular risk is above average for your demographic group.",
            "action": "Consider scheduling a check-up with your healthcare provider within the next month.",
            "note": "Lifestyle modifications can significantly reduce this risk over time."
        },
        "Moderate": {
            "meaning": "Your risk level is within the expected range but shows some areas for improvement.",
            "action": "Focus on the recommended lifestyle changes and monitor your health metrics.",
            "note": "Regular monitoring and healthy habits can keep your risk from increasing."
        },
        "Low": {
            "meaning": "Your cardiovascular risk appears to be below average.",
            "action": "Continue your current healthy practices.",
            "note": "Maintain regular check-ups to ensure continued heart health."
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
    """Generate a text-based report (simplified PDF alternative)."""
    report = []
    report.append("=" * 60)
    report.append("CARDIOPREDICT - HEART DISEASE RISK ASSESSMENT REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    
    report.append("PATIENT INFORMATION")
    report.append("-" * 40)
    report.append(f"Age: {inputs['age']} years")
    report.append(f"Gender: {inputs['gender'].title()}")
    report.append(f"Height: {inputs['height']} cm")
    report.append(f"Weight: {inputs['weight']} kg")
    report.append(f"Blood Pressure: {inputs['ap_hi']}/{inputs['ap_lo']} mmHg")
    report.append("")
    
    report.append("RISK ASSESSMENT")
    report.append("-" * 40)
    report.append(f"Risk Probability: {output['probability']*100:.1f}%")
    report.append(f"Risk Classification: {output['risk_level']}")
    report.append(f"BMI: {output['risk_factors'].get('bmi', 'N/A')}")
    report.append("")
    
    report.append("KEY RISK CONTRIBUTORS")
    report.append("-" * 40)
    for level, factor, desc in contributors:
        report.append(f"[{level}] {factor}: {desc}")
    report.append("")
    
    high_p, med_p, long_p = recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("High Priority:")
    for r in high_p:
        report.append(f"  * {r}")
    report.append("\nMedium Priority:")
    for r in med_p:
        report.append(f"  * {r}")
    report.append("\nLong-term Goals:")
    for r in long_p:
        report.append(f"  * {r}")
    report.append("")
    
    report.append("=" * 60)
    report.append("DISCLAIMER")
    report.append("This assessment is for educational purposes only.")
    report.append("It is not a substitute for professional medical advice.")
    report.append("Please consult a healthcare provider for diagnosis.")
    report.append("=" * 60)
    
    return "\n".join(report)


def main():
    # Header
    st.markdown('<p class="main-header">CardioPredict - Heart Disease Risk Assessment</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered cardiovascular risk analysis with personalized recommendations</p>', unsafe_allow_html=True)
    
    # Input Form
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.number_input("Age (years)", 18, 100, 45)
            gender = st.selectbox("Gender", ["male", "female"])
            height = st.number_input("Height (cm)", 120, 220, 170)
            weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0, step=0.5)
        
        with col2:
            st.subheader("Health Metrics")
            ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", 80, 250, 135)
            ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 200, 85)
            
            cholesterol = st.selectbox(
                "Cholesterol Level",
                ["Normal (< 200 mg/dL)", "Above Normal (200-239 mg/dL)", "High (>= 240 mg/dL)"]
            )
            gluc = st.selectbox(
                "Glucose Level (Fasting)",
                ["Normal (< 100 mg/dL)", "Above Normal (100-125 mg/dL)", "High (>= 126 mg/dL)"]
            )
        
        st.subheader("Lifestyle Factors")
        col3, col4, col5 = st.columns(3)
        with col3:
            smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
        with col4:
            alco = st.selectbox("Do you consume alcohol regularly?", ["No", "Yes"])
        with col5:
            active = st.selectbox("Are you physically active?", ["Yes", "No"])
        
        submitted = st.form_submit_button("Analyze Heart Risk", use_container_width=True)
    
    # Process Submission
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
        
        # Store in session for What-If analysis
        st.session_state["base_payload"] = payload
        
        with st.spinner("Analyzing your cardiovascular risk..."):
            output = send_request(payload)
        
        if "error" in output:
            st.error(f"Error: {output['error']}")
            return
        
        if output.get("warning"):
            st.warning(output["warning"])
            return
        
        # Calculate derived values
        bmi = weight / ((height / 100) ** 2)
        bp_cat = output.get("risk_factors", {}).get("bp_category", 2)
        if isinstance(bp_cat, str):
            bp_cat = 2
        
        probability = output["probability"]
        risk_level = output["risk_level"]
        
        st.write("---")
        
        # ===== SECTION 1: PREDICTION RESULTS =====
        st.subheader("Prediction Results")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            pred_text = "Elevated Risk Detected" if output["prediction"] == 1 else "Lower Risk Profile"
            st.metric("Assessment", pred_text)
        with c2:
            # Improved probability presentation
            prob_pct = probability * 100
            if prob_pct >= 70:
                prob_text = f"{prob_pct:.1f}% (Significantly Elevated)"
            elif prob_pct >= 50:
                prob_text = f"{prob_pct:.1f}% (Moderately Elevated)"
            else:
                prob_text = f"{prob_pct:.1f}% (Within Expected Range)"
            st.metric("Risk Probability", prob_text)
        with c3:
            st.metric("Risk Classification", risk_level)
        
        # ===== SECTION 2: RISK CONTEXT (CRITICAL) =====
        context = get_risk_context(risk_level, probability)
        
        st.markdown("##### Understanding Your Risk Level")
        st.info(f"""
**What does "{risk_level}" mean?**

{context['meaning']}

**Recommended Action:** {context['action']}

**Important Note:** {context['note']}
        """)
        
        # ===== SECTION 3: AGE-BASED COMPARISON =====
        percentile = get_age_percentile(age, probability)
        
        st.markdown("##### Compared to Your Age Group")
        if percentile >= 75:
            st.warning(f"Your estimated risk is higher than approximately {percentile}% of individuals in your age group ({age-5} to {age+5} years).")
        elif percentile >= 50:
            st.info(f"Your estimated risk is higher than approximately {percentile}% of individuals in your age group ({age-5} to {age+5} years).")
        else:
            st.success(f"Your estimated risk is lower than approximately {100-percentile}% of individuals in your age group ({age-5} to {age+5} years).")
        
        st.write("---")
        
        # ===== SECTION 4: TOP RISK CONTRIBUTORS =====
        st.subheader("Top Contributors to Your Risk")
        
        contributors = get_risk_contributors(
            bmi, bp_cat, chol_map[cholesterol], gluc_map[gluc],
            payload["smoke"], payload["alco"], payload["active"]
        )
        
        if contributors:
            for level, factor, desc in contributors:
                if level == "HIGH":
                    st.markdown(f'<div class="priority-high"><strong>[HIGH]</strong> {factor}: {desc}</div>', unsafe_allow_html=True)
                elif level == "MEDIUM":
                    st.markdown(f'<div class="priority-medium"><strong>[MEDIUM]</strong> {factor}: {desc}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="priority-low"><strong>[LOW]</strong> {factor}: {desc}</div>', unsafe_allow_html=True)
        else:
            st.success("No significant risk factors identified. Maintain your healthy lifestyle!")
        
        st.write("---")
        
        # ===== SECTION 5: PRIORITIZED RECOMMENDATIONS =====
        st.subheader("Personalized Action Plan")
        
        high_p, med_p, long_p = get_prioritized_recommendations(
            contributors, bmi, bp_cat, chol_map[cholesterol], gluc_map[gluc],
            payload["smoke"], payload["alco"], payload["active"]
        )
        
        st.markdown("**High Priority (Start This Week)**")
        for rec in high_p:
            st.markdown(f"- {rec}")
        
        st.markdown("**Medium Priority (Next 30 Days)**")
        for rec in med_p:
            st.markdown(f"- {rec}")
        
        st.markdown("**Long-term Goals (Ongoing)**")
        for rec in long_p:
            st.markdown(f"- {rec}")
        
        st.write("---")
        
        # ===== SECTION 6: WHAT-IF ANALYSIS =====
        st.subheader("What-If Scenario Analysis")
        st.write("See how lifestyle changes could affect your risk:")
        
        wif_col1, wif_col2, wif_col3 = st.columns(3)
        
        with wif_col1:
            if payload["smoke"] == 1:
                new_prob = calculate_what_if(payload, smoke_change=0)
                if new_prob:
                    reduction = probability * 100 - new_prob
                    st.metric("If you quit smoking", f"{new_prob:.1f}%", f"-{reduction:.1f}%")
        
        with wif_col2:
            if payload["active"] == 0:
                new_prob = calculate_what_if(payload, active_change=1)
                if new_prob:
                    reduction = probability * 100 - new_prob
                    st.metric("If you become active", f"{new_prob:.1f}%", f"-{reduction:.1f}%")
        
        with wif_col3:
            if ap_hi > 130:
                new_prob = calculate_what_if(payload, bp_reduction=15)
                if new_prob:
                    reduction = probability * 100 - new_prob
                    st.metric("If BP reduced by 15 mmHg", f"{new_prob:.1f}%", f"-{reduction:.1f}%")
        
        st.write("---")
        
        # ===== SECTION 7: CLINICAL INTERPRETATIONS =====
        st.subheader("Clinical Interpretations")
        
        int_col1, int_col2, int_col3 = st.columns(3)
        with int_col1:
            st.markdown(f"**Blood Pressure**")
            st.write(output.get("bp_interpretation", "N/A"))
        with int_col2:
            st.markdown(f"**Cholesterol**")
            st.write(output.get("cholesterol_interpretation", "N/A"))
        with int_col3:
            st.markdown(f"**Glucose**")
            st.write(output.get("glucose_interpretation", "N/A"))
        
        st.write("---")
        
        # ===== SECTION 8: DOWNLOAD REPORT =====
        st.subheader("Export Report")
        
        report_text = generate_pdf_report(
            payload, output, contributors, (high_p, med_p, long_p)
        )
        
        st.download_button(
            label="Download Risk Assessment Report",
            data=report_text,
            file_name=f"cardiopredict_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # ===== SECTION 9: MEDICAL DISCLAIMER =====
        st.markdown("""
<div class="disclaimer">
<strong>Medical Disclaimer</strong><br>
This tool is for educational and informational purposes only. It is not intended to be a substitute 
for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
or other qualified health provider with any questions you may have regarding a medical condition. 
Never disregard professional medical advice or delay in seeking it because of something you have 
read or seen in this application.
</div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
