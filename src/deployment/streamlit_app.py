import streamlit as st
import requests

API_URL = "https://cardiopredict-heart-disease-risk.onrender.com/predict"

def send_request(payload):
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code != 200:
            return {"error": response.text}
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():

    st.set_page_config(page_title="CardioPredict", layout="wide")

    st.title("CardioPredict – Heart Disease Risk Assessment")
    st.write("Provide your health details for a **personalized cardiovascular risk report**.")

    # ---------------------------------------------------------------
    # USER INPUT FORM (Improved labels + Yes/No)
    # ---------------------------------------------------------------
    with st.form("input_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 100, 45)
            gender = st.selectbox("Gender", ["male", "female"])
            height = st.number_input("Height (cm)", 120, 220, 170)
            weight = st.number_input("Weight (kg)", 30, 200, 75)

        with c2:
            ap_hi = st.number_input("Systolic Blood Pressure (SBP)", 80, 250, 135)
            ap_lo = st.number_input("Diastolic Blood Pressure (DBP)", 40, 200, 85)

            cholesterol = st.selectbox(
                "Cholesterol Level",
                [
                    "Normal (< 200 mg/dL)",
                    "High (200–239 mg/dL)",
                    "Very High (≥ 240 mg/dL)"
                ]
            )
            gluc = st.selectbox(
                "Glucose Level (Fasting)",
                [
                    "Normal (< 100 mg/dL)",
                    "High (100–125 mg/dL)",
                    "Very High (≥ 126 mg/dL)"
                ]
            )

            smoke = st.selectbox("Do you smoke?", ["Yes", "No"])
            alco = st.selectbox("Do you consume alcohol?", ["Yes", "No"])
            active = st.selectbox("Are you physically active?", ["Yes", "No"])

        submitted = st.form_submit_button("Predict Heart Risk")

    # ---------------------------------------------------------------
    # CALL API
    # ---------------------------------------------------------------
    if submitted:

        # Convert Yes/No → Numeric
        smoke_val = 1 if smoke == "Yes" else 0
        alco_val = 1 if alco == "Yes" else 0
        active_val = 1 if active == "Yes" else 0

        # Proper Mapping for Cholesterol
        chol_map = {
            "Normal (< 200 mg/dL)": 1,
            "High (200–239 mg/dL)": 2,
            "Very High (≥ 240 mg/dL)": 3
        }
        chol_val = chol_map[cholesterol]

        # Proper Mapping for Glucose
        gluc_map = {
            "Normal (< 100 mg/dL)": 1,
            "High (100–125 mg/dL)": 2,
            "Very High (≥ 126 mg/dL)": 3
        }
        gluc_val = gluc_map[gluc]

        # Build JSON Payload
        payload = {
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": chol_val,
            "gluc": gluc_val,
            "smoke": smoke_val,
            "alco": alco_val,
            "active": active_val
        }

        output = send_request(payload)

        if "error" in output:
            st.error(f"Backend Error: {output['error']}")
            return

        # ---------------------------------------------------------------
        # SPECIAL CASE (Hypotension, invalid values)
        # ---------------------------------------------------------------
        if output.get("warning") is not None:
            st.error(output["warning"])
            st.info(output["summary"])
            st.warning(output["bp_interpretation"])
            st.write("**Cholesterol:**", output["cholesterol_interpretation"])
            st.write("**Glucose:**", output["glucose_interpretation"])
            for r in output["recommendations"]:
                st.markdown(f"- {r}")
            st.stop()

        # ---------------------------------------------------------------
        # PREDICTION SUMMARY
        # ---------------------------------------------------------------
        st.subheader("Prediction Summary")
        colA, colB = st.columns(2)

        with colA:
            st.metric("Prediction", "Disease" if output["prediction"] == 1 else "No Disease")
            st.metric("Disease Probability", f"{output['probability']*100:.1f}%")
            st.metric("Risk Classification", output["risk_level"])

        with colB:
            rf = output["risk_factors"]
            st.metric("BMI", rf["bmi"])
            st.metric("Blood Pressure Category", rf["blood_pressure_category"])
            st.metric("Combined Risk Score", rf["combined_risk_score"])

        # ---------------------------------------------------------------
        # HEART HEALTH
        # ---------------------------------------------------------------
        if "heart_health" in output:
            st.write("---")
            st.subheader("Heart Health Score")
            heart = output["heart_health"]
            st.metric("Heart Score (0–100)", heart["score"])
            st.info(f"Status: {heart['status']}")

        # ---------------------------------------------------------------
        # SUMMARY TEXT
        # ---------------------------------------------------------------
        st.write("---")
        st.subheader("Detailed Summary")
        st.info(output["summary"])

        # ---------------------------------------------------------------
        # CLINICAL INTERPRETATIONS
        # ---------------------------------------------------------------
        st.write("---")
        st.subheader("Clinical Interpretations")
        st.write(f"**Blood Pressure:** {output['bp_interpretation']}")
        st.write(f"**Cholesterol:** {output['cholesterol_interpretation']}")
        st.write(f"**Glucose:** {output['glucose_interpretation']}")

        # ---------------------------------------------------------------
        # KEY RISK MARKERS
        # ---------------------------------------------------------------
        st.write("---")
        st.subheader("Key Risk Markers")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("BMI", rf["bmi"])
        with c2:
            st.metric("BP Category", rf["blood_pressure_category"])
        with c3:
            st.metric("Combined Score", rf["combined_risk_score"])

        # ---------------------------------------------------------------
        # RECOMMENDATIONS
        # ---------------------------------------------------------------
        st.write("---")
        st.subheader("Personalized Recommendations")
        for r in output["recommendations"]:
            st.markdown(f"- {r}")

if __name__ == "__main__":
    main()
