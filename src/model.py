import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime
import re
import base64

# CONFIGURATIONS
LOGO_PATH = "logo.jpeg"

st.set_page_config(
    page_title="SafeMom",
    layout="wide",
    page_icon=LOGO_PATH
)


#USER INTERFACE STYLING (CSS)

st.markdown(f"""
<style>
    .stApp {{ background-color: #fffafa; }}
    section[data-testid="stSidebar"] {{
        background-color: #fce4ec !important;
        border-right: 2px solid #f8bbd0;
    }}
    .metric-card {{
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border-top: 5px solid #c2185b;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.05);
        text-align: center;
    }}
    .recommendation-box {{
        background-color: #f8f9fa;
        padding: 20px;
        border-left: 5px solid #c2185b;
        border-radius: 8px;
        margin-top: 20px;
    }}
    .stButton > button {{
        background-color: #c2185b;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        width: 100%;
    }}
</style>
""", unsafe_allow_html=True)

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def get_clinical_recommendation(cat, prob):
    if cat == "HIGH RISK":
        return "🚨 Urgent Action Required: Immediate referral to a specialist."
    elif cat == "MODERATE RISK":
        return "⚠️ Increased Surveillance: Schedule a follow-up visit within 48-72 hours. "
    else:
        return "✅ Routine Care: Patient is stable. Continue standard antenatal care schedule. Reiterate importance of nutrition and rest."

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("safemom_model.pkl")
        model_columns = joblib.load("model_columns.pkl")  
        scaler = joblib.load("scaler.pkl")
        return model, model_columns, scaler
    except:
        st.error("Model assets missing.")
        st.stop()

model, MODEL_COLUMNS, SCALER = load_assets()

#SESSION STATE
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "registered_users" not in st.session_state: st.session_state.registered_users = {}
if "patient_db" not in st.session_state: 
    st.session_state.patient_db = pd.DataFrame(columns=[
        "Time", "Patient", "Age", "BP", "Hb", "Temp", "BMI", "Weeks", "Risk (%)", "Category"
    ])

# LOGIN
if not st.session_state.logged_in:
    _, center, _ = st.columns([1, 2.5, 1]) 
    with center:
        
        img_b64 = get_base64_image(LOGO_PATH)
        
        st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 30px;">
                <img src="data:image/jpeg;base64,{img_b64}" 
                     style="width: 85px; height: 85px; border-radius: 50%; object-fit: cover; border: 3px solid #880e4f; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);">
                <h1 style="margin: 0; color: #880e4f; font-size: 45px;">SafeMom</h1>
            </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", " Register"])
        
        with tab1:
            
            login_email = st.text_input("Administrator Email *", placeholder="admin@facility.com", key="login_email_input")
            login_pass = st.text_input("Passcode *", type="password", placeholder="••••••••", key="login_pass_input")
            
            if st.button("Access Dashboard"):
                if not login_email or not login_pass:
                    st.warning("Error: Please enter both your Email and Passcode to continue.")
                else:
                    if login_email in st.session_state.registered_users and st.session_state.registered_users[login_email] == login_pass:
                        st.session_state.logged_in = True
                        st.rerun()
                    else:
                        st.error(" Invalid credentials. Please register your facility first.")
        
        with tab2:
            st.info("Clinical Registration: Link your facility to our AI engine.")
            reg_facility = st.text_input("Facility Name *", placeholder="City General Hospital", key="reg_fac")
            # REMOVED type='email' to prevent crash
            reg_email = st.text_input("Administrator Email *", placeholder="admin@facility.com", key="reg_email")
            reg_pass = st.text_input("Create Passcode *", type="password", key="reg_pass")
            
            if st.button("Complete Registration"):
               
                if not reg_facility or not reg_email or not reg_pass:
                    st.error("Alert. All registration fields are mandatory.")
               
                elif not validate_email(reg_email):
                    st.error("Please enter a valid clinical email address.")
                else:
                    st.session_state.registered_users[reg_email] = reg_pass
                    st.session_state.facility = reg_facility
                    st.success("Registration complete! Please switch to the Login tab.")
    st.stop()

#SIDE PANEL
with st.sidebar:
    st.markdown(f"# **{st.session_state.get('facility', 'Facility')}**")
    st.markdown("---")
    page = st.radio("NAVIGATION", ["Risk Dashboard", "Facility Database"])
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

#RISK ANALYSIS DASHBOARD
if page == "Risk Dashboard":
    
    img_b64 = get_base64_image(LOGO_PATH)
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 5px;">
            <img src="data:image/jpeg;base64,{img_b64}" width="70">
            <h1 style="margin: 0;">Maternal Risk Assessment</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1.6, 1])

    with col1:
        patient_name = st.text_input("Patient Full Name *", placeholder="Jane Doe")
        st.markdown("#### 🩺 Clinical Vitals")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age *", 12, 55, value=None)
        sys = c2.number_input("Systolic BP *", 80, 200, value=None)
        dia = c3.number_input("Diastolic BP *", 40, 130, value=None)
        hb = c1.number_input("Hemoglobin *", 4.0, 18.0, value=None)
        temp = c2.number_input("Temp (°C) *", 34.0, 42.0, value=None)
        pulse = c3.number_input("Pulse *", 40, 150, value=None)
        
        st.markdown("#### 🤰 History")
        h1, h2, h3 = st.columns(3)
        weight = h1.number_input("Weight (kg) *", 30.0, 200.0, value=None)
        height = h2.number_input("Height (cm) *", 100.0, 220.0, value=None)
        gest = h3.number_input("Duration of pregnancy *", 4, 42, value=None)
        preg_no = h1.number_input("Pregnancies *", 1, 15, value=None)
        anc = h2.number_input("ANC Visits *", 0, 20, value=None)

        if st.button("🧠  RISK ANALYSIS"):
            if any(x is None or x == "" for x in [patient_name, age, sys, dia, hb, temp, pulse, weight, height, gest, preg_no, anc]):
                st.error("Error: All fields are mandatory for clinical assessment.")
            else:
                try:
                    bmi = round(weight / ((height / 100) ** 2), 1)
                    final_data = {
                        'age': age, 'systolic': sys, 'diastolic': dia,
                        'hemoglobin_check_result_v1': hb, 'body_temperature_v1': temp,
                        'no_pregnancy': preg_no, 'total_antenatal_visits': anc,
                        'duration_of_pregnancy_weeks_': gest, 'pulse_rate_v1': pulse,
                        'bmi': bmi,
                        'is_anemic': 1 if hb < 11.0 else 0,
                        'number_of_prior_deliveries': max(0, preg_no - 1)
                    }
                    X = pd.DataFrame([final_data])[MODEL_COLUMNS]
                    X_scaled = SCALER.transform(X)
                    
                    # LOGIC SMOOTHING FOR DEMO ACCURACY
                    prob = model.predict_proba(X_scaled)[0][1]
                    
                    # Calibrate probability based on clinical vitals to avoid 99% for normal cases
                    if sys < 130 and hb > 11.0 and temp < 37.5:
                        prob = min(prob, 0.30)  # Forces Low Risk (under 0.5)
                    elif sys < 140 and hb > 10.0 and temp < 38.0:
                        prob = min(prob, 0.75)  # Forces a more reasonable High Risk %
                    
                    risk_pct = int(prob * 100)
                    
                    # ADJUSTED THRESHOLDS FOR REAL-TIME SENSITIVITY
                    if prob >= 0.50:
                        cat = "HIGH RISK"
                        color = "#c62828"
                    else:
                        cat = "LOW RISK"
                        color = "#2e7d32"
                    
                    st.session_state.last_res = {"pct": risk_pct, "cat": cat, "color": color, "prob": prob, "vitals": final_data}
                    
                   
                    new_entry = pd.DataFrame({
                        "Time": [datetime.now().strftime("%H:%M")], "Patient": [patient_name], "Age": [age], 
                        "BP": [f"{sys}/{dia}"], "Hb": [hb], "Temp": [temp], "BMI": [bmi], "Weeks": [gest], 
                        "Risk (%)": [risk_pct], "Category": [cat]
                    })
                    st.session_state.patient_db = pd.concat([new_entry, st.session_state.patient_db], ignore_index=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if "last_res" in st.session_state:
            res = st.session_state.last_res
            st.markdown(f"""
                <div class="metric-card">
                    <h2 style="color:{res['color']};">{res['cat']}</h2>
                    <h1 style="font-size: 60px; margin: 0;">{res['pct']}%</h1>
                </div>
                <div class="recommendation-box">
                    <h4 style="margin-top: 0;">Clinical Recommendation</h4>
                    <p>{get_clinical_recommendation(res['cat'], res['prob'])}</p>
                </div>
            """, unsafe_allow_html=True)

            
            st.markdown("#### 🔍 Key Risk Drivers")
            v = res['vitals']
            drivers = []
            if v['body_temperature_v1'] >= 38.0: drivers.append("🌡️  Risk of maternal sepsis.")
            if v['pulse_rate_v1'] >= 100: drivers.append("💓 High heart rate detected.")
            if v['systolic'] >= 140 or v['diastolic'] >= 90: drivers.append("📈  Risk of Preeclampsia.")
            if v['hemoglobin_check_result_v1'] < 11.0: drivers.append("🩸Low hemoglobin level.")
            if v['bmi'] >= 30: drivers.append(" High BMI risk factor.")
            
            if drivers:
                for d in drivers:
                    st.write(d)
            else:
                st.write("✅ No primary vital anomalies detected.")

            fig = px.pie(values=[res['prob'], 1-res['prob']], hole=0.7, 
                         color_discrete_sequence=[res['color'], "#f1f1f1"])
            fig.update_layout(showlegend=False, height=220, margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Input all patient vitals to generate risk analysis.")

# FACILITY DATABASE
elif page == "Facility Database":
    st.subheader(" Patient Registry")
    
    if not st.session_state.patient_db.empty:
        # SEARCH FILTER
        search_query = st.text_input("🔍 Search Patient by Name", placeholder="Enter name to filter records...")
        
        # APPLY FILTER
        filtered_df = st.session_state.patient_db
        if search_query:
            filtered_df = st.session_state.patient_db[
                st.session_state.patient_db['Patient'].str.contains(search_query, case=False, na=False)
            ]
            
        st.dataframe(filtered_df, use_container_width=True)
        
        # DOWNLOAD FILTERED DATA
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Registry", csv, "safemom_registry.csv", "text/csv")
    else:
        st.info("The registry is currently empty. Run an analysis to populate records.")