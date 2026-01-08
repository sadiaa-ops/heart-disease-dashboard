import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Heart Disease Risk Assessment | Bangladesh",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DARK THEME CSS WITH PERFECT CONTRAST
# ============================================
st.markdown("""
    <style>
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Sidebar - Dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00d9ff !important;
    }
    
    /* Radio buttons in sidebar */
    .stRadio > label {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 12px;
        border-radius: 8px;
        margin: 5px 0;
    }
    
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Main content - Light text */
    h1, h2, h3, h4, h5, h6 {
        color: #00d9ff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    p, li, span, div {
        color: #e8e8e8 !important;
    }
    
    /* Input labels - Bright and readable */
    label {
        color: #00d9ff !important;
        font-weight: 600;
        font-size: 1.05rem !important;
    }
    
    /* Content cards - Dark with light text */
    .content-card {
        background: rgba(15, 52, 96, 0.6);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(0, 217, 255, 0.2);
        margin-bottom: 20px;
    }
    
    .content-card * {
        color: #e8e8e8 !important;
    }
    
    .content-card h2,
    .content-card h3 {
        color: #00d9ff !important;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: rgba(15, 52, 96, 0.8) !important;
        color: #e8e8e8 !important;
        border-radius: 10px;
        border-left: 4px solid #00d9ff;
    }
    
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.2) !important;
        border-left: 4px solid #10b981 !important;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.2) !important;
        border-left: 4px solid #f59e0b !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.2) !important;
        border-left: 4px solid #ef4444 !important;
    }
    
    /* Buttons - Vibrant */
    .stButton>button {
        background: linear-gradient(90deg, #00d9ff 0%, #0099cc 100%);
        color: #000000;
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 217, 255, 0.6);
        background: linear-gradient(90deg, #00ffff 0%, #00ccff 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d9ff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e8e8e8 !important;
        font-size: 1.1rem !important;
    }
    
    /* Input fields - Dark theme */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        background-color: rgba(22, 33, 62, 0.8) !important;
        color: #e8e8e8 !important;
        border: 1px solid rgba(0, 217, 255, 0.3) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(15, 52, 96, 0.6) !important;
        color: #00d9ff !important;
        border-radius: 8px;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(15, 52, 96, 0.4) !important;
        border: 1px solid rgba(0, 217, 255, 0.2);
    }
    
    /* Tables and dataframes */
    [data-testid="stDataFrame"] {
        background-color: rgba(15, 52, 96, 0.6) !important;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    """Load all models and metadata"""
    try:
        # Clinical model (UCI dataset) - Logistic Regression with StandardScaler
        clinical_model = pickle.load(open("models/heart_disease_model.pkl", "rb"))
        
        # Get feature names
        clinical_features_raw = pickle.load(open("models/feature_names.pkl", "rb"))
        if hasattr(clinical_features_raw, 'tolist'):
            clinical_features = clinical_features_raw.tolist()
        else:
            clinical_features = list(clinical_features_raw)
        
        return {
            "clinical_model": clinical_model,
            "clinical_features": clinical_features
        }
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please ensure clinical model files are in the 'models/' directory")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.exception(e)
        st.stop()

# Load models once at startup
models = load_models()

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.markdown("### 🫀 Assessment Portal")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Choose Assessment Type",
    ["🏠 Home", "🇧🇩 Bangladesh CVD Insights", "🏥 Clinical Diagnosis"],
    help="Select based on available information"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**ℹ️ About**")
st.sidebar.info("""
**🧠 Dual-Purpose System**

🇧🇩 **Bangladesh CVD Insights**
- Educational dashboard
- Population health patterns
- Risk factor analysis
- Based on 1,529 patients from CAIR-CVD 2025

🏥 **Clinical Diagnostic Model (UCI)**
- Model: Calibrated Logistic Regression
- F1 Score: **83.5%**
- ROC AUC: **0.945**
- Brier Score: **0.102** (excellent calibration)
- Realistic probability predictions

**⚠️ Note:**  
Bangladesh section shows population patterns,  
not individual risk predictions.
""")

# ============================================
# HOME PAGE
# ============================================
if app_mode == "🏠 Home":
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 10px;'>🫀 Bangladesh Heart Disease AI</h1>
            <p style='font-size: 1.5rem; color: #00d9ff;'>Dual-Engine Risk Assessment System</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='content-card'>
            <h2>📊 Why This Matters</h2>
            <p style='font-size: 1.2rem; line-height: 1.8;'>
                Cardiovascular Disease (CVD) accounts for <strong style='color: #00d9ff;'>28% of all deaths in Bangladesh</strong>, 
                making it the <strong>leading cause of mortality</strong>. Early detection through AI-powered screening 
                can save lives and reduce healthcare burden.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class='content-card'>
            <h3>Impact of CVD in Bangladesh</h3>
            <p style='color: #b8b8b8; margin-bottom: 20px; font-size: 1.05rem;'>
                This visualization shows that nearly <strong style='color: #00d9ff;'>1 in 3 deaths</strong> 
                in Bangladesh are CVD-related, emphasizing the critical need for accessible screening tools.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    fig = px.pie(
        values=[28, 72], 
        names=['CVD Deaths (28%)', 'Other Causes (72%)'],
        hole=0.4,
        color_discrete_sequence=['#ef4444', '#3b82f6']
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=16, color='#e8e8e8')
    )
    st.plotly_chart(fig, width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: #2d3748; 
                        padding: 30px; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                        border: 2px solid #00d9ff;
                        min-height: 420px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;'>
                <div>
                    <h3 style='color: #00d9ff !important;'>🇧🇩 Bangladesh CVD Insights</h3>
                    <p style='font-size: 1.1rem; line-height: 1.8; color: #e8e8e8 !important;'>
                        <strong>Educational Dashboard</strong><br>
                        Explore population health patterns:
                    </p>
                    <ul style='font-size: 1.05rem; line-height: 2; color: #e8e8e8 !important;'>
                        <li>Risk factor prevalence</li>
                        <li>Age & gender patterns</li>
                        <li>Lifestyle impact analysis</li>
                    </ul>
                </div>
                <p style='margin-top: 15px; margin-bottom: 0; font-size: 1rem; color: #e8e8e8 !important;'>
                    <strong>Data Source:</strong> CAIR-CVD 2025<br>
                    <strong>Patients:</strong> 1,529 from Bangladesh
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: #2d3748; 
                        padding: 30px; border-radius: 15px; 
                        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                        border: 2px solid #a78bfa;
                        min-height: 420px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;'>
                <div>
                    <h3 style='color: #a78bfa !important;'>🏥 Clinical Diagnosis</h3>
                    <p style='font-size: 1.1rem; line-height: 1.8; color: #e8e8e8 !important;'>
                        <strong>For Medical Professionals</strong><br>
                        Requires advanced tests:
                    </p>
                    <ul style='font-size: 1.05rem; line-height: 2; color: #e8e8e8 !important;'>
                        <li>ECG & stress test results</li>
                        <li>Cardiac catheterization</li>
                        <li>Thalassemia screening</li>
                    </ul>
                </div>
                <p style='margin-top: 15px; margin-bottom: 0; font-size: 1rem; color: #e8e8e8 !important;'>
                    <strong>Data Source:</strong> UCI Heart Disease<br>
                    <strong>Model:</strong> Calibrated Logistic Regression                    
                </p>
            </div>
        """, unsafe_allow_html=True)

# ============================================
# BANGLADESH CVD INSIGHTS DASHBOARD
# ============================================
elif app_mode == "🇧🇩 Bangladesh CVD Insights":
    st.markdown("<h1>🇧🇩 Cardiovascular Disease Patterns in Bangladesh</h1>", unsafe_allow_html=True)
    
    st.info("""
    **📊 Educational Dashboard**  
    Explore CVD patterns and risk factors in the Bangladeshi population based on the CAIR-CVD 2025 dataset (1,529 patients).
    This section provides insights into population health, not individual risk predictions.
    """)
    
    # Load insights data
    try:
        import json
        with open("insights/key_insights.json", "r") as f:
            insights = json.load(f)
        with open("insights/demographic_insights.json", "r") as f:
            demographics = json.load(f)
        with open("insights/lifestyle_impact.json", "r") as f:
            lifestyle_impact = json.load(f)
    except FileNotFoundError:
        st.error("⚠️ Insights data not found. Please ensure the 'insights' folder is in your project directory.")
        st.stop()
    
    # ============================================
    # KEY STATISTICS
    # ============================================
    st.markdown("### 📈 Key Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Patients Analyzed",
            f"{insights['dataset_info']['total_patients']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            "CVD Prevalence",
            insights['dataset_info']['cvd_prevalence'],
            delta="High prevalence cohort",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Most Affected Age",
            insights['demographics']['most_affected_age_group'],
            delta=f"{demographics['age_groups']['50-60']:.1f}% prevalence"
        )
    
    with col4:
        st.metric(
            "Gender Difference",
            "Minimal",
            delta=f"F: {insights['demographics']['female_prevalence']} | M: {insights['demographics']['male_prevalence']}"
        )
    
    st.markdown("---")
    
    # ============================================
    # RISK FACTORS COMPARISON
    # ============================================
    st.markdown("### 🎯 Major Risk Factors: CVD Patients vs Healthy Individuals")
    
    st.markdown("""
    <div style='background: rgba(15, 52, 96, 0.6); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='color: #e8e8e8; font-size: 1.05rem; line-height: 1.6;'>
            This chart shows the prevalence of major cardiovascular risk factors among CVD patients (red) 
            compared to healthy individuals (green) in the Bangladesh dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from PIL import Image
        import os
        
        # Try different file extensions
        possible_files = [
            "insights/risk_factors_comparison.png",
            "insights/risk_factors_comparison.jpg",
            "insights/risk_factors_comparison"
        ]
        
        risk_img = None
        for filepath in possible_files:
            if os.path.exists(filepath):
                risk_img = Image.open(filepath)
                break
        
        if risk_img:
            st.image(risk_img, width='stretch')
        else:
            st.error("⚠️ Risk factors chart not found. Please ensure the file is in the insights folder.")
    except Exception as e:
        st.error(f"⚠️ Error loading risk factors chart: {e}")
    
    # Top 3 risk factors
    st.markdown("#### 🔝 Top 3 Risk Factor Differences")
    
    cols = st.columns(3)
    for idx, factor in enumerate(insights['top_risk_factors']):
        with cols[idx]:
            st.markdown(f"""
                <div style='background: rgba(239, 68, 68, 0.2); padding: 15px; border-radius: 10px; 
                            border-left: 4px solid #ef4444; text-align: center;'>
                    <h4 style='color: #00d9ff; margin: 0;'>{factor['name']}</h4>
                    <p style='color: #e8e8e8; font-size: 1.2rem; margin: 10px 0;'>
                        <strong>{factor['difference']}</strong> higher in CVD patients
                    </p>
                    <p style='color: #b8b8b8; font-size: 0.9rem; margin: 0;'>
                        CVD: {factor['cvd_prevalence']} | Healthy: {factor['healthy_prevalence']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # AGE & GENDER PATTERNS
    # ============================================
    st.markdown("### 👥 CVD Prevalence by Age and Gender")
    
    st.markdown("""
    <div style='background: rgba(15, 52, 96, 0.6); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='color: #e8e8e8; font-size: 1.05rem; line-height: 1.6;'>
            CVD prevalence increases with age, peaking in the 50-60 age group at over 90%, then declining slightly 
            in those over 60. Both genders show similar patterns with minimal differences.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        import os
        
        possible_files = [
            "insights/age_gender_patterns.png",
            "insights/age_gender_patterns.jpg",
            "insights/age_gender_patterns"
        ]
        
        age_img = None
        for filepath in possible_files:
            if os.path.exists(filepath):
                age_img = Image.open(filepath)
                break
        
        if age_img:
            st.image(age_img, width='stretch')
        else:
            st.error("⚠️ Age/gender chart not found.")
    except Exception as e:
        st.error(f"⚠️ Error loading age/gender chart: {e}")
    
    # Age group breakdown
    st.markdown("#### 📊 Prevalence by Age Group")
    
    age_data = demographics['age_groups']
    age_df = pd.DataFrame([
        {'Age Group': k, 'CVD Prevalence (%)': v}
        for k, v in age_data.items()
    ])
    
    fig = px.bar(
        age_df,
        x='Age Group',
        y='CVD Prevalence (%)',
        color='CVD Prevalence (%)',
        color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
        text='CVD Prevalence (%)'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e8e8e8'),
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # ============================================
    # LIFESTYLE IMPACT
    # ============================================
    st.markdown("### 💪 Impact of Physical Activity on Health Markers")
    
    st.markdown("""
    <div style='background: rgba(15, 52, 96, 0.6); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='color: #e8e8e8; font-size: 1.05rem; line-height: 1.6;'>
            Physical activity shows measurable benefits on cardiovascular health markers. 
            Individuals with high physical activity levels show improved blood pressure and blood sugar levels 
            compared to those with low activity.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        import os
        
        possible_files = [
            "insights/lifestyle_impact.png",
            "insights/lifestyle_impact.jpg",
            "insights/lifestyle_impact"
        ]
        
        lifestyle_img = None
        for filepath in possible_files:
            if os.path.exists(filepath):
                lifestyle_img = Image.open(filepath)
                break
        
        if lifestyle_img:
            st.image(lifestyle_img, width='stretch')
        else:
            st.error("⚠️ Lifestyle impact chart not found.")
    except Exception as e:
        st.error(f"⚠️ Error loading lifestyle impact chart: {e}")
    
    # Benefits summary
    st.markdown("#### ✅ Benefits of High Physical Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.2); padding: 20px; border-radius: 10px; 
                        border-left: 4px solid #10b981;'>
                <h4 style='color: #10b981; margin-top: 0;'>Blood Pressure Improvements</h4>
        """, unsafe_allow_html=True)
        
        sys_improvement = lifestyle_impact['Systolic BP']['Difference']
        dia_improvement = lifestyle_impact['Diastolic BP']['Difference']
        
        st.write(f"• **Systolic BP:** {sys_improvement:.1f} mmHg lower")
        st.write(f"• **Diastolic BP:** {dia_improvement:.1f} mmHg lower")
        st.write("• Reduces strain on the heart")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.2); padding: 20px; border-radius: 10px; 
                        border-left: 4px solid #10b981;'>
                <h4 style='color: #10b981; margin-top: 0;'>Metabolic Benefits</h4>
        """, unsafe_allow_html=True)
        
        st.write("• Helps maintain healthy weight")
        st.write("• Improves insulin sensitivity")
        st.write("• Reduces diabetes risk")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # INTERACTIVE EXPLORER
    # ============================================
    st.markdown("### 🔍 Interactive Risk Factor Explorer")
    
    st.markdown("""
    <div style='background: rgba(15, 52, 96, 0.6); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='color: #e8e8e8; font-size: 1.05rem; line-height: 1.6;'>
            Explore how different demographic and risk factor combinations affect CVD prevalence in the dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_age = st.selectbox(
            "Select Age Group",
            ["<30", "30-40", "40-50", "50-60", "60+"]
        )
    
    with col2:
        selected_gender = st.selectbox(
            "Select Gender",
            ["Male", "Female"]
        )
    
    # Load age-gender data
    try:
        with open("insights/age_gender_data.json", "r") as f:
            age_gender_data = json.load(f)
        
        gender_code = "M" if selected_gender == "Male" else "F"
        key = f"{selected_age}_{gender_code}"
        
        if key in age_gender_data:
            data = age_gender_data[key]
            
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(236, 72, 153, 0.3)); 
                            padding: 30px; border-radius: 15px; margin-top: 20px; text-align: center;'>
                    <h3 style='color: #00d9ff; margin: 0;'>Selected Group: {selected_age} {selected_gender}s</h3>
                    <div style='margin-top: 20px;'>
                        <div style='display: inline-block; margin: 0 20px;'>
                            <p style='color: #b8b8b8; font-size: 0.9rem; margin: 0;'>CVD Prevalence</p>
                            <p style='color: #00d9ff; font-size: 2.5rem; font-weight: bold; margin: 5px 0;'>
                                {data['prevalence']:.1f}%
                            </p>
                        </div>
                        <div style='display: inline-block; margin: 0 20px;'>
                            <p style='color: #b8b8b8; font-size: 0.9rem; margin: 0;'>Total Patients</p>
                            <p style='color: #e8e8e8; font-size: 2rem; font-weight: bold; margin: 5px 0;'>
                                {data['total_patients']}
                            </p>
                        </div>
                        <div style='display: inline-block; margin: 0 20px;'>
                            <p style='color: #b8b8b8; font-size: 0.9rem; margin: 0;'>CVD Cases</p>
                            <p style='color: #ef4444; font-size: 2rem; font-weight: bold; margin: 5px 0;'>
                                {data['cvd_cases']}
                            </p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    except:
        st.error("⚠️ Age-gender data not found.")
    
    st.markdown("---")
    
    # ============================================
    # KEY TAKEAWAYS
    # ============================================
    st.markdown("### 💡 Key Takeaways")
    
    st.markdown("""
        <div style='background: rgba(59, 130, 246, 0.2); padding: 25px; border-radius: 10px; 
                    border-left: 4px solid #3b82f6;'>
            <h4 style='color: #3b82f6; margin-top: 0;'>Important Insights from Bangladesh CVD Data</h4>
            <ul style='color: #e8e8e8; font-size: 1.05rem; line-height: 1.8;'>
                <li><strong>Family History</strong> is the strongest differentiating factor (13.2% higher in CVD patients)</li>
                <li><strong>Physical Activity</strong> shows clear benefits on blood pressure control</li>
                <li><strong>Middle Age (50-60)</strong> shows the highest CVD prevalence (91.4%)</li>
                <li><strong>Gender differences</strong> are minimal in this population</li>
                <li><strong>Multiple risk factors</strong> often co-occur (clustering effect)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.warning("""
        **⚠️ Data Interpretation Note**
        
        This dataset has 85.6% CVD prevalence, indicating it primarily consists of patients seeking cardiovascular care 
        rather than a representative population sample. The insights show patterns among this clinical population and 
        may not directly reflect risk in the general Bangladeshi population.
        
        For personal health assessment, please consult with healthcare professionals.
    """)

# ============================================
# CLINICAL DIAGNOSIS
# ============================================
else:
    st.markdown("<h1>🏥 Clinical Diagnosis</h1>", unsafe_allow_html=True)
    st.warning(
        "**⚠️ For Healthcare Professionals Only**  \n"
        "Requires laboratory and diagnostic test results."
    )
    
    with st.expander("📚 Parameter Definitions", expanded=False):
        st.markdown("""
        **Chest Pain (cp)**  
        - 1: Typical Angina  
        - 2: Atypical Angina  
        - 3: Non-anginal Pain  
        - 4: Asymptomatic  
        
        **Thalassemia (thal)**  
        - 3: Normal  
        - 6: Fixed Defect  
        - 7: Reversible Defect  
        
        **Major Vessels (ca)**  
        - Number of vessels (0–3) visible via fluoroscopy  
        
        **ST Depression (oldpeak)**  
        - ECG stress-induced ST depression relative to rest
        
        **Slope**
        - 1: Upsloping
        - 2: Flat
        - 3: Downsloping
        """)
    
    st.markdown("---")
    st.markdown("<h3>📋 Clinical Parameters</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🫀 Cardiac**")
        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4],
            format_func=lambda x: ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][x - 1]
        )
        thalach = st.number_input("Maximum Heart Rate (bpm)", 60, 220, 150)
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox(
            "ST Slope", 
            [1, 2, 3],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x-1]
        )
    
    with col2:
        st.markdown("**🩸 Diagnostic Tests**")
        ca = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            [3, 6, 7],
            format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x]
        )
        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
            format_func=lambda x: ["No", "Yes"][x]
        )
        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x]
        )
    
    with col3:
        st.markdown("**📊 Patient Info**")
        age_c = st.number_input("Age", 20, 100, 50)
        sex_c = st.selectbox("Sex", ["Female", "Male"])
        bp_c = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
        chol_c = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
        fbs_c = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL", 
            [0, 1], 
            format_func=lambda x: ["No", "Yes"][x]
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        diag_btn = st.button("🔬 Run Diagnostic", type="primary", key="clinical_btn")
    
    # ===============================
    # PREDICTION
    # ===============================
    if diag_btn:
        try:
            # Encode sex
            sex_enc = 1 if sex_c == "Male" else 0
            
            # Build input exactly as training data
            input_data = {
                "age": float(age_c),
                "sex": float(sex_enc),
                "cp": float(cp),
                "trestbps": float(bp_c),
                "chol": float(chol_c),
                "fbs": float(fbs_c),
                "restecg": float(restecg),
                "thalach": float(thalach),
                "exang": float(exang),
                "oldpeak": float(oldpeak),
                "slope": float(slope),
                "ca": float(ca),
                "thal": float(thal)
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Ensure column order matches training
            input_df = input_df[models["clinical_features"]]
            
            # Predict (model has StandardScaler built-in)
            prob = models["clinical_model"].predict_proba(input_df)[0][1]
            pred = models["clinical_model"].predict(input_df)[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align:center;'>🔬 Diagnostic Result</h2>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                title={"text": "Heart Disease Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444"},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16,185,129,0.3)"},
                        {"range": [30, 70], "color": "rgba(245,158,11,0.3)"},
                        {"range": [70, 100], "color": "rgba(239,68,68,0.3)"}
                    ]
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8e8e8'),
                height=300
            )
            st.plotly_chart(fig, width='stretch')
            
            col1, col2 = st.columns(2)
            with col1:
                if pred == 1:
                    st.error(f"### ⚠️ POSITIVE\n\nRisk: {prob*100:.1f}%")
                else:
                    st.success(f"### ✅ NEGATIVE\n\nRisk: {prob*100:.1f}%")
            
            with col2:
                st.metric("Confidence", f"{max(prob, 1 - prob) * 100:.1f}%")
            
            st.markdown("---")
            
            if pred == 1:
                st.error("""
                **Recommended Actions**
                - Immediate cardiology consultation  
                - Consider angiography / catheterization  
                - Initiate guideline-based therapy  
                - Lifestyle modification counseling
                """)
            else:
                st.success("""
                **Maintenance Advice**
                - Continue routine monitoring  
                - Maintain heart-healthy lifestyle  
                - Schedule annual clinical follow-up
                """)
                
        except Exception as e:
            st.error(f"⚠️ Diagnostic failed: {str(e)}")
            st.exception(e)

# ============================================
# FOOTER
# ============================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #00d9ff; padding: 20px; 
                background: rgba(15, 52, 96, 0.4); border-radius: 10px; 
                border: 1px solid rgba(0, 217, 255, 0.3);'>
        <p style='margin: 0; font-size: 1.05rem;'>
            <strong>⚠️ Medical Disclaimer:</strong> This is a screening tool only and not a substitute for professional medical advice, diagnosis, or treatment.
        </p>
    </div>
""", unsafe_allow_html=True)