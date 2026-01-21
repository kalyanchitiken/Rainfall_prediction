import streamlit as st
import pandas as pd
import pickle

# page config
st.set_page_config(
    page_title="Rainfall prediction * ML APP",
    page_icon="🌧️",
    layout="wide"
)

# custom csc
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(120deg, #0f172a 0%, #111827 40%, #0b1220 100%);
    color: #ffffff;
}

/* Title style */
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 6px;
}
.sub-title {
    font-size: 16px;
    opacity: 0.85;
    margin-bottom: 20px;
}

/* Card style */
.card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0px 8px 22px rgba(0,0,0,0.28);
}

/* Small tag */
.tag {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.35);
    font-size: 13px;
    margin-top: 8px;
}

/* Button style */
div.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 12px 16px;
    font-size: 16px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(90deg, #22c55e 0%, #06b6d4 55%, #3b82f6 100%);
    color: white;
    transition: 0.2s ease-in-out;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 10px 25px rgba(59,130,246,0.25);
}

/* Input fields */
div[data-baseweb="input"] > div {
    border-radius: 12px !important;
}

/* Alerts styling */
.stAlert {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# load model
with open("rainfall_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("feature_columns.pkl", "rb") as file:
    feature_cols = pickle.load(file)

# Header
st.markdown('<div class="main-title">🌧️ Rainfall Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict rainfall using real weather parameters • Designed for interview deployment ✅</div>', unsafe_allow_html=True)

st.markdown('<span class="tag">ML Classification • Real-time Prediction • Streamlit UI</span>', unsafe_allow_html=True)
st.write("")

# layout
left,right = st.columns([1.25,1],gap="large")

# left: Inputs
with left:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("Enter Weather Details")

    col1,col2,col3 = st.columns(3)

    with col1:
        pressure = st.number_input("pressure",value=1010.0,step=0.1)
        maxtemp = st.number_input("Max Temp",value=32.0,step=0.1)
        mintemp = st.number_input("Min Temp",value=24.0,step=0.1)
        
    with col2:
        temparature = st.number_input("Temperature (temparature)", value=28.0, step=0.1)
        dewpoint = st.number_input("Dew Point", value=23.0, step=0.1)
        humidity = st.number_input("Humidity (%)", value=85.0, step=1.0)
    
    with col3:
        cloud = st.number_input("Cloud (0–10)", value=7.0, step=0.5)
        sunshine = st.number_input("Sunshine (hours)", value=2.5, step=0.1)
        windspeed = st.number_input("Wind Speed", value=12.0, step=0.5)

    winddirection = st.slider("Wind Direction (0° to 360°)", 0, 360, 180)

    st.write("")
    predict_btn = st.button("🚀 Predict Rainfall")

    st.markdown('</div>', unsafe_allow_html=True)

    # result
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Prediction Result")
        
        if predict_btn:
            new_data = pd.DataFrame([{
                "pressure": pressure,
                "maxtemp": maxtemp,
                "temparature": temparature, 
                "mintemp": mintemp,
                "dewpoint": dewpoint,
                "humidity": humidity,
                "cloud": cloud,
                "sunshine": sunshine,
                "winddirection": float(winddirection),
                "windspeed": windspeed
            }])

            # Ensure correct feature order
            new_data = new_data[feature_cols]

            # Predict
            pred = model.predict(new_data)[0]
            proba_yes = model.predict_proba(new_data)[0][1]
            proba_no = 1 - proba_yes

             # Show output
            if pred == 1:
                st.success(f"✅ Rainfall: YES 🌧️  | Confidence: {proba_yes:.2f}")
            else:
                st.warning(f"✅ Rainfall: NO ☀️  | Confidence: {proba_no:.2f}")
            
            st.write("### 🎯 Probability Meter")
            st.progress(float(proba_yes))
            
            # Simple human-like conclusion
            st.write("### 🧠 Simple Explanation")
            if pred == 1:
                st.info(
                    "Rain is likely because **humidity and cloud cover are high**, "
                    "and **sunshine is low**. These conditions generally support rainfall."
                )
            else:
                st.info(
                    "Rain is less likely because **sunshine is present** and/or "
                    "**humidity and cloud cover are not high enough** for rainfall."
                )
            
            # Show input summary (professional)
            st.write("### 📄 Input Summary")
            st.dataframe(new_data, use_container_width=True)

        else:
            st.info("👈 Enter values and click **Predict Rainfall** to get output.")
            st.write("✨ Tip: High humidity + high cloud + low sunshine usually gives **YES** 🌧️")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.write("")
st.caption("Rainfall Prediction ML Project 🌦️")