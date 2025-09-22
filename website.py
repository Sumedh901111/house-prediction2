import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load pipeline ---
try:
    pipeline = joblib.load("pipeline.pkl")
except Exception as e:
    st.error(f"Error loading pipeline.pkl: {e}")
    st.stop()

# --- Page config ---
st.set_page_config(
    page_title="🏡 House Price Predictor", 
    page_icon="🏠", 
    layout="wide"
)

# --- Dark mode styling ---
st.markdown(
    """
    <style>
    /* Background and text */
    body, .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Titles and headers */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #FAFAFA !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1f77b4 !important;
        color: white !important;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #166298 !important;
        color: #ffffff !important;
    }

    /* Sidebar inputs (sliders, selectbox, number input) */
    .stSlider>div>div>div>div, 
    .stNumberInput input, 
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2C2F38 !important;
        color: #FAFAFA !important;
        border-radius: 6px;
    }

    /* Dataframe display */
    .stDataFrame div[data-testid="stDataFrame"] {
        background-color: #1E222A !important;
        color: #FAFAFA !important;
    }

    /* Table text */
    .stDataFrame td, .stDataFrame th {
        color: #FAFAFA !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Title ---
st.title("🏡 House Price Prediction App")
st.markdown("Enter the details below to predict the house price.")

# --- Initialize session state for history ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Sidebar Inputs ---
st.sidebar.header("🏠 House Features Input")

area = st.sidebar.number_input("Total Area (sqft)", min_value=100, max_value=20000, value=1000, step=50)
bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 10, 2)
stories = st.sidebar.slider("Number of Stories", 1, 5, 1)
parking = st.sidebar.slider("Number of Parking Spaces", 0, 10, 1)

mainroad = st.sidebar.selectbox("Close to Main Road?", ["yes", "no"])
guestroom = st.sidebar.selectbox("Has Guest Room?", ["yes", "no"])
basement = st.sidebar.selectbox("Has Basement?", ["yes", "no"])
hotwaterheating = st.sidebar.selectbox("Has Hot Water Heating?", ["yes", "no"])
airconditioning = st.sidebar.selectbox("Has AC?", ["yes", "no"])
prefarea = st.sidebar.selectbox("Located in Preferred Area?", ["yes", "no"])
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# --- Collect inputs ---
input_dict = {
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "parking": [parking],
    "prefarea": [prefarea],
    "furnishingstatus": [furnishingstatus]
}
input_df = pd.DataFrame(input_dict)

# --- Predict ---
st.header("Predicted House Price")
if st.button("Predict Price"):
    try:
        prediction = pipeline.predict(input_df)[0]
        st.success(f"🏠 Predicted House Price: ₹ {prediction:,.0f}")
        
        # Save inputs + prediction to session state
        st.session_state.history.append({**input_dict, "predicted_price": prediction})
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# --- Show previous predictions ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Previous Predictions")
    
    # Flatten the dictionary for display
    history_df = pd.DataFrame([
        {k: (v[0] if isinstance(v, list) else v) for k, v in record.items()} 
        for record in st.session_state.history
    ])
    st.dataframe(history_df, use_container_width=True)

st.markdown("---")
st.markdown("**Note:** The prediction is based on a trained machine learning model. Actual prices may vary.")
