import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress minor warnings for a clean terminal
warnings.filterwarnings('ignore')

# --- 1. Load Data & Train Model ---
@st.cache_data
def load_and_train():
    df = pd.read_csv('checkout_data.csv')
    X = df[['Device_OS', 'Cart_Value_USD']].copy()
    y = df['Successful_Payment_Method']
    
    le = LabelEncoder()
    X['Device_OS_Encoded'] = le.fit_transform(X['Device_OS'])
    X_train = X[['Device_OS_Encoded', 'Cart_Value_USD']]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y)
    
    return model, le

model, le = load_and_train()

# --- 2. Build the Streamlit UI ---
st.set_page_config(page_title="Smart Checkout Engine", page_icon="💳")
st.title("💳 AI Smart Checkout POC")
st.markdown("Adjust the user's session data in the sidebar to see how the Machine Learning model dynamically reorders the checkout experience.")

st.sidebar.header("User Session Context")
input_device = st.sidebar.selectbox("Device OS", ["iOS", "Android", "Windows", "Mac"])
input_cart_value = st.sidebar.slider("Cart Value ($)", min_value=15, max_value=1200, value=150)

# --- 3. Make the ML Prediction ---
input_device_encoded = le.transform([input_device])[0]
prediction = model.predict([[input_device_encoded, input_cart_value]])[0]

# --- 4. Render the Dynamic Checkout UI ---
st.write("---")
st.subheader("🛒 Checkout Page")
st.caption(f"Simulating a user on an **{input_device}** device with a **${input_cart_value}** cart.")

st.success(f"✨ **AI Recommended Default:** {prediction}")
st.button(f"Pay with {prediction}", use_container_width=True, type="primary")

all_methods = ["Apple Pay", "Google Pay", "Credit Card", "BNPL"]
for method in all_methods:
    if method != prediction:
        st.button(f"Pay with {method}", use_container_width=True)  
