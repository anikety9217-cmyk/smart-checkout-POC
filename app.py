import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import time

warnings.filterwarnings('ignore')

# st.set_page_config must be the very first Streamlit command
st.set_page_config(page_title="Smart Checkout Engine", page_icon="💳")

# --- 1. Data Generation & Caching ---
def generate_mock_data():
    np.random.seed(42)
    n_samples = 200
    devices = ["iOS", "Android", "Windows", "Mac"]
    
    data = {
        'Device_OS': np.random.choice(devices, n_samples),
        'Cart_Value_USD': np.random.randint(15, 1200, n_samples)
    }
    
    successful_methods = []
    for device, val in zip(data['Device_OS'], data['Cart_Value_USD']):
        if device == "iOS" and val < 150:
            successful_methods.append("Apple Pay")
        elif device == "Android" and val < 150:
            successful_methods.append("Google Pay")
        elif val >= 500:
            successful_methods.append("BNPL")
        else:
            successful_methods.append("Credit Card")
            
    data['Successful_Payment_Method'] = successful_methods
    df = pd.DataFrame(data)
    df.to_csv('checkout_data.csv', index=False)

if not os.path.exists('checkout_data.csv'):
    generate_mock_data()

# OPTIMIZATION: Use cache_resource for ML Models
@st.cache_resource 
def load_and_train():
    df = pd.read_csv('checkout_data.csv')
    
    X = df[['Device_OS', 'Cart_Value_USD']].copy()
    y = df['Successful_Payment_Method']
    
    le = LabelEncoder()
    le.fit(["iOS", "Android", "Windows", "Mac"])
    
    X['Device_OS_Encoded'] = le.transform(X['Device_OS'])
    X_train = X[['Device_OS_Encoded', 'Cart_Value_USD']]
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y)
    
    return model, le

model, le = load_and_train()

# --- 2. Session State Initialization ---
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.change_count = 0
    st.session_state.display_device = "iOS"
    st.session_state.display_value = 150.0

# --- 3. UI and Form Layout ---
st.title("💳 AI Smart Checkout POC")
st.markdown("Adjust the user's session data in the sidebar to see how the Machine Learning model dynamically reorders the checkout experience.")

# OPTIMIZATION: Wrap sidebar in a form to batch inputs and prevent auto-reruns
with st.sidebar.form("user_context_form"):
    st.header("User Session Context")
    input_device = st.selectbox("Device OS", ["iOS", "Android", "Windows", "Mac"])
    input_cart_value = st.number_input(
        "Cart Value ($)", 
        min_value=15.0, 
        max_value=2000.0, 
        value=150.0, 
        step=10.0,
        help="Type in the user's current cart value."
    )
    
    # This button acts as your logic gate
    submit_clicked = st.form_submit_button("Update Checkout", use_container_width=True)

# --- 4. Prediction Logic ---
if submit_clicked or st.session_state.prediction is None:
    st.session_state.change_count += 1
    
    # Simulated UX Latency
    if st.session_state.change_count == 1:
        thinking_msg = "Initializing predictive engine..."
    elif st.session_state.change_count == 2:
        thinking_msg = "Recalculating optimal checkout flow..."
    else:
        thinking_msg = f"Re-evaluating user context... (Adjustment #{st.session_state.change_count - 1})"
        
    with st.spinner(thinking_msg):
        time.sleep(1.0) 
        
    # Run Prediction
    input_device_encoded = le.transform([input_device])[0]
    input_df = pd.DataFrame(
        [[input_device_encoded, input_cart_value]], 
        columns=['Device_OS_Encoded', 'Cart_Value_USD']
    )
    
    # Save results to session state so they persist
    st.session_state.prediction = model.predict(input_df)[0]
    st.session_state.display_device = input_device
    st.session_state.display_value = input_cart_value

# --- 5. Render Dynamic Checkout ---
st.write("---")
st.subheader("🛒 Checkout Page")
st.caption(f"Simulating a user on an **{st.session_state.display_device}** device with a **${st.session_state.display_value:.2f}** cart.")

st.success(f"✨ **AI Recommended Default:** {st.session_state.prediction}")

st.button(f"Pay with {st.session_state.prediction}", use_container_width=True, type="primary", key="primary_pay_btn")

all_methods = ["Apple Pay", "Google Pay", "Credit Card", "BNPL"]
for method in all_methods:
    if method != st.session_state.prediction:
        st.button(
            f"Pay with {method}", 
            use_container_width=True, 
            key=f"pay_{method.lower().replace(' ', '_')}"
        )
