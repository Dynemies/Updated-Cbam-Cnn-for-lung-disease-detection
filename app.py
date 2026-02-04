import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

# --- 1. MODEL ARCHITECTURE ---
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Input
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Multiply, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def cbam_block(x, ratio=8):
    batch, _, _, channel = x.shape
    l1 = Dense(channel // ratio, activation="relu", use_bias=False)
    l2 = Dense(channel, use_bias=False)
    x_avg = l2(l1(Reshape((1, 1, channel))(GlobalAveragePooling2D()(x))))
    x_max = l2(l1(Reshape((1, 1, channel))(GlobalMaxPooling2D()(x))))
    x_cbam = Activation("sigmoid")(tf.keras.layers.Add()([x_avg, x_max]))
    x = Multiply()([x, x_cbam])

    x_avg = Lambda(lambda y: tf.reduce_mean(y, axis=-1, keepdims=True))(x)
    x_max = Lambda(lambda y: tf.reduce_max(y, axis=-1, keepdims=True))(x)
    x_cat = Concatenate(axis=-1)([x_avg, x_max])
    x_cbam = Conv2D(1, (7, 7), padding="same", activation="sigmoid")(x_cat)
    return Multiply()([x, x_cbam])

def build_model():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = cbam_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(4, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="SRM Medical AI", page_icon="🫁", layout="wide", initial_sidebar_state="collapsed")

# --- 3. DARK MODE CSS (With File Uploader Fix) ---
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: white; }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 3rem !important; padding-bottom: 2rem !important; }
    .login-card {
        background-color: #1e293b; padding: 40px; border-radius: 12px;
        border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;
    }
    h1, h2, h3 { color: #f8fafc !important; }
    p, li { color: #cbd5e1 !important; }
    .stTextInput > div > div > input { background-color: #0f172a; color: white; border: 1px solid #475569; }
    .stButton > button { background-color: #3b82f6; color: white; border: none; font-weight: bold; }
    .stButton > button:hover { background-color: #2563eb; }
    [data-testid='stFileUploader'] { background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px dashed #475569; }
    [data-testid='stFileUploader'] section { background-color: #1e293b; }
    button[kind="secondary"] { background-color: #f8fafc !important; color: #0f172a !important; border: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SESSION STATE ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'user_name' not in st.session_state: st.session_state.user_name = ""

# --- 5. LOGIN SCREEN (FIXED PASSWORD) ---
def login_screen():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("## 🫁 SRM MEDICAL AI")
        st.markdown("<p style='margin-bottom: 30px;'>Secure Diagnostic Portal | Dept. of ECE</p>", unsafe_allow_html=True)
        
        username = st.text_input("ID", placeholder="Enter Username")
        password = st.text_input("PIN", type="password", placeholder="Enter Password")
        
        st.write("") 
        
        if st.button("ACCESS SYSTEM 🔓", use_container_width=True):
            # --- FIX: HARDCODED PASSWORD RESTORED ---
            # This ensures it works on your laptop immediately.
            correct_pass = "Bomi@2026"
            
            if username.strip().capitalize() in ["Vamsi", "Anika", "Harsha", "Teacher"] and password == correct_pass:
                st.session_state.logged_in = True
                st.session_state.user_name = username.strip().capitalize()
                st.rerun()
            else:
                st.error("⛔ Access Denied")
        st.markdown('</div>', unsafe_allow_html=True)

# --- 6. MAIN APP ---
if not st.session_state.logged_in:
    login_screen()
else:
    # Sidebar
    with st.sidebar:
        st.title(f"👨‍⚕️ Dr. {st.session_state.user_name}")
        st.write("Status: 🟢 Online")
        st.markdown("---")
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.rerun()

    # Model
    MODEL_PATH = 'lung_disease_4_class_cbam.h5'
    CLASS_NAMES = ['Covid', 'Normal', 'Pneumonia', 'Tuberculosis']

    @st.cache_resource
    def load_ai_model():
        if not os.path.exists(MODEL_PATH): return None
        try:
            model = build_model()
            model.load_weights(MODEL_PATH)
            return model
        except: return None

    model = load_ai_model()

    # Dashboard
    st.markdown("## 🔎 X-Ray Diagnostic Center")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("📂 Step 1: Upload Patient Scan")
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Preview", use_container_width=True)

    with col2:
        st.info("🤖 Step 2: AI Analysis")
        
        if model and uploaded_file:
            if st.button("RUN DIAGNOSIS ⚡", use_container_width=True):
                with st.spinner("Processing..."):
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0).astype('float32') / 255.0
                    
                    predictions = model.predict(img_array)
                    confidence = np.max(predictions) * 100
                    predicted_class = CLASS_NAMES[np.argmax(predictions)]
                    
                    st.markdown("### Result:")
                    if predicted_class == "Normal":
                        st.success(f"## ✅ {predicted_class}")
                        st.caption("No pathology detected.")
                    else:
                        st.error(f"## ⚠️ {predicted_class}")
                        st.caption(f"Confidence: {confidence:.2f}%")
                    
                    st.bar_chart(data={"Condition": CLASS_NAMES, "Probability": predictions[0]}, x="Condition", y="Probability")
        
        elif not model:
            st.error("System Error: Model file missing.")
        else:
            st.write("Waiting for input...")