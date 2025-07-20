import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import base64

# ==============================================================================
# Page Configuration - MUST BE THE FIRST STREAMLIT COMMAND
# ==============================================================================
st.set_page_config(
    page_title="Candidate Insight Engine",
    page_icon="🦋",
    layout="centered"
)

# ==============================================================================
# Helper function to load and encode the local image
# ==============================================================================
@st.cache_data
def get_image_as_base64(path):
    try:
        with open('/Users/biswa/Desktop/Project File 3/App/pexels-mo-eid-1268975-9454915.jpg', "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Image file not found at {path}. Please make sure it's in the same folder as the app.")
        return None

# ==============================================================================
# Final Custom CSS
# ==============================================================================
sidebar_image_base64 = get_image_as_base64("sidebar_image.jpeg")

if sidebar_image_base64:
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        html, body, [class*="st-"], .st-emotion-cache-1629p8f e1nzilvr5 {{
            font-family: 'Roboto', sans-serif;
        }}

        /* Main background */
        .main {{
            background-image: url("https://images.unsplash.com/photo-1541701494587-cb58502866ab?q=80&w=2070&auto=format&fit=crop");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Sidebar with background image */
        [data-testid="stSidebar"] {{
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("data:image/jpeg;base64,{sidebar_image_base64}");
            background-size: cover;
            background-position: center;
        }}
        
        /* --- FINAL FIX for Sidebar Text Backgrounds and Styles --- */
        /* Target the header specifically */
        [data-testid="stSidebar"] h2 {{
            text-align: center;
            font-size: 26px !important;
            font-weight: bold;
            background: transparent !important;
            color: white !important;
            text-shadow: 1px 1px 5px rgba(0,0,0,0.8);
        }}
        
        /* Target the labels for all widgets in the sidebar */
        [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] label {{
            background: transparent !important;
            color: white !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
            font-weight: bold;
        }}
        /* -------------------------------------------------------- */

        /* Center the button */
        [data-testid="stSidebar"] .stButton {{
            display: flex;
            justify-content: center;
        }}
        .stButton>button {{
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
            background-color: rgba(255, 255, 255, 0.25) !important;
            color: white !important;
            font-weight: bold !important;
            width: 80%;
        }}
        
        /* Transparent text boxes with white text */
        [data-testid="stSidebar"] .stTextArea textarea {{
            background-color: rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #FFFFFF;
        }}
        
        /* Main page title */
        h1 {{
            color: #FFFFFF !important;
            text-shadow: 1px 1px 5px rgba(0,0,0,0.7);
        }}

        /* Glassmorphism for cards on main page */
        [data-testid="stMetric"], .stAlert, [data-testid="stExpander"] {{
            background-color: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
            color: #FFFFFF !important;
        }}
    </style>

    <div class="butterfly"></div>
    <style>
        .butterfly {{
            position: fixed; bottom: 20px; right: 20px; width: 150px;
            height: 150px; background-image: url('https://i.imgur.com/v28Kq8v.png');
            background-size: contain; background-repeat: no-repeat;
            opacity: 0.7; z-index: -1;
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# Constants & Model Loading (No changes below this line)
# ==============================================================================
MODEL_PATH = 'multi_input_model.keras'
JD_TOKENIZER_PATH = 'jd_tokenizer.pickle'
SKILLS_TOKENIZER_PATH = 'skills_tokenizer.pickle'
TRANSCRIPT_TOKENIZER_PATH = 'transcript_tokenizer.pickle'
GRADE_ENCODER_PATH = 'grade_encoder.pickle'
SENTIMENT_ENCODER_PATH = 'sentiment_encoder.pickle'

MAX_LEN_JD = 150
MAX_LEN_SKILLS = 50
MAX_LEN_TRANSCRIPT = 300

@st.cache_resource
def load_all_components():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(JD_TOKENIZER_PATH, 'rb') as handle: jd_tokenizer = pickle.load(handle)
        with open(SKILLS_TOKENIZER_PATH, 'rb') as handle: skills_tokenizer = pickle.load(handle)
        with open(TRANSCRIPT_TOKENIZER_PATH, 'rb') as handle: transcript_tokenizer = pickle.load(handle)
        with open(GRADE_ENCODER_PATH, 'rb') as handle: grade_encoder = pickle.load(handle)
        with open(SENTIMENT_ENCODER_PATH, 'rb') as handle: sentiment_encoder = pickle.load(handle)
        return model, jd_tokenizer, skills_tokenizer, transcript_tokenizer, grade_encoder, sentiment_encoder
    except Exception as e:
        st.error(f"Error loading necessary files: {e}")
        return (None,) * 6

# ==============================================================================
# Main App Interface
# ==============================================================================
with st.sidebar:
    st.header("Input Data")
    model, jd_tokenizer, skills_tokenizer, transcript_tokenizer, grade_encoder, sentiment_encoder = load_all_components()
    
    if model is not None:
        jd_input = st.text_area("1. Job Description", height=150, placeholder="Paste JD here...")
        skills_input = st.text_area("2. Skills from Resume", height=100, placeholder="Paste skills here...")
        transcript_input = st.text_area("3. Interview Transcript", height=250, placeholder="Paste transcript here...")
        analyze_button = st.button("Analyze Interview")
    else:
        st.error("Model files not loaded.")
        analyze_button = False

st.title("Candidate Insight Engine")

if analyze_button:
    if jd_input and skills_input and transcript_input:
        with st.spinner('Analyzing... Please wait.'):
            jd_seq = pad_sequences(jd_tokenizer.texts_to_sequences([jd_input]), maxlen=MAX_LEN_JD)
            skills_seq = pad_sequences(skills_tokenizer.texts_to_sequences([skills_input]), maxlen=MAX_LEN_SKILLS)
            transcript_seq = pad_sequences(transcript_tokenizer.texts_to_sequences([transcript_input]), maxlen=MAX_LEN_TRANSCRIPT)
            model_input = [jd_seq, skills_seq, transcript_seq]
            predictions = model.predict(model_input)
            grade_pred_probs, sentiment_pred_probs = predictions
            sentiment_index = np.argmax(sentiment_pred_probs, axis=1)[0]
            predicted_sentiment = sentiment_encoder.classes_[sentiment_index]
            grade_index = np.argmax(grade_pred_probs, axis=1)[0]
            predicted_grade = grade_encoder.classes_[grade_index]
            
            st.subheader("📊 Analysis Results")
            col1, col2 = st.columns(2)
            with col1: st.metric(label="Predicted Grade", value=predicted_grade)
            with col2: st.metric(label="Predicted Sentiment", value=predicted_sentiment.capitalize())
            
            with st.expander("View Detailed Confidence Scores"):
                prob_data = { "Confidence": sentiment_pred_probs.flatten() }
                st.bar_chart(pd.DataFrame(prob_data, index=sentiment_encoder.classes_))
    else:
        st.warning("Please fill in all three text fields in the sidebar.")
else:
    st.info("Please provide inputs in the sidebar and click 'Analyze Interview'.")