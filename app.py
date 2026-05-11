import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Titanic AI Predictor", page_icon="🚢", layout="wide")

# Custom CSS taake app thori "Whiteboard" aur saaf lage
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar mein information
st.sidebar.title("About Project")
st.sidebar.info("Yeh AI model Random Forest algorithm par mabni hai jo Titanic ke historical data se seekha gaya hai.")

# Main Header
st.title("🚢 Titanic Survival Prediction System")
st.write("Data enter karein aur AI se result maloom karein.")
st.divider()

# Input Fields ko Groups mein divide kiya (Whiteboard Layout)
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Passenger Details")
    age = st.slider("Age (Umar)", 1, 100, 25)
    gender = st.radio("Gender (Jins)", ["Male", "Female"], horizontal=True)
    p_class = st.selectbox("Passenger Class (Status)", [1, 2, 3], format_func=lambda x: f"Class {x}")

with col2:
    st.subheader("💰 Journey Details")
    fare = st.number_input("Fare (Ticket ki Qeemat $)", 0, 500, 32)
    alone = st.selectbox("Kya aap akele thay?", ["Yes", "No"])

# Conversion logic
who_num = 1 if gender == "Male" else 2
alone_num = 1 if alone == "Yes" else 0

# Predict Button
st.divider()
if st.button("PREDICT SURVIVAL CHANCES"):
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Model Input
    features = np.array([[age, fare, p_class-1, who_num, alone_num]])
    prediction = model.predict(features)
    
    # Final Result Display
    if prediction[0] == 1:
        st.balloons()
        st.success("### ✅ RESULT: YOU SURVIVED!")
        st.write("AI ke mutabiq aap un khush-kismat logon mein se hotay jo bacha liye gaye.")
    else:
        st.error("### ❌ RESULT: DID NOT SURVIVE")
        st.write("Afsos, data ke mutabiq is surat-e-haal mein bachna mushkil tha.")
