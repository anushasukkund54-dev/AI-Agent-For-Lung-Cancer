import streamlit as st
import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model

from google import genai


# =========================
# GEMINI API
# =========================

API_KEY = "AIzaSyDj6JudrOXAxuQPFETZcC16M1pzcZV0bwo"

client = genai.Client(api_key=API_KEY)


# =========================
# LOAD RESNET MODEL
# =========================

cancer_model = load_model(
    "lung_biopsy_resnet_model.h5"
)


# =========================
# STREAMLIT PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="Lung Cancer AI Agent",
    page_icon="🫁",
    layout="centered"
)


# =========================
# TITLE
# =========================

st.title("🫁 Lung Cancer AI Agent")

st.write(
    "Upload biopsy images or ask medical questions."
)
# =========================
# CHAT HISTORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================
# IMAGE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Biopsy Image",
    type=["jpg", "jpeg", "png"]
)
send_button = st.button("Send")


# =========================
# CHAT INPUT
# =========================




# =========================
# IMAGE ANALYSIS
# =========================

if uploaded_file:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Biopsy Image",
        use_container_width=True
    )

    # =========================
    # IMAGE PREPROCESSING
    # =========================

    img = img.resize((128, 128))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # RESNET PREDICTION
    # =========================

    prediction = cancer_model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # =========================
    # CLASS LABELS
    # =========================

    classes = {
        0: "Lung Benign Tissue",
        1: "Lung Adenocarcinoma",
        2: "Lung Squamous Cell Carcinoma",
        3: "Colon Benign Tissue",
        4: "Colon Adenocarcinoma"
    }

    result = classes[predicted_class]

    # =========================
    # SHOW RESULTS
    # =========================

    st.subheader("Biopsy Analysis")

    st.write(f"Prediction: {result}")

    st.write(f"Confidence: {confidence:.2f}%")

    # =========================
    # GEMINI PROMPT
    # =========================

    prompt = f"""
    A biopsy image was analyzed using a ResNet50 model.

    Prediction:
    {result}

    Confidence:
    {confidence:.2f}%

    Explain:
    - possible abnormalities
    - tissue changes
    - possible medical meaning
    - simple explanation for patient
    """

    # =========================
    # GEMINI RESPONSE
    # =========================

    try:

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )

        st.subheader("AI Medical Explanation")

        st.write(response.text)

    except Exception as e:

        st.error(f"Gemini Error: {str(e)}")
        
        
user_question = st.text_input(
    "Ask a medical question..."
)        


# =========================
# QUESTION ANSWERING
# =========================

# =========================
# QUESTION ANSWERING
# =========================

if user_question:

    # save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.write(user_question)

    try:

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=user_question
        )

        ai_response = response.text

    except Exception as e:

        ai_response = f"Error: {str(e)}"

    # save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )

    with st.chat_message("assistant"):

        st.write(ai_response)

             
   