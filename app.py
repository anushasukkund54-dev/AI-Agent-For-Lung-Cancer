import streamlit as st
import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model

from google import genai


# =========================
# GEMINI API
# =========================

API_KEY = "AIzaSyCdojAN1xttjg5-MndsC9A7bA2mMLiqxQM"

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
# SIDEBAR IMAGE UPLOAD
# =========================



# =========================
# IMAGE UPLOAD
# =========================


# =========================
# CHAT INPUT
# =========================




# =========================
# IMAGE ANALYSIS
# =========================

        
# =========================
# BOTTOM INPUT AREA
# =========================



# =========================
# QUESTION ANSWERING
# =========================

# =========================
# QUESTION ANSWERING
# =========================

# =========================
# CHATGPT STYLE INPUT
# =========================

user_question = st.chat_input(
    "Ask a medical question...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"]
)

# =========================
# MAIN FLOW
# =========================

if user_question:

    # user text

    message_text = user_question["text"]

    # uploaded image files

    uploaded_files = user_question["files"]

    # =========================
    # SHOW USER MESSAGE
    # =========================

    if message_text:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": message_text
            }
        )

        with st.chat_message("user"):

            st.write(message_text)

    # =========================
    # IMAGE ANALYSIS
    # =========================

    if uploaded_files:

        img = Image.open(uploaded_files[0])

        st.image(
            img,
            caption="Uploaded Biopsy Image",
            use_container_width=True
        )

        # preprocessing

        img = img.resize((128, 128))

        img_array = np.array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        # prediction

        prediction = cancer_model.predict(img_array)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        classes = {
            0: "Lung Benign Tissue",
            1: "Lung Adenocarcinoma",
            2: "Lung Squamous Cell Carcinoma",
            3: "Colon Benign Tissue",
            4: "Colon Adenocarcinoma"
        }

        result = classes[predicted_class]

        st.subheader("Biopsy Analysis")

        st.write(f"Prediction: {result}")

        st.write(f"Confidence: {confidence:.2f}%")

        analysis_prompt = f"""
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

        try:

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=analysis_prompt
            )

            ai_response = response.text

        except Exception as e:

            ai_response = f"Gemini Error: {str(e)}"

    # =========================
    # NORMAL QUESTION ANSWERING
    # =========================

    elif message_text:

        try:

            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=message_text
            )

            ai_response = response.text

        except Exception as e:

            ai_response = f"Gemini Error: {str(e)}"

    # =========================
    # SHOW ASSISTANT RESPONSE
    # =========================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ai_response
        }
    )

    with st.chat_message("assistant"):

        st.write(ai_response)