import streamlit as st
import cv2
import numpy as np
import speech_recognition as sr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from transformers import pipeline, set_seed

st.set_page_config(page_title="AI Farmer Helper")

# session variables
if 'camera_on' not in st.session_state:
    st.session_state['camera_on'] = False
if 'image_taken' not in st.session_state:
    st.session_state['image_taken'] = False
if 'saved_img' not in st.session_state:
    st.session_state['saved_img'] = False 
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
import streamlit as st

# Inject CSS for background
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Call function with your image
add_bg_from_local("g.png")
st.markdown("""
<div style='
    background-color: #e0f7fa;
    border-left: 5px solid #00796b;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 1px 1px 6px rgba(0,0,0,0.1);
    margin-bottom: 10px;
    text-align: center;
'>
    <h3 style='color: #00695c; margin: 0;'>🌾 AI Farmer Helper</h3>
    <p style='color:#333; font-size:15px; margin-top: 8px;'>
        Upload a crop image to detect diseases and get smart treatment plans instantly.
    </p>
</div>
""", unsafe_allow_html=True)


# crop selector
crop = st.selectbox("Select your crop", ["Select", "Tomato", "Rice", "Corn", "Blackgram", "Surgane"])

# tomato disease info
corn_labels = {
    0: "healthy",
    1: "blight",
    2: "common_rust",
    3: "gray_leaf_spot"
}

corn_treatment = {
    "healthy": (
        "The corn plant is healthy; continue with routine crop care. "
        "Maintain field hygiene, monitor regularly for early signs of disease or pests, "
        "and follow proper irrigation and fertilization schedules. "
        "No chemical treatment is needed at this stage."
    ),

    "blight": (
        "Use disease-resistant corn varieties and rotate crops with non-host species like legumes. "
        "Remove and destroy infected plant residues after harvest to reduce pathogen buildup. "
        "If infection is visible, spray Mancozeb at 2–2.5g/litre of water at 10-day intervals. "
        "Avoid overhead irrigation and dense planting to reduce humidity."
    ),

    "common_rust": (
        "Grow rust-resistant corn hybrids and monitor fields during humid conditions. "
        "Ensure adequate plant spacing to promote airflow and reduce leaf wetness. "
        "In case of severe infection, apply Propiconazole (1 ml/litre) or Mancozeb (2g/litre). "
        "Early spraying at the first sign of pustules is crucial for effective control."
    ),

    "gray_leaf_spot": (
        "Rotate crops annually to reduce fungal spore load in soil and residue. "
        "Use resistant varieties and maintain proper row spacing to increase air movement. "
        "At disease onset, spray Azoxystrobin (as per label) or Mancozeb (2g/litre) for control. "
        "Avoid high plant density and excessive nitrogen application."
    )
}


blackgram_labels = {
    0: "healthy",
    1: "anthracnose",
    2: "leaf_crinkle",
    3: "powdery_mildew",
    4: "yellow_mosaic"
}

blackgram_treatment = {
    "healthy": (
        "The plant is healthy; maintain routine crop care and monitoring. "
        "Ensure timely irrigation, balanced fertilization, and weed management. "
        "Inspect plants regularly for early signs of pests or disease. "
        "No treatment is required at this stage."
    ),

    "anthracnose": (
        "Use certified disease-free seeds and treat them with Carbendazim or Thiram before sowing (2g/kg seed). "
        "At the first sign of infection, spray Mancozeb 0.2% (2g/litre) every 10 days. "
        "Remove and destroy infected plant parts and crop debris after harvest. "
        "Avoid waterlogging and overcrowded planting."
    ),

    "leaf_crinkle": (
        "Remove and destroy infected plants to limit spread. "
        "Control whitefly vectors using insecticides like Imidacloprid (0.3 ml/litre) or Thiamethoxam. "
        "Use tolerant or resistant varieties if available. "
        "Maintain field sanitation and avoid overlapping crops."
    ),

    "powdery_mildew": (
        "Spray wettable sulfur (0.2%) or Karathane (0.1%) as soon as symptoms appear. "
        "Ensure good spacing for air circulation to reduce humidity around plants. "
        "Avoid using excessive nitrogen fertilizers. "
        "Repeat spray after 10–12 days if infection persists."
    ),

    "yellow_mosaic": (
        "Use resistant or tolerant varieties for better disease control. "
        "Control whiteflies (vectors) using yellow sticky traps and systemic insecticides like Thiamethoxam. "
        "Remove and destroy infected plants early. "
        "Monitor the crop regularly for whitefly population and avoid growing susceptible crops nearby."
    )
}


surgane_labels = {
    0: 'healthy',
    1: 'mosaic',
    2: 'red_rot',
    3: 'rust',
    4: 'yellow_leaf_disease'
}

sugarcane_treatment = {
    "healthy": (
        "The crop is healthy; continue with regular monitoring and best practices. "
        "Ensure balanced fertilization, timely irrigation, and weed control. "
        "Inspect regularly for any early symptoms of disease or pest attack. "
        "No chemical treatment is needed at this stage."
    ),

    "mosaic": (
        "Remove and destroy infected plants immediately to stop the spread. "
        "Control aphid vectors using systemic insecticides like Imidacloprid (0.3 ml/litre). "
        "Use mosaic-resistant sugarcane varieties and certified disease-free planting material. "
        "Practice crop rotation to reduce viral persistence in the soil."
    ),

    "red_rot": (
        "Use only disease-free, healthy setts for planting. "
        "Before planting, treat seed setts in hot water or Carbendazim 0.1% solution for 30 minutes. "
        "Destroy and burn infected clumps from the field. "
        "Plant red rot–resistant varieties and avoid waterlogging conditions."
    ),

    "rust": (
        "Remove and destroy heavily infected leaves from the field. "
        "Spray with Propiconazole 0.1% (1 ml/litre) at the first sign of rust pustules. "
        "Repeat spraying at 10–15 day intervals if needed. "
        "Prefer rust-resistant sugarcane varieties in affected regions."
    ),

    "yellow_leaf_disease": (
        "Use only healthy and certified planting material. "
        "Control aphid vectors using recommended insecticides (e.g., Thiamethoxam or Imidacloprid). "
        "Uproot and destroy infected plants to prevent disease spread. "
        "Monitor crop regularly and maintain good drainage and nutrition."
    )
}


tomato_labels = {
    0: "Tomato_Bacterial_spot",
    1: "Tomato_Early_blight",
    2: "Tomato_healthy",
    3: "Tomato_Late_blight",
    4: "Tomato_Leaf_Mold",
    5: "Tomato_Septoria_leaf_spot",
    6: "Tomato_Spider_mites Two-spotted_spider_mite",
    7: "Tomato_Target_Spot",
    8: "Tomato_Tomato_mosaic_virus",
    9: "Tomato_Tomato_Yellow_Leaf_Curl_Virus"
}

tomato_treatment = {
    "Tomato_Bacterial_spot": (
        "Remove and destroy infected leaves to prevent spread. "
        "Avoid working with wet plants and overhead irrigation. "
        "Use resistant varieties where available. "
        "Apply copper-based bactericide spray (Copper Oxychloride at 3g/litre)."
    ),

    "Tomato_Early_blight": (
        "Ensure proper plant spacing and remove lower infected leaves. "
        "Use crop rotation and avoid planting in the same spot each year. "
        "Provide balanced nutrients to reduce stress. "
        "Spray with Mancozeb or Chlorothalonil at 2–3g/litre every 7–10 days."
    ),

    "Tomato_healthy": (
        "Plant appears healthy; continue with standard care. "
        "Maintain good watering, fertilization, and weed control. "
        "Regularly monitor for early signs of diseases or pests. "
        "No chemical treatment is needed at this time."
    ),

    "Tomato_Late_blight": (
        "Remove and destroy infected plants immediately. "
        "Avoid leaf wetting and ensure proper air circulation. "
        "Use resistant tomato varieties if available. "
        "Apply Bordeaux mixture (1% solution) or Metalaxyl-based fungicide."
    ),

    "Tomato_Leaf_Mold": (
        "Improve ventilation in greenhouses or dense plantings. "
        "Remove infected leaf parts and avoid overhead watering. "
        "Apply fungicides preventively during humid conditions. "
        "Spray Dithane M-45 (2.5g/litre) every 10 days."
    ),

    "Tomato_Septoria_leaf_spot": (
        "Remove lower leaves as soon as spots appear. "
        "Avoid wetting foliage and irrigate at soil level. "
        "Practice crop rotation and clean up debris after harvest. "
        "Spray with Zineb or Mancozeb (2–3g/litre) weekly as needed."
    ),

    "Tomato_Spider_mites Two-spotted_spider_mite": (
        "Wash plants with water to dislodge mites and reduce dust. "
        "Use insecticidal soap or neem oil for organic control. "
        "Introduce natural predators like ladybugs or predatory mites. "
        "Spray neem oil (5ml/litre) every 5–7 days if infestation persists."
    ),

    "Tomato_Target_Spot": (
        "Remove and destroy infected leaves promptly. "
        "Avoid overhead irrigation and reduce humidity around plants. "
        "Use disease-free seeds and maintain crop rotation. "
        "Spray with Azoxystrobin-based fungicide (as per label)."
    ),

    "Tomato_Tomato_mosaic_virus": (
        "Immediately remove and destroy infected plants. "
        "Disinfect tools and wash hands after handling plants. "
        "Avoid tobacco use while working with plants. "
        "No chemical cure; rely on sanitation and resistant varieties."
    ),

    "Tomato_Tomato_Yellow_Leaf_Curl_Virus": (
        "Control whitefly population using yellow sticky traps or neem oil. "
        "Remove and destroy infected plants to prevent spread. "
        "Use insect netting or barriers during early growth stages. "
        "Apply Imidacloprid-based insecticides as a last resort."
    )
}


# rice disease info
rice_labels = {
    0: 'bacterial_leaf_blight',
    1: 'brown_spot',
    2: 'healthy',
    3: 'leaf_blast',
    4: 'leaf_scald',
    5: 'narrow_brown_spot'
}

rice_treatment = {
    "bacterial_leaf_blight": (
        "Remove and destroy infected plants to reduce disease spread. "
        "Avoid excessive nitrogen application and water stagnation. "
        "Use resistant varieties if available. "
        "Spray with Copper Oxychloride at 3g/litre of water."
    ),
    
    "brown_spot": (
        "Use certified disease-free seeds and ensure proper seed treatment before sowing. "
        "Maintain balanced nutrition, especially adequate potassium. "
        "Avoid overcrowding of plants and excessive irrigation. "
        "Spray Mancozeb or Carbendazim at 2g/litre of water."
    ),
    
    "healthy": (
        "No disease symptoms found. Continue regular crop care practices. "
        "Monitor crop regularly for early signs of pests or diseases. "
        "Maintain proper irrigation and apply fertilizers as per schedule. "
        "No chemical treatment needed at this stage."
    ),
    
    "leaf_blast": (
        "Avoid excessive nitrogen fertilizers and water stress during early stages. "
        "Maintain field sanitation and reduce humidity by proper spacing. "
        "Use resistant varieties if possible. "
        "Spray Tricyclazole at 0.6g/litre of water at 10–12 day intervals."
    ),
    
    "leaf_scald": (
        "Ensure good field drainage and reduce leaf wetness by avoiding overhead irrigation. "
        "Do not overcrowd plants and ensure proper airflow. "
        "Remove and destroy infected debris after harvest. "
        "Spray Propiconazole at 1ml/litre of water for control."
    ),
    
    "narrow_brown_spot": (
        "Remove infected leaves and improve plant spacing to enhance airflow. "
        "Ensure balanced fertilization, especially potassium and phosphorus. "
        "Avoid continuous monocropping of rice. "
        "Apply Carbendazim at 1g/litre of water to manage infection."
    )
}


# main function to handle prediction
def predict_image(img_path, model_path, label_map, suggestion_map, crop_name):
    # load model
    model = load_model(model_path)

    # load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr / 255.0

    # prediction
    prediction = model.predict(img_arr)
    pred_index = np.argmax(prediction)
    predicted_label = label_map[pred_index]
    confidence = np.max(prediction)

    #st.write(f"Predicted: {predicted_label} with confidence {confidence:.2f}")

    #st.success(f"**Disease Detected:** {predicted_label}")
    #st.write(f"**Advice:** {suggestion_map.get(predicted_label, '❓ No recommendations available for this disease')}")
    st.success(f"🔍 Predicted: **{predicted_label}** with confidence **{confidence:.2f}**")

# Stylish Result Box
    st.markdown(f"""
<div style='
    background-color: #e8f5e9;
    border-left: 6px solid #43a047;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
'>
    <h4 style='color:#1b5e20;'>🦠 Disease Detected: <span style='color:#2e7d32;'>{predicted_label}</span></h4>
    <p style='font-size:15px; color:#333; margin-top:10px;'>
        💡 <strong>Advice:</strong> {suggestion_map.get(predicted_label, '❓ No recommendations available for this disease')}
    </p>
</div>
""", unsafe_allow_html=True)

    # text generation
    try:
        with open("note.txt", "r") as f:
            note = f.read()
    except:
        note = ""
    @st.cache_resource  
    def get_generator():
        return pipeline("text-generation", model="gpt2")

    gen = get_generator()
    set_seed(42)



    with st.spinner("Making a short treatment plan..."):
        prompt = f"Write a short treatment plan for {crop_name} crop affected by {predicted_label} disease."
        if note != "":
            prompt +=  note
        st.write(prompt)
        result = gen(
        prompt,
        max_length=50,
        num_return_sequences=1,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2
    )
    #st.write("**Generated Treatment:**")
    #st.write(result[0]["generated_text"])
    st.markdown("<h4 style='margin-top:30px; color:#2e7d32;'>🧪 Generated Treatment:</h4>", unsafe_allow_html=True)

    st.markdown(f"""
<div style='
    background-color:#f1f8e9;
    padding:15px;
    border-radius:10px;
    font-size:16px;
    color:#1b5e20;
    line-height:1.6;
    overflow-wrap: break-word;
'>
    {result[0]['generated_text']}
</div>
""", unsafe_allow_html=True)




# camera functionality
if crop != "Select":
    st.write(f"**{crop} selected**")
    st.markdown("<h3 style='color:white;'>🚀 Choose Input Mode:</h3>", unsafe_allow_html=True)
    input_mode = st.radio(
    "Select Input Mode",
    [" Image",  "📷 Open Camera"],
    horizontal=True,
    label_visibility="collapsed"
    )
    if input_mode == "📷 Open Camera":
        if not st.session_state['camera_on'] and not st.session_state['image_taken']:
            if st.button("📷 Open Camera"):
                st.session_state['camera_on'] = True
    elif input_mode == " Image":
        st.session_state['camera_on'] = False
        uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file
            # save uploaded image
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("crop.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            st.session_state['saved_img'] = img
            st.session_state['image_taken'] = True
            st.success("Image uploaded successfully.")

    if st.session_state['camera_on'] and not st.session_state['image_taken']:
        st.session_state['uploaded_file'] = None
        st.info("Camera is active. Click Capture when ready.")
        frame_area = st.empty()
        capture = st.button("Capture Image")

        cap = cv2.VideoCapture(1)

        while st.session_state['camera_on']:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not read from camera.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame,(320,240))
            frame_area.image(rgb_frame, channels="RGB")

            if capture:
                # save image
                cv2.imwrite("crop.jpg", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                st.session_state['saved_img'] = rgb_frame
                st.session_state['image_taken'] = True
                st.session_state['camera_on'] = False
                st.success("Image captured and saved as crop.jpg")
                frame_area.empty()
                break

        cap.release()

    # show captured image and other buttons
    if st.session_state['image_taken']:
        st.image(st.session_state['saved_img'], width=200)

        # record voice
        if st.button("🎤 Speak Your Problem"):
            st.info("Recording voice. Please speak clearly...")

            recognizer = sr.Recognizer()
            mic = sr.Microphone()

            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)

            try:
                text = recognizer.recognize_google(audio)
                st.success(f"You said: {text}")

                # Optionally, you can store this voice note in a text file
                with open("note.txt", "w") as f:
                    f.write(text)

                st.write("Voice note saved. You can now use it for treatment plan generation.")

            except sr.UnknownValueError:
                st.error("Sorry, could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Speech recognition error: {e}")


        # analyze button


        st.markdown("<h3 style='color:white;'>🧪 Analyze Image</h3>", unsafe_allow_html=True)
        if st.button("Analyze Image"):
            
            if crop == "Tomato":
                predict_image("crop.jpg", "tomato.h5", tomato_labels, tomato_treatment, "Tomato")
            elif crop == "Rice":
                predict_image("crop.jpg", "rice.h5", rice_labels, rice_treatment, "Rice")
            elif crop == "Corn":
                predict_image("crop.jpg", "corn.h5", corn_labels, corn_treatment, "Corn")
            elif crop == "Blackgram":
                predict_image("crop.jpg", "gram.h5", blackgram_labels, blackgram_treatment, "Blackgram")
            elif crop == "Surgane":
                predict_image("crop.jpg", "surgane.h5", surgane_labels, surgane_treatment, "Surgane")
            with open("note.txt", "w") as f:
                f.write("")

# reset
if st.button("Reset"):
    st.session_state['camera_on'] = False
    st.session_state['image_taken'] = False
    st.session_state['saved_img'] = False
    st.session_state['uploaded_file'] = None
    st.success("Session reset.")

