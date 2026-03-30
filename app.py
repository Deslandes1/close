import streamlit as st
import datetime
import uuid
import pandas as pd
import base64
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import io
import folium
from streamlit_folium import folium_static
import json

# -------------------------------------------------------------------
# AGRICULTURAL KNOWLEDGE BASE
# -------------------------------------------------------------------
SOIL_TYPES = {
    "loam": {
        "fertility": {"en": "High", "fr": "Élevée", "es": "Alta", "ht": "Wòl"},
        "crops": {"en": ["rice", "corn", "beans", "sorghum", "vegetables"],
                  "fr": ["riz", "maïs", "haricots", "sorgho", "légumes"],
                  "es": ["arroz", "maíz", "frijoles", "sorgo", "vegetales"],
                  "ht": ["diri", "mayi", "pwa", "sorgo", "legim"]},
        "improvement": {"en": "Maintain organic matter; rotate crops.",
                        "fr": "Maintenez la matière organique; pratiquez la rotation des cultures.",
                        "es": "Mantenga la materia orgánica; rote los cultivos.",
                        "ht": "Kenbe matyè òganik; fè wotasyon rekòt."},
        "planting_season": {"en": "Spring / Early Summer", "fr": "Printemps / Début d'été", "es": "Primavera / Principios de verano", "ht": "Prentan / Kòmansman ete"},
        "harvest_months": {"en": "3‑4 months after planting", "fr": "3‑4 mois après la plantation", "es": "3‑4 meses después de la siembra", "ht": "3‑4 mwa apre plante"}
    },
    "clay": {
        "fertility": {"en": "Medium (waterlogging risk)", "fr": "Moyenne (risque d'engorgement)", "es": "Media (riesgo de encharcamiento)", "ht": "Mwayen (ris dlo kouche)"},
        "crops": {"en": ["rice", "sugarcane", "soybeans"], "fr": ["riz", "canne à sucre", "soja"], "es": ["arroz", "caña de azúcar", "soja"], "ht": ["diri", "kann", "soya"]},
        "improvement": {"en": "Add sand and compost; improve drainage.", "fr": "Ajoutez du sable et du compost; améliorez le drainage.", "es": "Añada arena y compost; mejore el drenaje.", "ht": "Ajoute sab ak konpò; amelyore drenaj."},
        "planting_season": {"en": "Late Spring", "fr": "Fin du printemps", "es": "Finales de primavera", "ht": "Prentan an reta"},
        "harvest_months": {"en": "4‑5 months after planting", "fr": "4‑5 mois après la plantation", "es": "4‑5 meses después de la siembra", "ht": "4‑5 mwa apre plante"}
    },
    "sandy": {
        "fertility": {"en": "Low (leaching risk)", "fr": "Faible (risque de lessivage)", "es": "Baja (riesgo de lixiviación)", "ht": "Ba (ris pèdi angrè)"},
        "crops": {"en": ["peanuts", "sweet potatoes", "carrots"], "fr": ["arachides", "patates douces", "carottes"], "es": ["cacahuetes", "batatas", "zanahorias"], "ht": ["pistach", "patat", "kawòt"]},
        "improvement": {"en": "Add organic matter; frequent irrigation.", "fr": "Ajoutez de la matière organique; irrigation fréquente.", "es": "Añada materia orgánica; riego frecuente.", "ht": "Ajoute matyè òganik; irigasyon souvan."},
        "planting_season": {"en": "Early Spring / Autumn", "fr": "Début du printemps / Automne", "es": "Principios de primavera / Otoño", "ht": "Prentan bonè / Otòn"},
        "harvest_months": {"en": "3‑4 months after planting", "fr": "3‑4 mois après la plantation", "es": "3‑4 meses después de la siembra", "ht": "3‑4 mwa apre plante"}
    },
    "unknown": {
        "fertility": {"en": "Unknown", "fr": "Inconnue", "es": "Desconocida", "ht": "Enkoni"},
        "crops": {"en": ["General Veggies"], "fr": ["Légumes généraux"], "es": ["Vegetales generales"], "ht": ["Legim jeneral"]},
        "improvement": {"en": "Consult local agronomist.", "fr": "Consultez un agronome.", "es": "Consulte a un agrónomo.", "ht": "Konsilte yon agwonòm."},
        "planting_season": {"en": "N/A", "fr": "N/A", "es": "N/A", "ht": "N/A"},
        "harvest_months": {"en": "N/A", "fr": "N/A", "es": "N/A", "ht": "N/A"}
    }
}

# -------------------------------------------------------------------
# CONFIG & ASSETS
# -------------------------------------------------------------------
MASTER_KEY = "20082010"

if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'discovery_log' not in st.session_state: st.session_state.discovery_log = []
if 'language' not in st.session_state: st.session_state.language = 'en'

HAITIAN_FLAG_HTML = """
<div style="display: flex; justify-content: center; margin: 10px 0;">
    <svg width="240" height="144" viewBox="0 0 960 576" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        <defs>
            <clipPath id="a"><path d="M0 0h960v576H0z"/></clipPath>
            <image id="symbol" width="131" height="114" xlink:href="data:image/png;base64,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"/>
        </defs>
        <g clip-path="url(#a)">
            <rect width="960" height="288" fill="#00209F" />
            <rect y="288" width="960" height="288" fill="#D21034" />
            <rect x="360" y="216" width="240" height="144" fill="#FFFFFF" />
            <use xlink:href="#symbol" x="414.5" y="231"/>
        </g>
    </svg>
</div>
"""

# -------------------------------------------------------------------
# MODEL & VIDEO LOGIC
# -------------------------------------------------------------------
@st.cache_resource
def load_agri_model():
    return MobileNetV2(weights='imagenet')

class SoilProcessor(VideoProcessorBase):
    def __init__(self): self.frame = None
    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(self.frame, format="bgr24")

def classify_soil(image_bytes):
    try:
        model = load_agri_model()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
        x = preprocess_input(np.expand_dims(np.array(img), axis=0))
        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=3)[0]
        # Mapping logic (simplified for this block)
        labels = [d[1].lower() for d in decoded]
        if any(s in labels for s in ['sand', 'desert']): return 'sandy'
        if any(s in labels for s in ['mud', 'clay']): return 'clay'
        return 'loam'
    except: return 'unknown'

# -------------------------------------------------------------------
# MAIN APP EXECUTION
# -------------------------------------------------------------------
st.set_page_config(page_title="EduHumanity Agri-AI", layout="wide")

# Sidebar Logic
with st.sidebar:
    st.markdown(HAITIAN_FLAG_HTML.replace('width="240"', 'width="150"').replace('height="144"', 'height="90"'), unsafe_allow_html=True)
    st.title("🛡️ Secure Access")
    if not st.session_state.authenticated:
        key = st.text_input("Master Key:", type="password")
        if st.button("Unlock Engine"):
            if key == MASTER_KEY:
                st.session_state.authenticated = True
                st.rerun()
            else: st.error("Access Denied")
    else:
        st.success("Verified Account")
        if st.button("Logout"): 
            st.session_state.authenticated = False
            st.rerun()
    st.divider()
    st.info("Founder: Gesner Deslandes\nEduHumanity 2026")

# App Interface
if not st.session_state.authenticated:
    st.markdown(HAITIAN_FLAG_HTML, unsafe_allow_html=True)
    st.header("AGRICULTURAL AI ENGINE v1.0")
    st.write("Welcome to the EduHumanity Resource Detection portal. Please authenticate to use the AI classification tools.")
else:
    st.markdown(HAITIAN_FLAG_HTML, unsafe_allow_html=True)
    st.title("AGRICULTURAL AI ENGINE v1.0")
    
    col_input, col_map = st.columns([1, 1])

    with col_input:
        st.subheader("🔍 Soil Analysis")
        lang = st.selectbox("Language", ["en", "fr", "es", "ht"])
        st.session_state.language = lang
        
        mode = st.radio("Capture Mode", ["Live Camera", "Upload Image"])
        captured_img = None

        if mode == "Live Camera":
            ctx = webrtc_streamer(key="soil-cam", video_processor_factory=SoilProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            if st.button("Capture Frame"):
                if ctx.video_processor and ctx.video_processor.frame is not None:
                    captured_img = ctx.video_processor.frame
                    st.image(captured_img, caption="Captured", width=300)
        else:
            file = st.file_uploader("Upload Soil Sample", type=["jpg", "png"])
            if file: 
                captured_img = np.array(Image.open(file))
                st.image(captured_img, width=300)

        site = st.text_input("Field Location Name", "My Farm")
        if st.button("🚀 EXECUTE AI SCAN"):
            if captured_img is not None:
                # Convert to bytes for classification
                is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(captured_img, cv2.COLOR_RGB2BGR))
                soil_type = classify_soil(buffer.tobytes())
                
                res = SOIL_TYPES.get(soil_type, SOIL_TYPES['unknown'])
                st.session_state.discovery_log.append({"site": site, "soil": soil_type, "lat": 18.5, "lon": -72.3})
                
                st.markdown(f"### Results for {site}")
                st.write(f"**Type:** {soil_type.upper()}")
                st.write(f"**Fertility:** {res['fertility'][lang]}")
                st.write(f"**Recommended:** {', '.join(res['crops'][lang])}")
                st.success(f"**Action Plan:** {res['improvement'][lang]}")

    with col_map:
        st.subheader("🗺️ Field Mapping")
        m = folium.Map(location=[18.5, -72.3], zoom_start=8)
        for entry in st.session_state.discovery_log:
            folium.Marker([entry['lat'], entry['lon']], popup=f"{entry['site']}: {entry['soil']}").add_to(m)
        folium_static(m)

st.divider()
st.markdown("<p style='text-align: center;'>Propriété de Gesner Deslandes | Made in 🇭🇹 Haiti</p>", unsafe_allow_html=True)
