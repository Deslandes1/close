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
# Mapping of soil types to fertility and crop suitability
SOIL_TYPES = {
    "loam": {
        "fertility": "High",
        "crops": ["rice", "corn", "beans", "sorghum", "vegetables"],
        "improvement": "Maintain organic matter; rotate crops.",
        "planting_season": "Spring / Early Summer",
        "harvest_months": "3‑4 months after planting"
    },
    "clay": {
        "fertility": "Medium (but prone to waterlogging)",
        "crops": ["rice", "sugarcane", "soybeans"],
        "improvement": "Add sand and organic compost; improve drainage.",
        "planting_season": "Late Spring",
        "harvest_months": "4‑5 months after planting"
    },
    "sandy": {
        "fertility": "Low (poor water retention)",
        "crops": ["peanuts", "sweet potatoes", "carrots"],
        "improvement": "Add clay and organic matter; frequent irrigation.",
        "planting_season": "Early Spring / Autumn",
        "harvest_months": "3‑4 months after planting"
    },
    "silt": {
        "fertility": "High (but erodes easily)",
        "crops": ["corn", "beans", "wheat"],
        "improvement": "Plant cover crops; avoid over‑tilling.",
        "planting_season": "Spring",
        "harvest_months": "3‑4 months after planting"
    },
    "peat": {
        "fertility": "High (rich in organic matter)",
        "crops": ["vegetables", "berries", "potatoes"],
        "improvement": "Maintain pH; avoid over‑drainage.",
        "planting_season": "Spring",
        "harvest_months": "3‑4 months after planting"
    },
    "chalky": {
        "fertility": "Low (alkaline, stony)",
        "crops": ["barley", "sugar beets", "spinach"],
        "improvement": "Add sulphur and organic fertilisers.",
        "planting_season": "Late Spring",
        "harvest_months": "4‑5 months after planting"
    },
    "rocky": {
        "fertility": "Very Low",
        "crops": ["olives", "grapes (vines)"],
        "improvement": "Remove large rocks; build raised beds.",
        "planting_season": "Not recommended for staple crops",
        "harvest_months": "N/A"
    },
    "unknown": {
        "fertility": "Unknown",
        "crops": ["Perform soil test first"],
        "improvement": "Consult local agronomist.",
        "planting_season": "N/A",
        "harvest_months": "N/A"
    }
}

# Haitian staple crops
HAITIAN_CROPS = ["rice", "corn", "beans", "sorghum", "sugarcane", "sweet potato", "yam", "plantain"]

def soil_type_from_text(text):
    """Map user notes or AI prediction to a soil type."""
    text_lower = text.lower()
    if any(word in text_lower for word in ["loam", "loamy"]):
        return "loam"
    if any(word in text_lower for word in ["clay", "clayey"]):
        return "clay"
    if any(word in text_lower for word in ["sand", "sandy"]):
        return "sandy"
    if any(word in text_lower for word in ["silt", "silty"]):
        return "silt"
    if any(word in text_lower for word in ["peat", "peaty"]):
        return "peat"
    if any(word in text_lower for word in ["chalk", "chalky"]):
        return "chalky"
    if any(word in text_lower for word in ["rock", "rocky", "stone"]):
        return "rocky"
    return "unknown"

# -------------------------------------------------------------------
# GLOBAL DATABASE (for storing analysis logs)
# -------------------------------------------------------------------
MASTER_KEY = "20082010"
MONCASH_ID = "50947385663"

# --- SESSION STATE ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'discovery_log' not in st.session_state:
    st.session_state.discovery_log = []   # now stores agricultural reports
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'camera_method' not in st.session_state:
    st.session_state.camera_method = 'camera'
if 'current_lat' not in st.session_state:
    st.session_state.current_lat = 18.5  # Haiti approximate center
if 'current_lon' not in st.session_state:
    st.session_state.current_lon = -72.3

# --- TRANSLATIONS (simplified for agriculture – reuse most keys, add new ones) ---
# We'll extend the existing translations with new agricultural terms.
# For brevity, I'll only show English and add placeholders for others.
# In the final code, you should keep all four languages as in the original.
TRANSLATIONS = {
    'en': {
        'app_title': 'AGRICULTURAL AI ENGINE v1.0',
        'app_subtitle': 'Soil Analysis & Crop Planning',
        'owner_collab': 'Owner: <strong>Gesner Deslandes</strong> &nbsp;|&nbsp; Collaborators: Gesner Junior Deslandes, Roosevelt Deslandes, Sebastien Stephane Deslandes & Zendaya Christelle Deslandes',
        'sidebar_title': '🛡️ Access Tool',
        'sidebar_activation': 'Activation via MonCash: **{moncash}**',
        'sidebar_key_label': 'Key:',
        'sidebar_unlock': 'Unlock',
        'sidebar_invalid': 'Invalid Key',
        'sidebar_granted': '✅ ACCESS GRANTED',
        'sidebar_logout': 'Logout',
        'welcome_sound_js': """
            function playBeep() {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                oscillator.type = 'sine';
                oscillator.frequency.value = 880;
                gainNode.gain.value = 0.3;
                oscillator.start();
                gainNode.gain.exponentialRampToValueAtTime(0.00001, audioContext.currentTime + 0.5);
                oscillator.stop(audioContext.currentTime + 0.5);
            }
            playBeep();
            const url = new URL(window.location);
            url.searchParams.delete('play_sound');
            window.history.replaceState({}, document.title, url.pathname + url.search);
        """,
        'main_header': 'AGRICULTURAL AI ENGINE v1.0',
        'main_subheader': 'Empower farmers with AI soil intelligence',
        'scan_subheader': '🔍 Soil Analysis',
        'camera_method_label': 'How to capture the soil sample:',
        'camera_option': '📸 Take photo with camera (reverse button below)',
        'upload_option': '📁 Upload photo from device',
        'camera_instruction': '📸 Point the camera at the soil surface. Use the Reverse button to switch between front and rear cameras.',
        'upload_instruction': '📸 Take a photo of your soil and upload it here.',
        'reverse_button': '↻ Reverse Camera',
        'capture_button': '📷 Capture Image',
        'camera_placeholder': 'Camera feed will appear here after granting permission.',
        'site_label': 'Field Name:',
        'site_placeholder': 'North Field',
        'location_label': '📍 Field Location (Lat/Lon)',
        'location_manual': 'Manual coordinates',
        'location_auto': 'Use my current location',
        'lat_label': 'Latitude',
        'lon_label': 'Longitude',
        'get_location_button': '📍 Get My Location',
        'photo_label': 'Soil Sample Photo',
        'notes_label': 'Additional observations (e.g., soil colour, texture):',
        'weight_label': 'Field Area (hectares):',
        'execute_button': '🚀 ANALYSE SOIL',
        'no_photo_error': 'Please capture or upload an image first.',
        'report_title': 'SOIL ANALYSIS REPORT',
        'soil_type_label': 'Soil Type:',
        'fertility_label': 'Fertility Level:',
        'recommended_crops': 'Recommended Crops for this soil:',
        'improvement_label': 'How to improve this soil:',
        'planting_season_label': 'Optimal planting season:',
        'harvest_label': 'Expected harvest time:',
        'value_usd_label': 'Estimated yield value (USD): ${value:,.2f}',
        'value_htg_label': 'Estimated yield value (HTG): {value:,.2f}',
        'solution_label': 'Farmer Advice:',
        'solution_text': 'Focus on {crops}. {improvement}',
        'strategic_intel': '🌍 Field History',
        'recent_log': '**Recent Soil Analyses:**',
        'download_button': '📊 Download Analysis History (CSV)',
        'no_data_info': 'No analyses recorded yet. Perform a scan to generate data.',
        'access_warning': 'Please enter your Master Key in the sidebar to begin.',
        'language_selector': 'Language / Langue / Lang / Lang',
        'unknown_soil': 'Unknown Soil Type',
        'map_title': '🗺️ Analysed Fields Map',
        'map_marker_popup': 'Field: {site}\nSoil: {soil}\nCrops: {crops}',
        'made_in_haiti': 'Made in 🇭🇹 Haiti by GlobalInternet.py',
        'contact_info': '📞 Owner Phone: (509) 4738-5663 | 📧 Email: deslandes78@gmail.com',
    },
    'fr': { ... },  # (keep existing French dictionary with similar agricultural terms)
    'es': { ... },  # (keep existing Spanish)
    'ht': { ... },  # (keep existing Haitian Creole)
}

def get_text(key, lang=None, **kwargs):
    if lang is None:
        lang = st.session_state.language
    text = TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text

# --- IMAGE CLASSIFICATION MODEL (unchanged) ---
@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def classify_image(image_bytes):
    """
    Classify image and return a soil type based on visual cues.
    """
    try:
        model = load_model()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array, verbose=0)
        decoded = decode_predictions(preds, top=3)[0]

        # Simple mapping from ImageNet labels to soil type
        mapping = {
            'soil': 'loam', 'earth': 'loam', 'dirt': 'loam',
            'sand': 'sandy', 'sandbar': 'sandy',
            'clay': 'clay', 'pottery': 'clay',
            'rock': 'rocky', 'stone': 'rocky', 'boulder': 'rocky',
            'peat': 'peat', 'marsh': 'peat',
            'chalk': 'chalky', 'limestone': 'chalky'
        }
        for label, prob in decoded:
            label_lower = label.lower()
            for keyword, soil in mapping.items():
                if keyword in label_lower:
                    return soil, prob
        # Default
        return "unknown", decoded[0][2]
    except Exception as e:
        st.error(f"Image classification failed: {e}")
        return "unknown", 0

# --- Video processor (same) ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.image = None
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.image = img
        return frame

def camera_widget():
    webrtc_ctx = webrtc_streamer(
        key="sample-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if webrtc_ctx.video_processor:
        if st.button(get_text('capture_button'), key="capture_btn"):
            img = webrtc_ctx.video_processor.image
            if img is not None:
                success, buffer = cv2.imencode('.jpg', img)
                if success:
                    img_base64 = base64.b64encode(buffer).decode()
                    st.session_state.captured_image = f"data:image/jpeg;base64,{img_base64}"
                    st.rerun()
            else:
                st.error("No image captured. Please ensure the camera is working.")
    else:
        st.info(get_text('camera_placeholder'))

# -------------------------------------------------------------------
# UI CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Agricultural AI Engine", layout="centered")

# Display Haitian flag and branding
st.markdown("""
<div style="display: flex; justify-content: center; margin: 20px 0;">
    <svg width="320" height="192" viewBox="0 0 960 576" xmlns="http://www.w3.org/2000/svg">
        <rect width="960" height="288" fill="#00209F" />
        <rect y="288" width="960" height="288" fill="#D21034" />
    </svg>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(get_text('owner_collab'), unsafe_allow_html=True)
    st.markdown(f"*{get_text('made_in_haiti')}*")
with col2:
    lang_options = {'en': '🇺🇸 English', 'fr': '🇫🇷 Français', 'es': '🇪🇸 Español', 'ht': '🇭🇹 Kreyòl'}
    selected_lang = st.selectbox(
        get_text('language_selector'),
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(st.session_state.language)
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

st.markdown(f"<div class='main-header'><h1>{get_text('main_header')}</h1><p>{get_text('main_subheader')}</p></div>", unsafe_allow_html=True)

# Sidebar authentication
with st.sidebar:
    st.title(get_text('sidebar_title'))
    if not st.session_state.authenticated:
        st.write(get_text('sidebar_activation', moncash=MONCASH_ID))
        user_key = st.text_input(get_text('sidebar_key_label'), type="password")
        if st.button(get_text('sidebar_unlock')):
            if user_key == MASTER_KEY:
                st.session_state.authenticated = True
                st.query_params["play_sound"] = "true"
                st.rerun()
            else:
                st.error(get_text('sidebar_invalid'))
    else:
        st.success(get_text('sidebar_granted'))
        if st.button(get_text('sidebar_logout')):
            st.session_state.authenticated = False
            st.rerun()

# Welcome sound
if st.session_state.authenticated and st.query_params.get("play_sound") == "true":
    st.markdown(f"<script>{get_text('welcome_sound_js')}</script>", unsafe_allow_html=True)

if st.session_state.authenticated:
    st.subheader(get_text('scan_subheader'))

    method = st.radio(
        get_text('camera_method_label'),
        options=['camera', 'upload'],
        format_func=lambda x: get_text('camera_option') if x == 'camera' else get_text('upload_option'),
        horizontal=True
    )
    st.session_state.camera_method = method

    if method == 'camera':
        st.markdown(f"<p style='font-size:0.9rem; color:#555;'>{get_text('camera_instruction')}</p>", unsafe_allow_html=True)
        camera_widget()
        if st.session_state.captured_image:
            st.image(st.session_state.captured_image, caption="Captured soil image", width=200)
            if st.button("Clear image"):
                st.session_state.captured_image = None
                st.rerun()
    else:
        st.markdown(f"<p style='font-size:0.9rem; color:#555;'>{get_text('upload_instruction')}</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader(get_text('photo_label'), type=['jpg', 'jpeg', 'png'])
        if uploaded:
            bytes_data = uploaded.read()
            b64 = base64.b64encode(bytes_data).decode()
            st.session_state.captured_image = f"data:image/{uploaded.type.split('/')[-1]};base64,{b64}"
            st.rerun()

    site = st.text_input(get_text('site_label'), get_text('site_placeholder'))

    # Location input (same as before)
    st.subheader(get_text('location_label'))
    loc_method = st.radio(
        "",
        options=['manual', 'auto'],
        format_func=lambda x: get_text('location_manual') if x == 'manual' else get_text('location_auto'),
        horizontal=True,
        key="loc_method"
    )
    if loc_method == 'manual':
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input(get_text('lat_label'), value=st.session_state.current_lat, format="%.6f")
        with col_lon:
            lon = st.number_input(get_text('lon_label'), value=st.session_state.current_lon, format="%.6f")
        if st.button(get_text('get_location_button'), key="manual_loc"):
            st.session_state.current_lat = lat
            st.session_state.current_lon = lon
            st.success(f"Location set to {lat:.5f}, {lon:.5f}")
    else:
        st.markdown("""
            <script>
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        const lat = position.coords.latitude;
                        const lon = position.coords.longitude;
                        const url = new URL(window.location.href);
                        url.searchParams.set('lat', lat);
                        url.searchParams.set('lon', lon);
                        window.location.href = url.href;
                    },
                    (error) => {
                        alert("Geolocation error: " + error.message);
                    }
                );
            } else {
                alert("Geolocation is not supported by your browser.");
            }
            </script>
        """, unsafe_allow_html=True)
        query_params = st.query_params
        lat_param = query_params.get("lat")
        lon_param = query_params.get("lon")
        if lat_param is not None and lon_param is not None:
            try:
                st.session_state.current_lat = float(lat_param)
                st.session_state.current_lon = float(lon_param)
                st.success(f"📍 Location acquired: {st.session_state.current_lat:.5f}, {st.session_state.current_lon:.5f}")
                st.query_params.clear()
            except:
                pass
        st.write(f"Current location: **{st.session_state.current_lat:.5f}, {st.session_state.current_lon:.5f}**")
        if st.button(get_text('get_location_button'), key="auto_loc"):
            st.experimental_rerun()
        lat = st.session_state.current_lat
        lon = st.session_state.current_lon

    notes = st.text_area(get_text('notes_label'))
    area_hectares = st.number_input(get_text('weight_label'), value=1.0, min_value=0.1, step=0.1)

    if st.button(get_text('execute_button')):
        if st.session_state.captured_image:
            # Extract image bytes
            img_data = st.session_state.captured_image
            if img_data.startswith('data:image'):
                img_base64 = img_data.split(',')[1]
                img_bytes = base64.b64decode(img_base64)
            else:
                img_bytes = img_data.encode()

            # Classify soil type from image
            with st.spinner("Analysing soil with AI..."):
                soil_type, confidence = classify_image(img_bytes)

            # Optionally refine using notes
            if notes.strip():
                notes_soil = soil_type_from_text(notes)
                if notes_soil != "unknown":
                    soil_type = notes_soil

            # Get soil data
            soil_info = SOIL_TYPES.get(soil_type, SOIL_TYPES["unknown"])
            fertility = soil_info["fertility"]
            recommended_crops = soil_info["crops"]
            improvement = soil_info["improvement"]
            planting_season = soil_info["planting_season"]
            harvest = soil_info["harvest_months"]

            # Estimate yield value (simple dummy calculation)
            # Assume average yield per hectare for a generic crop
            base_yield_usd = 800  # USD per hectare (placeholder)
            if soil_type == "loam":
                multiplier = 1.2
            elif soil_type == "clay":
                multiplier = 0.9
            elif soil_type == "sandy":
                multiplier = 0.6
            else:
                multiplier = 0.5
            estimated_value_usd = base_yield_usd * area_hectares * multiplier
            estimated_value_htg = estimated_value_usd * 131.19  # HTG rate

            rep_id = f"AGRI-{uuid.uuid4().hex[:6].upper()}"

            st.session_state.discovery_log.append({
                "Date": str(datetime.date.today()),
                "ID": rep_id,
                "Soil_Type": soil_type.upper(),
                "Fertility": fertility,
                "Site": site,
                "Latitude": lat,
                "Longitude": lon,
                "Area_ha": area_hectares,
                "Est_Value_USD": estimated_value_usd,
                "Recommended_Crops": ", ".join(recommended_crops[:3]),
                "AI_Confidence": f"{confidence:.2f}"
            })

            # Build report
            report_html = f"""
            <div class="report-card">
                <h2 style="color:#D21034; text-align:center;">{get_text('report_title')}</h2>
                <hr>
                <p><b>{get_text('soil_type_label')}</b> {soil_type.capitalize()} (AI confidence: {confidence:.2%})</p>
                <p><b>{get_text('fertility_label')}</b> {fertility}</p>
                <p><b>{get_text('recommended_crops')}</b> {', '.join(recommended_crops[:5])}</p>
                <p><b>{get_text('improvement_label')}</b> {improvement}</p>
                <p><b>{get_text('planting_season_label')}</b> {planting_season}</p>
                <p><b>{get_text('harvest_label')}</b> {harvest}</p>
                <h3 style="color:green;">{get_text('value_usd_label', value=estimated_value_usd)}</h3>
                <h3 style="color:#00209F;">{get_text('value_htg_label', value=estimated_value_htg)}</h3>
                <hr>
                <p><b>{get_text('solution_label')}</b> {get_text('solution_text', crops=', '.join(recommended_crops[:2]), improvement=improvement)}</p>
            </div>
            """
            st.markdown(report_html, unsafe_allow_html=True)
        else:
            st.error(get_text('no_photo_error'))

    # Map section
    st.divider()
    st.subheader(get_text('map_title'))
    if st.session_state.discovery_log:
        df_map = pd.DataFrame(st.session_state.discovery_log)
        df_map = df_map.dropna(subset=['Latitude', 'Longitude'])
        if not df_map.empty:
            center_lat = df_map['Latitude'].mean()
            center_lon = df_map['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")
            for _, row in df_map.iterrows():
                popup_text = get_text('map_marker_popup', site=row['Site'], soil=row['Soil_Type'], crops=row['Recommended_Crops'])
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=popup_text,
                    icon=folium.Icon(color="green", icon="leaf")
                ).add_to(m)
            folium_static(m, width=700, height=500)
        else:
            st.info("No fields with location data yet. Add coordinates to see them on the map.")
    else:
        st.info(get_text('no_data_info'))

    # History section
    st.divider()
    st.subheader(get_text('strategic_intel'))
    if st.session_state.discovery_log:
        st.markdown(get_text('recent_log'))
        df = pd.DataFrame(st.session_state.discovery_log)
        st.dataframe(df.tail(5), width='stretch')
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=get_text('download_button'),
            data=csv,
            file_name=f"Agricultural_History_{datetime.date.today()}.csv",
            mime='text/csv',
        )
    else:
        st.info(get_text('no_data_info'))

    # Contact info footer
    st.markdown("---")
    st.markdown(get_text('contact_info'))
else:
    st.warning(get_text('access_warning'))
