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

        "planting_season": {"en": "Spring / Early Summer",

                            "fr": "Printemps / Début d'été",

                            "es": "Primavera / Principios de verano",

                            "ht": "Prentan / Kòmansman ete"},

        "harvest_months": {"en": "3‑4 months after planting",

                           "fr": "3‑4 mois après la plantation",

                           "es": "3‑4 meses después de la siembra",

                           "ht": "3‑4 mwa apre plante"}

    },

    "clay": {

        "fertility": {"en": "Medium (but prone to waterlogging)", "fr": "Moyenne (mais sensible à l'engorgement)", "es": "Media (propensa al encharcamiento)", "ht": "Mwayen (men fasil pou gen dlo kouche)"},

        "crops": {"en": ["rice", "sugarcane", "soybeans"], "fr": ["riz", "canne à sucre", "soja"], "es": ["arroz", "caña de azúcar", "soja"], "ht": ["diri", "kann", "soya"]},

        "improvement": {"en": "Add sand and organic compost; improve drainage.", "fr": "Ajoutez du sable et du compost organique; améliorez le drainage.", "es": "Añada arena y compost orgánico; mejore el drenaje.", "ht": "Ajoute sab ak konpò òganik; amelyore drenaj."},

        "planting_season": {"en": "Late Spring", "fr": "Fin du printemps", "es": "Finales de primavera", "ht": "Prentan an reta"},

        "harvest_months": {"en": "4‑5 months after planting", "fr": "4‑5 mois après la plantation", "es": "4‑5 meses después de la siembra", "ht": "4‑5 mwa apre plante"}

    },

    "sandy": {

        "fertility": {"en": "Low (poor water retention)", "fr": "Faible (mauvaise rétention d'eau)", "es": "Baja (mala retención de agua)", "ht": "Ba (pa kenbe dlo)"},

        "crops": {"en": ["peanuts", "sweet potatoes", "carrots"], "fr": ["arachides", "patates douces", "carottes"], "es": ["cacahuetes", "batatas", "zanahorias"], "ht": ["pistach", "patat", "kawòt"]},

        "improvement": {"en": "Add clay and organic matter; frequent irrigation.", "fr": "Ajoutez de l'argile et de la matière organique; irrigation fréquente.", "es": "Añada arcilla y materia orgánica; riego frecuente.", "ht": "Ajoute ajil ak matyè òganik; irigasyon souvan."},

        "planting_season": {"en": "Early Spring / Autumn", "fr": "Début du printemps / Automne", "es": "Principios de primavera / Otoño", "ht": "Prentan bonè / Otòn"},

        "harvest_months": {"en": "3‑4 months after planting", "fr": "3‑4 mois après la plantation", "es": "3‑4 meses después de la siembra", "ht": "3‑4 mwa apre plante"}

    },

    "silt": {

        "fertility": {"en": "High (but erodes easily)", "fr": "Élevée (mais s'érode facilement)", "es": "Alta (pero se erosiona fácilmente)", "ht": "Wòl (men fasil pou erode)"},

        "crops": {"en": ["corn", "beans", "wheat"], "fr": ["maïs", "haricots", "blé"], "es": ["maíz", "frijoles", "trigo"], "ht": ["mayi", "pwa", "ble"]},

        "improvement": {"en": "Plant cover crops; avoid over‑tilling.", "fr": "Plantez des cultures de couverture; évitez le sur‑labourage.", "es": "Siembre cultivos de cobertura; evite el exceso de labranza.", "ht": "Plante rekòt kouvèti; evite twòp travay tè."},

        "planting_season": {"en": "Spring", "fr": "Printemps", "es": "Primavera", "ht": "Prentan"},

        "harvest_months": {"en": "3‑4 months after planting", "fr": "3‑4 mois après la plantation", "es": "3‑4 meses después de la siembra", "ht": "3‑4 mwa apre plante"}

    },

    "peat": {

        "fertility": {"en": "High (rich in organic matter)", "fr": "Élevée (riche en matière organique)", "es": "Alta (rica en materia orgánica)", "ht": "Wòl (rich an matyè òganik)"},

        "crops": {"en": ["vegetables", "berries", "potatoes"], "fr": ["légumes", "baies", "pommes de terre"], "es": ["vegetales", "bayas", "papas"], "ht": ["legim", "bè", "pòmdetè"]},

        "improvement": {"en": "Maintain pH; avoid over‑drainage.", "fr": "Maintenez le pH; évitez le sur‑drainage.", "es": "Mantenga el pH; evite el exceso de drenaje.", "ht": "Kenbe pH; evite twòp drenaj."},

        "planting_season": {"en": "Spring", "fr": "Printemps", "es": "Primavera", "ht": "Prentan"},

        "harvest_months": {"en": "3‑4 months after planting", "fr": "3‑4 mois après la plantation", "es": "3‑4 meses después de la siembra", "ht": "3‑4 mwa apre plante"}

    },

    "chalky": {

        "fertility": {"en": "Low (alkaline, stony)", "fr": "Faible (alcalin, pierreux)", "es": "Baja (alcalino, pedregoso)", "ht": "Ba (alkalin, gen wòch)"},

        "crops": {"en": ["barley", "sugar beets", "spinach"], "fr": ["orge", "betteraves sucrières", "épinards"], "es": ["cebada", "remolacha azucarera", "espinacas"], "ht": ["lòrj", "bètrav", "zepina"]},

        "improvement": {"en": "Add sulphur and organic fertilisers.", "fr": "Ajoutez du soufre et des engrais organiques.", "es": "Añada azufre y fertilizantes orgánicos.", "ht": "Ajoute souf ak angrè òganik."},

        "planting_season": {"en": "Late Spring", "fr": "Fin du printemps", "es": "Finales de primavera", "ht": "Prentan an reta"},

        "harvest_months": {"en": "4‑5 months after planting", "fr": "4‑5 mois après la plantation", "es": "4‑5 meses después de la siembra", "ht": "4‑5 mwa apre plante"}

    },

    "rocky": {

        "fertility": {"en": "Very Low", "fr": "Très faible", "es": "Muy baja", "ht": "Trè ba"},

        "crops": {"en": ["olives", "grapes (vines)"], "fr": ["olives", "raisins (vigne)"], "es": ["aceitunas", "uvas (vid)"], "ht": ["oliv", "rezen (pye rezen)"]},

        "improvement": {"en": "Remove large rocks; build raised beds.", "fr": "Enlevez les grosses pierres; construisez des plates‑bandes surélevées.", "es": "Retire las rocas grandes; construya camas elevadas.", "ht": "Retire gwo wòch; konstruire kabann ki wo."},

        "planting_season": {"en": "Not recommended for staple crops", "fr": "Non recommandé pour les cultures de base", "es": "No recomendado para cultivos básicos", "ht": "Pa rekòmande pou rekòt debaz"},

        "harvest_months": {"en": "N/A", "fr": "N/A", "es": "N/A", "ht": "N/A"}

    },

    "unknown": {

        "fertility": {"en": "Unknown", "fr": "Inconnue", "es": "Desconocida", "ht": "Enkoni"},

        "crops": {"en": ["Perform soil test first"], "fr": ["Effectuez d'abord une analyse de sol"], "es": ["Realice primero una prueba de suelo"], "ht": ["Fè tès tè an premye"]},

        "improvement": {"en": "Consult local agronomist.", "fr": "Consultez un agronome local.", "es": "Consulte a un agrónomo local.", "ht": "Konsilte yon agwonòm lokal."},

        "planting_season": {"en": "N/A", "fr": "N/A", "es": "N/A", "ht": "N/A"},

        "harvest_months": {"en": "N/A", "fr": "N/A", "es": "N/A", "ht": "N/A"}

    }

}



# -------------------------------------------------------------------

# GLOBAL DATABASE & SESSION STATE

# -------------------------------------------------------------------

MASTER_KEY = "20082010"

MONCASH_ID = "50947385663"



if 'authenticated' not in st.session_state:

    st.session_state.authenticated = False

if 'discovery_log' not in st.session_state:

    st.session_state.discovery_log = []

if 'language' not in st.session_state:

    st.session_state.language = 'en'

if 'captured_image' not in st.session_state:

    st.session_state.captured_image = None

if 'camera_method' not in st.session_state:

    st.session_state.camera_method = 'camera'

if 'current_lat' not in st.session_state:

    st.session_state.current_lat = 18.5

if 'current_lon' not in st.session_state:

    st.session_state.current_lon = -72.3



# -------------------------------------------------------------------

# TRANSLATIONS

# -------------------------------------------------------------------

TRANSLATIONS = {

    'en': {

        'app_title': 'AGRICULTURAL AI ENGINE v1.0',

        'app_subtitle': 'Soil Analysis & Crop Planning',

        'owner_collab': 'Owner: <strong>Gesner Deslandes</strong> &nbsp;|&nbsp; Collaborators: Gesner Junior Deslandes, Roosevelt Deslandes, Sebastien Stephane Deslandes & Zendaya Christelle Deslandes',

        'made_in_haiti': 'Made in 🇭🇹 Haiti by GlobalInternet.py',

        'contact_info': '📞 Owner Phone: (509) 4738-5663 | 📧 Email: deslandes78@gmail.com',

        'sidebar_title': '🛡️ Access Tool',

        'sidebar_activation': 'Activation via MonCash: **{moncash}**',

        'sidebar_key_label': 'Key:',

        'sidebar_unlock': 'Unlock',

        'sidebar_invalid': 'Invalid Key',

        'sidebar_granted': '✅ ACCESS GRANTED',

        'sidebar_logout': 'Logout',

        'welcome_sound_js': "console.log('Access granted');",

        'main_header': 'AGRICULTURAL AI ENGINE v1.0',

        'main_subheader': 'Empower farmers with AI soil intelligence',

        'scan_subheader': '🔍 Soil Analysis',

        'camera_method_label': 'How to capture the soil sample:',

        'camera_option': '📸 Take photo with camera',

        'upload_option': '📁 Upload photo from device',

        'camera_instruction': '📸 Point the camera at the soil surface.',

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

        'notes_label': 'Additional observations:',

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

        'no_data_info': 'No analyses recorded yet.',

        'access_warning': 'Please enter your Master Key in the sidebar to begin.',

        'language_selector': 'Language / Langue',

        'unknown_soil': 'Unknown Soil Type',

        'map_title': '🗺️ Analysed Fields Map',

        'map_marker_popup': 'Field: {site}\nSoil: {soil}\nCrops: {crops}',

        'translate_report': '🌐 Translate this report',

        'report_translated': 'Report translated to {lang}'

    },

    'fr': {

        'app_title': 'MOTEUR IA AGRICOLE v1.0',

        'app_subtitle': 'Analyse du sol et planification des cultures',

        'owner_collab': 'Propriétaire: <strong>Gesner Deslandes</strong>',

        'made_in_haiti': 'Fabriqué en 🇭🇹 Haïti par GlobalInternet.py',

        'contact_info': '📞 (509) 4738-5663 | 📧 deslandes78@gmail.com',

        'sidebar_title': '🛡️ Accès à l’outil',

        'sidebar_activation': 'Activation via MonCash: **{moncash}**',

        'sidebar_key_label': 'Clé:',

        'sidebar_unlock': 'Déverrouiller',

        'sidebar_invalid': 'Clé invalide',

        'sidebar_granted': '✅ ACCÈS AUTORISÉ',

        'sidebar_logout': 'Déconnexion',

        'welcome_sound_js': "console.log('Accès autorisé');",

        'main_header': 'MOTEUR IA AGRICOLE v1.0',

        'main_subheader': 'Donnez du pouvoir aux agriculteurs grâce à l’IA',

        'scan_subheader': '🔍 Analyse du sol',

        'camera_method_label': 'Méthode de capture:',

        'camera_option': '📸 Prendre une photo',

        'upload_option': '📁 Télécharger une photo',

        'camera_instruction': '📸 Pointez la caméra vers le sol.',

        'upload_instruction': '📸 Téléchargez une photo de votre sol.',

        'reverse_button': '↻ Retourner la caméra',

        'capture_button': '📷 Capturer l’image',

        'camera_placeholder': 'Le flux vidéo apparaîtra ici.',

        'site_label': 'Nom du champ:',

        'site_placeholder': 'Champ Nord',

        'location_label': '📍 Emplacement (Lat/Lon)',

        'location_manual': 'Coordonnées manuelles',

        'location_auto': 'Utiliser ma position',

        'lat_label': 'Latitude',

        'lon_label': 'Longitude',

        'get_location_button': '📍 Obtenir ma position',

        'photo_label': 'Photo de l’échantillon',

        'notes_label': 'Observations supplémentaires:',

        'weight_label': 'Superficie (hectares):',

        'execute_button': '🚀 ANALYSER LE SOL',

        'no_photo_error': 'Veuillez d’abord capturer ou télécharger une image.',

        'report_title': 'RAPPORT D’ANALYSE DU SOL',

        'soil_type_label': 'Type de sol:',

        'fertility_label': 'Fertilité:',

        'recommended_crops': 'Cultures recommandées:',

        'improvement_label': 'Amélioration:',

        'planting_season_label': 'Saison optimale:',

        'harvest_label': 'Récolte prévue:',

        'value_usd_label': 'Valeur estimée (USD) : ${value:,.2f}',

        'value_htg_label': 'Valeur estimée (HTG) : {value:,.2f}',

        'solution_label': 'Conseil:',

        'solution_text': 'Concentrez-vous sur {crops}. {improvement}',

        'strategic_intel': '🌍 Historique',

        'recent_log': '**Analyses récentes:**',

        'download_button': '📊 Télécharger (CSV)',

        'no_data_info': 'Aucune analyse enregistrée.',

        'access_warning': 'Entrez votre clé principale.',

        'language_selector': 'Langue / Language',

        'unknown_soil': 'Type de sol inconnu',

        'map_title': '🗺️ Carte',

        'map_marker_popup': 'Champ: {site}\nSol: {soil}',

        'translate_report': '🌐 Traduire',

        'report_translated': 'Traduit en {lang}'

    }

}



def get_text(key, lang=None, **kwargs):

    if lang is None:

        lang = st.session_state.language

    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS['en'])

    text = lang_dict.get(key, key)

    if kwargs:

        return text.format(**kwargs)

    return text



# -------------------------------------------------------------------

# IMAGE CLASSIFICATION

# -------------------------------------------------------------------

@st.cache_resource

def load_model():

    return MobileNetV2(weights='imagenet')



def classify_image(image_bytes):

    try:

        model = load_model()

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        img = img.resize((224, 224))

        img_array = np.array(img)

        img_array = np.expand_dims(img_array, axis=0)

        img_array = preprocess_input(img_array)



        preds = model.predict(img_array, verbose=0)

        decoded = decode_predictions(preds, top=3)[0]



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

        return "unknown", decoded[0][2]

    except Exception as e:

        st.error(f"Image classification failed: {e}")

        return "unknown", 0



# -------------------------------------------------------------------

# VIDEO PROCESSOR

# -------------------------------------------------------------------

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



# Haitian flag

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



st.markdown(f"<div style='text-align:center;'><h1>{get_text('main_header')}</h1><p>{get_text('main_subheader')}</p></div>", unsafe_allow_html=True)



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



# Logic flow after Auth

if st.session_state.authenticated:

    st.subheader(get_text('scan_subheader'))



    # FIXED LINE 565: Added descriptive label and visibility: collapsed

    method = st.radio(

        label="Capture Selection Method",

        options=['camera', 'upload'],

        format_func=lambda x: get_text('camera_option') if x == 'camera' else get_text('upload_option'),

        horizontal=True,

        label_visibility="collapsed"

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



    # Form details

    site = st.text_input(get_text('site_label'), placeholder=get_text('site_placeholder'))

    

    loc_col1, loc_col2 = st.columns(2)

    with loc_col1:

        lat = st.number_input(get_text('lat_label'), value=st.session_state.current_lat, format="%.6f")

    with loc_col2:

        lon = st.number_input(get_text('lon_label'), value=st.session_state.current_lon, format="%.6f")



    notes = st.text_area(get_text('notes_label'))

    area = st.number_input(get_text('weight_label'), min_value=0.1, value=1.0)



    # Execution

    if st.button(get_text('execute_button')):

        if st.session_state.captured_image:

            # Classification

            header, encoded = st.session_state.captured_image.split(",", 1)

            img_bytes = base64.b64decode(encoded)

            soil_key, prob = classify_image(img_bytes)

            

            soil_info = SOIL_TYPES.get(soil_key, SOIL_TYPES['unknown'])

            

            st.write(f"---")

            st.success(get_text('report_title'))

            st.write(f"**{get_text('soil_type_label')}** {soil_key.capitalize()}")

            st.write(f"**{get_text('fertility_label')}** {soil_info['fertility'].get(st.session_state.language)}")

            st.write(f"**{get_text('recommended_crops')}** {', '.join(soil_info['crops'].get(st.session_state.language))}")

            st.info(f"**{get_text('improvement_label')}** {soil_info['improvement'].get(st.session_state.language)}")

        else:

            st.error(get_text('no_photo_error'))

else:

    st.warning(get_text('access_warning'))
