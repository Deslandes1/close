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
def soil_type_from_text(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["loam", "loamy"]):
        return "loam"
    if any(w in text_lower for w in ["clay", "clayey"]):
        return "clay"
    if any(w in text_lower for w in ["sand", "sandy"]):
        return "sandy"
    if any(w in text_lower for w in ["silt", "silty"]):
        return "silt"
    if any(w in text_lower for w in ["peat", "peaty"]):
        return "peat"
    if any(w in text_lower for w in ["chalk", "chalky"]):
        return "chalky"
    if any(w in text_lower for w in ["rock", "rocky", "stone"]):
        return "rocky"
    return "unknown"

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
# TRANSLATIONS (full for all UI elements)
# -------------------------------------------------------------------
TRANSLATIONS = {
    'en': {
        'app_title': 'AGRICULTURAL AI ENGINE v1.0',
        'app_subtitle': 'Soil Analysis & Crop Planning',
        'owner_collab': 'Owner: Gesner Deslandes  | Collaborators: Gesner Junior Deslandes, Roosevelt Deslandes, Sebastien Stephane Deslandes & Zendaya Christelle Deslandes',
        'made_in_haiti': 'Made in 🇭🇹 Haiti by GlobalInternet.py',
        'contact_info': '📞 Owner Phone: (509) 4738-5663 | 📧 Email: deslandes78@gmail.com',
        'sidebar_title': '🛡️ Access Tool',
        'sidebar_activation': 'Activation via MonCash: {moncash}',
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
        'recent_log': 'Recent Soil Analyses:',
        'download_button': '📊 Download Analysis History (CSV)',
        'no_data_info': 'No analyses recorded yet. Perform a scan to generate data.',
        'access_warning': 'Please enter your Master Key in the sidebar to begin.',
        'language_selector': 'Language / Langue / Lang / Lang',
        'unknown_soil': 'Unknown Soil Type',
        'map_title': '🗺️ Analysed Fields Map',
        'map_marker_popup': 'Field: {site}\nSoil: {soil}\nCrops: {crops}',
        'translate_report': '🌐 Translate this report',
        'report_translated': 'Report translated to {lang}'
    },
    'fr': {
        'app_title': 'MOTEUR IA AGRICOLE v1.0',
        'app_subtitle': 'Analyse du sol et planification des cultures',
        'owner_collab': 'Propriétaire: Gesner Deslandes  | Collaborateurs: Gesner Junior Deslandes, Roosevelt Deslandes, Sebastien Stephane Deslandes & Zendaya Christelle Deslandes',
        'made_in_haiti': 'Fabriqué en 🇭🇹 Haïti par GlobalInternet.py',
        'contact_info': '📞 Téléphone du propriétaire: (509) 4738-5663 | 📧 Email: deslandes78@gmail.com',
        'sidebar_title': '🛡️ Accès à l’outil',
        'sidebar_activation': 'Activation via MonCash: {moncash}',
        'sidebar_key_label': 'Clé:',
        'sidebar_unlock': 'Déverrouiller',
        'sidebar_invalid': 'Clé invalide',
        'sidebar_granted': '✅ ACCÈS AUTORISÉ',
        'sidebar_logout': 'Déconnexion',
        'welcome_sound_js': """...""",
        'main_header': 'MOTEUR IA AGRICOLE v1.0',
        'main_subheader': 'Donnez du pouvoir aux agriculteurs grâce à l’IA',
        'scan_subheader': '🔍 Analyse du sol',
        'camera_method_label': 'Comment capturer l’échantillon de sol:',
        'camera_option': '📸 Prendre une photo avec la caméra (bouton de retournement ci-dessous)',
        'upload_option': '📁 Télécharger une photo depuis l’appareil',
        'camera_instruction': '📸 Pointez la caméra vers la surface du sol. Utilisez le bouton Retournement pour passer entre caméra avant et arrière.',
        'upload_instruction': '📸 Prenez une photo de votre sol et téléchargez-la ici.',
        'reverse_button': '↻ Retourner la caméra',
        'capture_button': '📷 Capturer l’image',
        'camera_placeholder': 'Le flux vidéo apparaîtra ici après autorisation.',
        'site_label': 'Nom du champ:',
        'site_placeholder': 'Champ Nord',
        'location_label': '📍 Emplacement du champ (Lat/Lon)',
        'location_manual': 'Coordonnées manuelles',
        'location_auto': 'Utiliser ma position actuelle',
        'lat_label': 'Latitude',
        'lon_label': 'Longitude',
        'get_location_button': '📍 Obtenir ma position',
        'photo_label': 'Photo de l’échantillon de sol',
        'notes_label': 'Observations supplémentaires (couleur, texture, etc.) :',
        'weight_label': 'Superficie du champ (hectares):',
        'execute_button': '🚀 ANALYSER LE SOL',
        'no_photo_error': 'Veuillez d’abord capturer ou télécharger une image.',
        'report_title': 'RAPPORT D’ANALYSE DU SOL',
        'soil_type_label': 'Type de sol:',
        'fertility_label': 'Niveau de fertilité:',
        'recommended_crops': 'Cultures recommandées pour ce sol:',
        'improvement_label': 'Comment améliorer ce sol:',
        'planting_season_label': 'Saison de plantation optimale:',
        'harvest_label': 'Période de récolte prévue:',
        'value_usd_label': 'Valeur estimée de la récolte (USD) : ${value:,.2f}',
        'value_htg_label': 'Valeur estimée de la récolte (HTG) : {value:,.2f}',
        'solution_label': 'Conseil à l’agriculteur:',
        'solution_text': 'Concentrez-vous sur {crops}. {improvement}',
        'strategic_intel': '🌍 Historique des champs',
        'recent_log': 'Analyses de sol récentes:',
        'download_button': '📊 Télécharger l’historique des analyses (CSV)',
        'no_data_info': 'Aucune analyse enregistrée pour le moment. Effectuez une analyse pour générer des données.',
        'access_warning': 'Veuillez entrer votre clé principale dans la barre latérale pour commencer.',
        'language_selector': 'Langue / Language',
        'unknown_soil': 'Type de sol inconnu',
        'map_title': '🗺️ Carte des champs analysés',
        'map_marker_popup': 'Champ: {site}\nSol: {soil}\nCultures: {crops}',
        'translate_report': '🌐 Traduire ce rapport',
        'report_translated': 'Rapport traduit en {lang}'
    },
    'es': {
        'app_title': 'MOTOR IA AGRÍCOLA v1.0',
        'app_subtitle': 'Análisis de suelo y planificación de cultivos',
        'owner_collab': 'Propietario: Gesner Deslandes  | Colaboradores: Gesner Junior Deslandes, Roosevelt Deslandes, Sebastien Stephane Deslandes & Zendaya Christelle Deslandes',
        'made_in_haiti': 'Hecho en 🇭🇹 Haití por GlobalInternet.py',
        'contact_info': '📞 Teléfono del propietario: (509) 4738-5663 | 📧 Correo: deslandes78@gmail.com',
        'sidebar_title': '🛡️ Acceso a la herramienta',
        'sidebar_activation': 'Activación vía MonCash: {moncash}',
        'sidebar_key_label': 'Clave:',
        'sidebar_unlock': 'Desbloquear',
        'sidebar_invalid': 'Clave inválida',
        'sidebar_granted': '✅ ACCESO CONCEDIDO',
        'sidebar_logout': 'Cerrar sesión',
        'welcome_sound_js': """...""",
        'main_header': 'MOTOR IA AGRÍCOLA v1.0',
        'main_subheader': 'Empodere a los agricultores con inteligencia artificial',
        'scan_subheader': '🔍 Análisis del suelo',
        'camera_method_label': 'Cómo capturar la muestra de suelo:',
        'camera_option': '📸 Tomar foto con la cámara (botón de volteo abajo)',
        'upload_option': '📁 Subir foto desde el dispositivo',
        'camera_instruction': '📸 Apunte la cámara a la superficie del suelo. Use el botón Voltear para cambiar entre cámara frontal y trasera.',
        'upload_instruction': '📸 Tome una foto de su suelo y súbala aquí.',
        'reverse_button': '↻ Voltear cámara',
        'capture_button': '📷 Capturar imagen',
        'camera_placeholder': 'La transmisión de la cámara aparecerá aquí después de conceder el permiso.',
        'site_label': 'Nombre del campo:',
        'site_placeholder': 'Campo Norte',
        'location_label': '📍 Ubicación del campo (Lat/Lon)',
        'location_manual': 'Coordenadas manuales',
        'location_auto': 'Usar mi ubicación actual',
        'lat_label': 'Latitud',
        'lon_label': 'Longitud',
        'get_location_button': '📍 Obtener mi ubicación',
        'photo_label': 'Foto de la muestra de suelo',
        'notes_label': 'Observaciones adicionales (color, textura, etc.):',
        'weight_label': 'Área del campo (hectáreas):',
        'execute_button': '🚀 ANALIZAR SUELO',
        'no_photo_error': 'Primero capture o suba una imagen.',
        'report_title': 'INFORME DE ANÁLISIS DE SUELO',
        'soil_type_label': 'Tipo de suelo:',
        'fertility_label': 'Nivel de fertilidad:',
        'recommended_crops': 'Cultivos recomendados para este suelo:',
        'improvement_label': 'Cómo mejorar este suelo:',
        'planting_season_label': 'Temporada de siembra óptima:',
        'harvest_label': 'Tiempo de cosecha esperado:',
        'value_usd_label': 'Valor estimado de la cosecha (USD): ${value:,.2f}',
        'value_htg_label': 'Valor estimado de la cosecha (HTG): {value:,.2f}',
        'solution_label': 'Consejo para el agricultor:',
        'solution_text': 'Concéntrese en {crops}. {improvement}',
        'strategic_intel': '🌍 Historial de campos',
        'recent_log': 'Análisis de suelo recientes:',
        'download_button': '📊 Descargar historial de análisis (CSV)',
        'no_data_info': 'Aún no se han registrado análisis. Realice un análisis para generar datos.',
        'access_warning': 'Por favor ingrese su clave maestra en la barra lateral para comenzar.',
        'language_selector': 'Idioma / Language',
        'unknown_soil': 'Tipo de suelo desconocido',
        'map_title': '🗺️ Mapa de campos analizados',
        'map_marker_popup': 'Campo: {site}\nSuelo: {soil}\nCultivos: {crops}',
        'translate_report': '🌐 Traducir este informe',
        'report_translated': 'Informe traducido al {lang}'
    },
    'ht': {
        'app_title': 'MOTEUR IA AGRYKÒL v1.0',
        'app_subtitle': 'Analiz tè ak planifikasyon rekòt',
        'owner_collab': 'Pwopriyetè: Gesner Deslandes  | Kolaboratè: Gesner Junior Deslandes, Roosevelt Deslandes, Sebastien Stephane Deslandes & Zendaya Christelle Deslandes',
        'made_in_haiti': 'Fèt nan 🇭🇹 Ayiti pa GlobalInternet.py',
        'contact_info': '📞 Telefòn pwopriyetè: (509) 4738-5663 | 📧 Imèl: deslandes78@gmail.com',
        'sidebar_title': '🛡️ Aksè zouti',
        'sidebar_activation': 'Aktivasyon atravè MonCash: {moncash}',
        'sidebar_key_label': 'Kle:',
        'sidebar_unlock': 'Deklannche',
        'sidebar_invalid': 'Kle pa bon',
        'sidebar_granted': '✅ AKSÈ AKÒDE',
        'sidebar_logout': 'Dekonekte',
        'welcome_sound_js': """...""",
        'main_header': 'MOTEUR IA AGRYKÒL v1.0',
        'main_subheader': 'Bay kiltivatè yo pouvwa ak entèlijans atifisyèl',
        'scan_subheader': '🔍 Analiz tè',
        'camera_method_label': 'Ki jan pou pran echantiyon tè a:',
        'camera_option': '📸 Pran foto ak kamera (bouton vire anba a)',
        'upload_option': '📁 Telechaje foto depi aparèy ou',
        'camera_instruction': '📸 Montre kamera ou sou sifas tè a. Sèvi ak bouton Vire pou chanje ant kamera devan ak dèyè.',
        'upload_instruction': '📸 Pran yon foto tè ou epi telechaje li isit la.',
        'reverse_button': '↻ Vire Kamera',
        'capture_button': '📷 Pran Foto',
        'camera_placeholder': 'Flò kamera a ap parèt isit la apre w bay pèmisyon.',
        'site_label': 'Non jaden:',
        'site_placeholder': 'Jaden Nò',
        'location_label': '📍 Kote jaden an (Lat/Lon)',
        'location_manual': 'Kowòdone manyèl',
        'location_auto': 'Sèvi ak pozisyon mwen kounye a',
        'lat_label': 'Latitid',
        'lon_label': 'Longitid',
        'get_location_button': '📍 Jwenn pozisyon mwen',
        'photo_label': 'Foto echantiyon tè',
        'notes_label': 'Lòt obsèvasyon (koulè, teksti, elatriye):',
        'weight_label': 'Sifas jaden an (ektar):',
        'execute_button': '🚀 ANALIZE TÈ',
        'no_photo_error': 'Tanpri pran yon foto oswa telechaje yon imaj an premye.',
        'report_title': 'RAPÒ ANALIZ TÈ',
        'soil_type_label': 'Kalite tè:',
        'fertility_label': 'Nivo fètilite:',
        'recommended_crops': 'Rekòt rekòmande pou tè sa a:',
        'improvement_label': 'Kijan pou amelyore tè sa a:',
        'planting_season_label': 'Sezon plante pi bon:',
        'harvest_label': 'Lè rekòlte espere:',
        'value_usd_label': 'Valè rekòlte estime (USD): ${value:,.2f}',
        'value_htg_label': 'Valè rekòlte estime (HTG): {value:,.2f}',
        'solution_label': 'Konsèy pou kiltivatè a:',
        'solution_text': 'Konsantre ou sou {crops}. {improvement}',
        'strategic_intel': '🌍 Istwa jaden',
        'recent_log': 'Analiz tè resan:',
        'download_button': '📊 Telechaje istorik analiz (CSV)',
        'no_data_info': 'Pa gen okenn analiz anrejistre ankò. Fè yon analiz pou jenere done.',
        'access_warning': 'Tanpri antre kle prensipal ou nan ba a pou kòmanse.',
        'language_selector': 'Lang / Language',
        'unknown_soil': 'Kalite tè enkoni',
        'map_title': '🗺️ Kat jaden yo analize',
        'map_marker_popup': 'Jaden: {site}\nTè: {soil}\nRekòt: {crops}',
        'translate_report': '🌐 Tradwi rapò sa a',
        'report_translated': 'Rapò a tradui an {lang}'
    },
}
def get_text(key, lang=None, **kwargs):
    if lang is None:
        lang = st.session_state.language
    text = TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
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
    def init(self):
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
    <svg width="320" height="192" viewBox="0 0 960 576" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
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
