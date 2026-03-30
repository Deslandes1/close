import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="DSM - Deslandes Stratosphere Monitor", layout="wide", page_icon="🇭🇹")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# --- 2. AUTHENTICATION & HAITIAN FLAG ---
def check_auth():
    if not st.session_state.authenticated:
        # Haitian Flag Presentation
        st.markdown("""
            <div style='text-align: center; background-color: #050505; padding: 40px; border-radius: 15px;'>
                <h1 style='font-size: 70px; margin: 0;'>🇭🇹</h1>
                <div style='height: 15px; background-color: #00209F; width: 100%;'></div>
                <div style='height: 15px; background-color: #D21034; width: 100%;'></div>
                <h1 style='color: white; margin-top: 20px;'>Deslandes Stratosphere Monitor</h1>
                <p style='color: #00FF41; font-family: monospace;'>GLOBALINTERNET.PY SECURE PORTAL</p>
            </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.write("")
            pw = st.text_input("Antre Mòdpas / Enter Access Key:", type="password")
            if st.button("Unlock System"):
                if pw == "20082010":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid Credentials")
        st.stop()

check_auth()

# --- 3. DICTIONARY & TRANSLATION ---
TRANSLATIONS = {
    'en': {
        'title': '🔴 DSM: DESLANDES STRATOSPHERE MONITOR',
        'selector': 'Select Operation Mode',
        'm1': '✈️ Aircraft Radar', 'm2': '🛰️ Satellite Tracker', 'm3': '🚀 Missile Detector',
        'threat': '⚠️ OBJECT DETECTED', 'demo': '📡 CACHE/DEMO ACTIVE',
        'report': '📥 Download Intel Report', 'owner': 'GlobaLInternet.py - Made in Haiti'
    },
    'fr': {
        'title': '🔴 DSM: MONITORING STRATOSPHÉRIQUE',
        'selector': 'Sélectionner le mode',
        'm1': '✈️ Radar Aéronefs', 'm2': '🛰️ Traqueur Satellites', 'm3': '🚀 Détecteur de Missiles',
        'threat': '⚠️ OBJET DÉTECTÉ', 'demo': '📡 MODE DÉMO/CACHE',
        'report': '📥 Télécharger le rapport', 'owner': 'GlobaLInternet.py - Fait en Haïti'
    },
    'ht': {
        'title': '🔴 DSM: RADAR SIVEYANS GLOBAL',
        'selector': 'Chwazi Mòd Operasyon',
        'm1': '✈️ Radar Avyon', 'm2': '🛰️ Swiv Satelit', 'm3': '🚀 Detektè Misil',
        'threat': '⚠️ MENAS DETEKTE', 'demo': '📡 MÒD DEMO AKTIF',
        'report': '📥 Telechaje Rapò a', 'owner': 'GlobaLInternet.py - Fèt an Ayiti'
    }
}

def t(key):
    return TRANSLATIONS[st.session_state.language].get(key, key)

# --- 4. DATA GENERATOR ---
@st.cache_data(ttl=300)
def get_data(mode):
    if "Aircraft" in mode:
        return [{"id": "CIV-44", "type": "Boeing 737", "r": 800, "th": 110, "s": 820}]
    elif "Satellite" in mode:
        return [{"id": "GPS-BII", "type": "Nav Sat", "r": 2500, "th": 45, "s": 14000}]
    else: # Missile
        return [{"id": "TGT-ALPHA", "type": "Hypersonic", "r": 1200, "th": 15, "s": 7500}]

# --- 5. UI LAYOUT ---
st.sidebar.title("DSM Control")
st.sidebar.info("Owner: Gesner Deslandes\n(509)-4738-5663")
lang = st.sidebar.selectbox("Language", ["English", "Français", "Kreyòl"])
st.session_state.language = {'English': 'en', 'Français': 'fr', 'Kreyòl': 'ht'}[lang]

st.title(t('title'))

# FIXED: Added a non-empty label to avoid the log error
mode_choice = st.radio(t('selector'), [t('m1'), t('m2'), t('m3')], horizontal=True)

# Map internal mode
if mode_choice == t('m1'): active = "Aircraft"
elif mode_choice == t('m2'): active = "Satellite"
else: active = "Missile"

# --- 6. RADAR & LOGS ---
col_r, col_l = st.columns([2, 1])
objs = get_data(active)

with col_r:
    st.subheader(f"{active} Scan")
    fig = go.Figure()
    # Sweep Animation
    angle = (time.time() * 80) % 360
    for offset in [0, 180]:
        fig.add_trace(go.Scatterpolar(r=[0, 3000], theta=[(angle+offset)%360]*2, mode='lines', line=dict(color='lime'), opacity=0.4, showlegend=False))
    # Points
    fig.add_trace(go.Scatterpolar(r=[o['r'] for o in objs], theta=[o['th'] for o in objs], mode='markers+text', marker=dict(color='red', size=12), text=[o['id'] for o in objs]))
    fig.update_layout(polar=dict(bgcolor="black", radialaxis=dict(color="lime"), angularaxis=dict(color="lime")), paper_bgcolor="black", font_color="lime", height=600)
    st.plotly_chart(fig, use_container_width=True)

with col_l:
    st.error(t('threat'))
    for o in objs:
        st.metric(label=f"Target: {o['id']}", value=f"{o['s']} km/h", delta=o['type'])
    st.download_button(t('report'), f"DSM LOG\n{active}\n{objs}", file_name="DSM_Report.txt")

st.caption(f"--- \n {t('owner')}")
time.sleep(2)
st.rerun()
