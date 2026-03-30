import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2, pi, asin, degrees
import time
from datetime import datetime

# --- 1. SYSTEM SETTINGS ---
st.set_page_config(page_title="DSM - Deslandes Stratosphere Monitor", layout="wide", page_icon="🇭🇹")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# --- 2. AUTHENTICATION & HAITIAN PRESENTATION ---
def check_auth():
    if not st.session_state.authenticated:
        # Haitian Flag Presentation Page
        st.markdown("""
            <div style='text-align: center; background-color: #050505; padding: 50px; border-radius: 15px;'>
                <h1 style='font-size: 80px; margin: 0;'>🇭🇹</h1>
                <div style='height: 15px; background-color: #00209F; width: 100%;'></div>
                <div style='height: 15px; background-color: #D21034; width: 100%;'></div>
                <h1 style='color: white; margin-top: 20px;'>Deslandes Stratosphere Monitor</h1>
                <p style='color: #00FF41; font-family: monospace;'>PROPRIETARY SOFTWARE BY GLOBALINTERNET.PY</p>
            </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns([1, 2, 1])
        with cols[1]:
            pw = st.text_input("Antre Mòdpas / Enter Access Key:", type="password")
            if st.button("Unlock System"):
                if pw == "20082010":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Credentials")
        st.stop()

check_auth()

# --- 3. BRANDING & TRANSLATIONS ---
BRAND_INFO = """
**GlobaLInternet.py** Owner: Gesner Deslandes  
Phone: (509)-4738-5663  
Email: deslandes78@gmail.com  
**MADE IN HAITI** 🇭🇹
"""

TRANSLATIONS = {
    'en': {
        'title': '🔴 DSM: DESLANDES STRATOSPHERE MONITOR',
        'm1': '✈️ Aircraft Radar', 'm2': '🛰️ Satellite Tracker', 'm3': '🚀 Missile Detector',
        'threat': '⚠️ OBJECT DETECTED', 'demo': '📡 DEMO/CACHE MODE',
        'report': '📥 Download Intel Report', 'owner': 'Licensed to GlobaLInternet.py'
    },
    'fr': {
        'title': '🔴 DSM: MONITORING STRATOSPHÉRIQUE',
        'm1': '✈️ Radar Aéronefs', 'm2': '🛰️ Traqueur Satellites', 'm3': '🚀 Détecteur de Missiles',
        'threat': '⚠️ OBJET DÉTECTÉ', 'demo': '📡 MODE DÉMO/CACHE',
        'report': '📥 Télécharger le rapport', 'owner': 'Sous licence GlobaLInternet.py'
    },
    'ht': {
        'title': '🔴 DSM: RADAR SIVEYANS GLOBAL',
        'm1': '✈️ Radar Avyon', 'm2': '🛰️ Swiv Satelit', 'm3': '🚀 Detektè Misil',
        'threat': '⚠️ MENAS DETEKTE', 'demo': '📡 MÒD DEMO',
        'report': '📥 Telechaje Rapò a', 'owner': 'Lisansye pa GlobaLInternet.py'
    }
}

def t(key):
    return TRANSLATIONS[st.session_state.language].get(key, key)

# --- 4. INTEGRATED DATA ENGINE (AIRCRAFT, SATELLITE, MISSILE) ---
@st.cache_data(ttl=300)
def get_integrated_data(mode_key):
    # Simulated detections for demo mode if APIs are unavailable
    if "Aircraft" in mode_key:
        return [
            {"id": "FL-702", "type": "Airbus A320", "dist": 450, "angle": 120, "speed": 840},
            {"id": "UAV-9", "type": "Drone (High Alt)", "dist": 150, "angle": 310, "speed": 120}
        ]
    elif "Satellite" in mode_key:
        return [
            {"id": "ISS", "type": "Space Station", "dist": 2800, "angle": 45, "speed": 27600},
            {"id": "STARLINK-K", "type": "Comm Sat", "dist": 2100, "angle": 220, "speed": 27000}
        ]
    else: # Missile Mode
        return [
            {"id": "THREAT-1", "type": "Hypersonic", "dist": 1100, "angle": 15, "speed": 6800},
            {"id": "THREAT-2", "type": "ICBM", "dist": 2400, "angle": 195, "speed": 9400}
        ]

# --- 5. MAIN UI LAYOUT ---
st.sidebar.title("DSM Systems")
st.sidebar.markdown(BRAND_INFO)
lang_select = st.sidebar.selectbox("Language", ["English", "Français", "Kreyòl"])
st.session_state.language = {'English': 'en', 'Français': 'fr', 'Kreyòl': 'ht'}[lang_select]

if st.sidebar.button("🔒 Lock DSM"):
    st.session_state.authenticated = False
    st.rerun()

st.title(t('title'))
mode_selection = st.radio("", [t('m1'), t('m2'), t('m3')], horizontal=True, index=2)

# Determine internal mode
if mode_selection == t('m1'): current_mode = "Aircraft"
elif mode_selection == t('m2'): current_mode = "Satellite"
else: current_mode = "Missile"

# --- 6. RADAR VISUALIZATION & INTEL ---
col_radar, col_intel = st.columns([2, 1])
active_data = get_integrated_data(current_mode)
MAX_SCAN = 3000

with col_radar:
    st.subheader(f"📡 {current_mode} Radar ({t('demo')})")
    fig = go.Figure()

    # Dynamic Dual-Needle Sweep
    rotation_speed = 120 if current_mode == "Missile" else 60
    angle = (time.time() * rotation_speed) % 360
    for offset in [0, 180]:
        fig.add_trace(go.Scatterpolar(
            r=[0, MAX_SCAN], theta=[(angle+offset)%360]*2,
            mode='lines', line=dict(color='#00FF41', width=4), opacity=0.6, showlegend=False
        ))

    # Plot Detections
    fig.add_trace(go.Scatterpolar(
        r=[d['dist'] for d in active_data], theta=[d['angle'] for d in active_data],
        mode='markers+text', marker=dict(size=16, color='red', symbol='x'),
        text=[d['id'] for d in active_data], textposition="top center"
    ))

    fig.update_layout(
        polar=dict(bgcolor="black", 
                   radialaxis=dict(gridcolor="#004400", color="lime", range=[0, MAX_SCAN]),
                   angularaxis=dict(gridcolor="#004400", color="lime")),
        paper_bgcolor="black", font_color="lime", height=750
    )
    st.plotly_chart(fig, use_container_width=True)

with col_intel:
    st.error(t('threat'))
    log_content = f"DSM INTELLIGENCE LOG - {current_mode}\nGlobaLInternet.py - MADE IN HAITI\n"
    log_content += f"Timestamp: {datetime.now()}\n" + "="*30 + "\n"
    
    for d in active_data:
        with st.container(border=True):
            st.markdown(f"### 🎯 {d['id']}")
            st.write(f"**Type:** {d['type']}")
            st.write(f"**Speed:** {d['speed']} km/h")
            log_content += f"ID: {d['id']} | Type: {d['type']} | Velocity: {d['speed']}\n"
    
    st.divider()
    st.download_button(t('report'), log_content, file_name=f"DSM_{current_mode}_Log.txt")

st.caption(f"--- \n {t('owner')}")

# Radar Update loop
time.sleep(1)
st.rerun()
