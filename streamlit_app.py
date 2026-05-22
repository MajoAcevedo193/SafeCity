# ============================================================
# SafeCity - Dashboard v6
# Cambios:
#   - Abre por defecto en ZMG con fecha/hora/franja del sistema
#   - 35 ajustadores por defecto
#   - Sin estadísticas clave en sidebar
#   - Sin tabla Top 10 colonias
#   - Sin badges en el header
#   - Diseño responsivo (laptop, tablet, móvil)
# Correr: python -m streamlit run 04_dashboard_safecity_v6.py
# ============================================================

import importlib
import streamlit as st
import pandas as pd
import numpy as np
import json
import xgboost as xgb
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Evitar el almacenamiento en caché de módulos importados para permitir la recarga en caliente instantánea
import colonias_heatmap
import colonias_temporal
importlib.reload(colonias_heatmap)
importlib.reload(colonias_temporal)

from colonias_heatmap import (
    entrenar_modelo_intensidad,
    fig_mapa_colonias,
    fig_mapa_municipios,
    predecir_por_colonia,
)
from colonias_temporal import repartir_enteros

st.set_page_config(page_title="SafeCity", page_icon="https://img.icons8.com/fluency/48/shield.png",
                   layout="wide", initial_sidebar_state="expanded")

# ── RUTA AUTOMÁTICA ──────────────────────────────────────────
def _ruta_proyecto() -> str:
    return str(Path(__file__).resolve().parent)

BASE = _ruta_proyecto()


def render_maplibre_map(target_lat, target_lon, target_zoom, active_mun, points_df, estilo_mapa, tipo_mapa, dia_sel, mes_sel, franja_dash):
    import json
    import hashlib
    
    # 1. Determinar URL del estilo
    style_mapping = {
        "Mapa Claro (Carto)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "Mapa Contraste (Oscuro)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "Mapa Detallado (OSM)": "OSM"
    }
    style_url = style_mapping.get(estilo_mapa, "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json")
    
    # 2. Formatear puntos de colonias a JSON
    points_list = []
    if points_df is not None and len(points_df) > 0:
        cols = ['colonia', 'municipio', 'lat', 'lon', 'prob_col', 'acc_col', 'nivel', 'ajust_col']
        df_temp = points_df.copy()
        for col in cols:
            if col not in df_temp.columns:
                df_temp[col] = 0.0 if col in ['lat', 'lon', 'prob_col'] else (0 if col in ['acc_col', 'ajust_col'] else '')
        df_temp = df_temp[cols].fillna({
            'colonia': 'Desconocida', 'municipio': 'Desconocido',
            'lat': 20.672, 'lon': -103.345, 'prob_col': 0.0,
            'acc_col': 0, 'nivel': 'BAJO', 'ajust_col': 0
        })
        points_list = df_temp.to_dict(orient="records")
        
    points_json = json.dumps(points_list)
    polygons_json = json.dumps(POLIGONOS)
    
    show_heatmap = "true" if tipo_mapa in ["Calor de Densidad", "Vista Híbrida"] else "false"
    show_points = "true" if tipo_mapa in ["Puntos de Riesgo", "Vista Híbrida"] else "false"
    
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>SafeCity Map</title>
    <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />
    <link href="https://cdn.jsdelivr.net/npm/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
    <script src="https://cdn.jsdelivr.net/npm/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <style>
        body { margin: 0; padding: 0; background: transparent; overflow: hidden; }
        #map { position: absolute; top: 0; bottom: 0; width: 100%; height: 100%; border-radius: 8px; }
        
        /* Tooltip Premium CSS */
        .mapboxgl-popup {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            z-index: 10000;
        }
        .mapboxgl-popup-content {
            background: rgba(255, 255, 255, 0.96) !important;
            backdrop-filter: blur(8px);
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05) !important;
            padding: 10px 12px !important;
            color: #1e293b !important;
            max-width: 240px;
        }
        .mapboxgl-popup-anchor-top .mapboxgl-popup-tip { border-bottom-color: rgba(255, 255, 255, 0.96) !important; }
        .mapboxgl-popup-anchor-bottom .mapboxgl-popup-tip { border-top-color: rgba(255, 255, 255, 0.96) !important; }
        .mapboxgl-popup-anchor-left .mapboxgl-popup-tip { border-right-color: rgba(255, 255, 255, 0.96) !important; }
        .mapboxgl-popup-anchor-right .mapboxgl-popup-tip { border-left-color: rgba(255, 255, 255, 0.96) !important; }
        
        .tooltip-title { font-weight: 700; font-size: 12px; margin-bottom: 4px; color: #0f172a; }
        .tooltip-row { display: flex; justify-content: space-between; font-size: 11px; margin: 2px 0; color: #475569; }
        .tooltip-value { font-weight: 600; color: #0f172a; }
        
        /* Leyenda */
        .map-legend {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(4px);
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px 10px;
            font-family: 'Inter', sans-serif;
            font-size: 10px;
            color: #334155;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            pointer-events: none;
            z-index: 1000;
        }
        .legend-item { display: flex; align-items: center; margin: 3px 0; }
        .legend-color { width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="map-legend">
        <div style="font-weight:700;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.05em;">Riesgo Vial</div>
        <div class="legend-item"><div class="legend-color" style="background:#ef4444;"></div>Alto Riesgo</div>
        <div class="legend-item"><div class="legend-color" style="background:#f59e0b;"></div>Medio Riesgo</div>
        <div class="legend-item"><div class="legend-color" style="background:#10b981;"></div>Bajo Riesgo</div>
    </div>
    
    <script>
        const targetLat = __TARGET_LAT__;
        const targetLon = __TARGET_LON__;
        const targetZoom = __TARGET_ZOOM__;
        const activeMun = "__ACTIVE_MUN__";
        const polygons = __POLYGONS_JSON__;
        const points = __POINTS_JSON__;
        const styleUrl = "__STYLE_URL__";
        const showHeatmap = __SHOW_HEATMAP__;
        const showPoints = __SHOW_POINTS__;

        // Read last map center/zoom and last municipality from localStorage
        let startLat = localStorage.getItem("safecity_lat");
        let startLon = localStorage.getItem("safecity_lon");
        let startZoom = localStorage.getItem("safecity_zoom");
        let lastMun = localStorage.getItem("safecity_last_mun");

        // Determine if selected municipality actually changed
        let munChanged = (lastMun !== activeMun);

        // Use current coordinates if no cache exists
        if (!startLat || !startLon || !startZoom) {
            startLat = targetLat;
            startLon = targetLon;
            startZoom = targetZoom;
            munChanged = false; // Do not fly on initial clean load
        } else {
            startLat = parseFloat(startLat);
            startLon = parseFloat(startLon);
            startZoom = parseFloat(startZoom);
        }

        // Save activeMun as lastMun immediately to block subsequent flyTos on simple reruns
        localStorage.setItem("safecity_last_mun", activeMun);

        let styleObj;
        if (styleUrl === "OSM") {
            styleObj = {
                "version": 8,
                "sources": {
                    "osm": {
                        "type": "raster",
                        "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                        "tileSize": 256,
                        "attribution": "© OpenStreetMap contributors"
                    }
                },
                "layers": [
                    {
                        "id": "osm-layer",
                        "type": "raster",
                        "source": "osm",
                        "minzoom": 0,
                        "maxzoom": 19
                    }
                ]
            };
        } else {
            styleObj = styleUrl;
        }

        const map = new maplibregl.Map({
            container: 'map',
            style: styleObj,
            center: [startLon, startLat],
            zoom: startZoom,
            attributionControl: false
        });

        // Add navigation controls
        map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');

        map.on('load', () => {
            // Draw Municipios boundaries
            Object.keys(polygons).forEach(munName => {
                const isSelected = (munName === activeMun);
                const isZmg = (activeMun === "ZMG — Todas las zonas");
                
                // Color based on selection or risk
                let fillColor = "rgba(16, 185, 129, 0.06)";
                let borderColor = "rgba(16, 185, 129, 0.5)";
                let borderWidth = 0.8;
                
                if (munName === "Guadalajara") { fillColor = "rgba(239, 68, 68, 0.06)"; borderColor = "rgba(239, 68, 68, 0.4)"; }
                else if (munName === "Zapopan") { fillColor = "rgba(245, 158, 11, 0.06)"; borderColor = "rgba(245, 158, 11, 0.4)"; }
                else if (munName === "Tlaquepaque") { fillColor = "rgba(16, 185, 129, 0.06)"; borderColor = "rgba(16, 185, 129, 0.4)"; }
                else if (munName === "Tonalá") { fillColor = "rgba(16, 185, 129, 0.06)"; borderColor = "rgba(16, 185, 129, 0.4)"; }

                if (!isZmg && isSelected) {
                    fillColor = "rgba(37, 99, 235, 0.16)";
                    borderColor = "#2563eb";
                    borderWidth = 2.5;
                }

                // Add source for the polygon
                map.addSource(`source-${munName}`, {
                    'type': 'geojson',
                    'data': {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [polygons[munName]]
                        }
                    }
                });

                // Add fill layer
                map.addLayer({
                    'id': `fill-${munName}`,
                    'type': 'fill',
                    'source': `source-${munName}`,
                    'layout': {},
                    'paint': {
                        'fill-color': fillColor,
                        'fill-opacity': 1.0
                    }
                });

                // Add outer glow layers if selected
                if (!isZmg && isSelected) {
                    map.addLayer({
                        'id': `glow-outer-${munName}`,
                        'type': 'line',
                        'source': `source-${munName}`,
                        'layout': {},
                        'paint': {
                            'line-color': 'rgba(37, 99, 235, 0.25)',
                            'line-width': 12
                        }
                    });
                    map.addLayer({
                        'id': `glow-mid-${munName}`,
                        'type': 'line',
                        'source': `source-${munName}`,
                        'layout': {},
                        'paint': {
                            'line-color': 'rgba(37, 99, 235, 0.45)',
                            'line-width': 7
                        }
                    });
                }

                // Add border layer
                map.addLayer({
                    'id': `border-${munName}`,
                    'type': 'line',
                    'source': `source-${munName}`,
                    'layout': {},
                    'paint': {
                        'line-color': borderColor,
                        'line-width': borderWidth
                    }
                });
            });

            // Convert points to GeoJSON
            const geojsonPoints = {
                'type': 'FeatureCollection',
                'features': points.map(p => ({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [parseFloat(p.lon), parseFloat(p.lat)]
                    },
                    'properties': {
                        'colonia': p.colonia,
                        'municipio': p.municipio,
                        'prob': p.prob_col * 100,
                        'acc': p.acc_col,
                        'nivel': p.nivel,
                        'ajust': p.ajust_col,
                        'color': p.nivel === 'ALTO' ? '#ef4444' : (p.nivel === 'MEDIO' ? '#f59e0b' : '#10b981'),
                        'size': p.nivel === 'ALTO' ? 10 + p.prob_col * 14 : (p.nivel === 'MEDIO' ? 8 + p.prob_col * 10 : 6 + p.prob_col * 6),
                        'opacity': p.nivel === 'ALTO' ? 0.85 : (p.nivel === 'MEDIO' ? 0.70 : 0.50)
                    }
                }))
            };

            map.addSource('source-points', {
                'type': 'geojson',
                'data': geojsonPoints
            });

            // Add Heatmap Layer if requested
            if (showHeatmap) {
                map.addLayer({
                    'id': 'layer-heatmap',
                    'type': 'heatmap',
                    'source': 'source-points',
                    'maxzoom': 15,
                    'paint': {
                        'heatmap-weight': {
                            'property': 'prob',
                            'type': 'exponential',
                            'stops': [[0, 0], [100, 1]]
                        },
                        'heatmap-intensity': 1.5,
                        'heatmap-color': [
                            'interpolate',
                            ['linear'],
                            ['heatmap-value'],
                            0, 'rgba(16,185,129,0)',
                            0.2, 'rgba(16,185,129,0.35)',
                            0.5, 'rgba(245,158,11,0.65)',
                            0.8, 'rgba(239,68,68,0.85)',
                            1.0, 'rgba(220,38,38,0.98)'
                        ],
                        'heatmap-radius': 22,
                        'heatmap-opacity': 0.8
                    }
                });
            }

            // Add Circle Points Layer if requested
            if (showPoints) {
                map.addLayer({
                    'id': 'layer-points',
                    'type': 'circle',
                    'source': 'source-points',
                    'paint': {
                        'circle-radius': ['get', 'size'],
                        'circle-color': ['get', 'color'],
                        'circle-opacity': ['get', 'opacity'],
                        'circle-stroke-width': 1.2,
                        'circle-stroke-color': '#ffffff'
                    }
                });

                // Create hover popup
                const popup = new maplibregl.Popup({
                    closeButton: false,
                    closeOnClick: false
                });

                map.on('mouseenter', 'layer-points', (e) => {
                    map.getCanvas().style.cursor = 'pointer';
                    const coordinates = e.features[0].geometry.coordinates.slice();
                    const props = e.features[0].properties;

                    const html = `
                        <div class="tooltip-title">${props.colonia}</div>
                        <div class="tooltip-row"><span>Municipio:</span><span class="tooltip-value">${props.municipio}</span></div>
                        <div class="tooltip-row"><span>Nivel Riesgo:</span><span class="tooltip-value" style="color:${props.color};font-weight:700;">${props.nivel}</span></div>
                        <div class="tooltip-row"><span>Prob. Siniestro:</span><span class="tooltip-value">${parseFloat(props.prob).toFixed(1)}%</span></div>
                        <div class="tooltip-row"><span>Siniestros Est:</span><span class="tooltip-value">${props.acc}</span></div>
                        <div class="tooltip-row"><span>Ajustadores:</span><span class="tooltip-value" style="color:#2563eb;font-weight:700;">${props.ajust}</span></div>
                    `;

                    popup.setLngLat(coordinates).setHTML(html).addTo(map);
                });

                map.on('mouseleave', 'layer-points', () => {
                    map.getCanvas().style.cursor = '';
                    popup.remove();
                });
            }

            // TRIGGER THE SMOOTH flyTo CAMERA TRANSITION ONLY IF SELECTED MUNICIPIO CHANGED!
            if (munChanged) {
                setTimeout(() => {
                    map.flyTo({
                        center: [targetLon, targetLat],
                        zoom: targetZoom,
                        speed: 1.2,
                        curve: 1.42,
                        duration: 1800,
                        essential: true
                    });
                }, 100);
            }
        });

        // Keep localStorage updated with final move positions
        map.on('moveend', () => {
            const center = map.getCenter();
            localStorage.setItem("safecity_lat", center.lat);
            localStorage.setItem("safecity_lon", center.lng);
            localStorage.setItem("safecity_zoom", map.getZoom());
        });
    </script>
</body>
</html>"""
    
    html = html_template \
        .replace("__TARGET_LAT__", str(target_lat)) \
        .replace("__TARGET_LON__", str(target_lon)) \
        .replace("__TARGET_ZOOM__", str(target_zoom)) \
        .replace("__ACTIVE_MUN__", active_mun) \
        .replace("__POLYGONS_JSON__", polygons_json) \
        .replace("__POINTS_JSON__", points_json) \
        .replace("__STYLE_URL__", style_url) \
        .replace("__SHOW_HEATMAP__", show_heatmap) \
        .replace("__SHOW_POINTS__", show_points)
        
    # 3. Generar una clave única para forzar a Streamlit/React a recrear el iframe.
    # El key cambia si cambia el municipio, estilo, tipo de mapa, día, mes o franja.
    # Esto asegura que el iframe se repinte instantáneamente al cambiar controles sin freezeos.
    # Como st.components.v1.html no acepta el argumento 'key' en esta versión de Streamlit,
    # envolvemos el iframe dentro de un st.container que sí acepta 'key'. Esto fuerza
    # a React a desmontar y montar el iframe al cambiar la clave.
    points_hash = hashlib.md5(points_json.encode('utf-8')).hexdigest()[:8] if points_json else "empty"
    map_key = f"maplibre_{active_mun}_{estilo_mapa}_{tipo_mapa}_{dia_sel}_{mes_sel}_{franja_dash}_{points_hash}_{target_lat}_{target_lon}_{target_zoom}"
    
    with st.container(key=map_key):
        st.components.v1.html(html, height=310)


def punto_en_poligono(lat: float, lon: float, poly: list) -> bool:
    """Ray-casting algorithm to determine if a point (lat, lon) is inside a polygon."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    p1x, p1y = poly[0]  # lon, lat
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if lat > min(p1y, p2y):
            if lat <= max(p1y, p2y):
                if lon <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lon <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# ── FECHA Y FRANJA ACTUAL ────────────────────────────────────
def franja_actual(hora: int) -> int:
    """Devuelve el índice de franja de 8 (0-7) según la hora actual."""
    if   hora <= 2:  return 0
    elif hora <= 5:  return 1
    elif hora <= 8:  return 2
    elif hora <= 11: return 3
    elif hora <= 14: return 4
    elif hora <= 17: return 5
    elif hora <= 20: return 6
    else:            return 7

def hora_a_franja_modelo(h: int) -> int:
    if   h < 6:  return 0
    elif h < 12: return 1
    elif h < 18: return 2
    else:        return 3

_ahora       = datetime.now()
_dia_idx     = _ahora.weekday()          # 0=Lunes … 6=Domingo
_mes_actual  = _ahora.month
_franja_idx  = franja_actual(_ahora.hour)

# ── ESTILOS RESPONSIVOS ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp, p, label, li, h1, h2, h3, h4, h5, h6, table, th, td, input, select, button {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background-color: #f8fafc !important; color: #1e293b !important; }

/* Hacer transparente el header de Streamlit y ocultar botones de Deploy */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    backdrop-filter: none !important;
    z-index: 99 !important;
}
header[data-testid="stHeader"] [data-testid="stHeaderActionElements"] {
    display: none !important;
}

/* Aumentar padding superior para que no choque con la barra de navegación del navegador */
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 1.0rem !important;
    padding-right: 1.0rem !important;
}
[data-testid="stVerticalBlock"] > div {
    padding-top: 0.05rem !important;
    padding-bottom: 0.05rem !important;
}

/* ── Contenedor Físico Premium (Dashboard Card) ── */
.dashboard-card, div[data-testid="stPlotlyChart"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.01) !important;
    margin-bottom: 8px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.dashboard-card:hover, div[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 10px 20px -3px rgba(0,0,0,0.04), 0 4px 6px -2px rgba(0,0,0,0.01) !important;
    transform: translateY(-1px);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] label {
    color: #475569 !important; font-size: 11px !important;
    font-weight: 700 !important; text-transform: uppercase;
    letter-spacing: .06em !important;
}
[data-testid="stSidebar"] p { color: #475569 !important; }
[data-testid="stSidebar"] h3 { 
    color: #0f172a !important; 
    font-size: 18px !important; 
    font-weight: 800 !important;
    letter-spacing: -.5px;
    margin-bottom: 15px;
}

/* Selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #1e293b !important; font-weight: 600 !important;
}
[data-baseweb="popover"], [data-baseweb="popover"] * {
    background: #ffffff !important; color: #1e293b !important;
}
[data-baseweb="popover"] [aria-selected="true"],
[data-baseweb="popover"] [data-baseweb="option"]:hover {
    background: #e2e8f0 !important; color: #0f172a !important;
}

/* Number input */
[data-testid="stSidebar"] input[type="number"] {
    background: #ffffff !important; color: #1e293b !important;
    font-weight: 700 !important; border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important; font-size: 13px !important;
}

/* Botón Premium */
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(90deg, #1e3a5f, #2563eb) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important; border-radius: 6px !important;
    font-size: 13px !important; padding: 8px !important;
    box-shadow: 0 4px 12px rgba(37,99,235,.15) !important;
    width: 100% !important; letter-spacing: .03em !important;
    transition: all 0.2s ease-in-out;
}
[data-testid="stSidebar"] .stButton button p,
[data-testid="stSidebar"] .stButton button span,
[data-testid="stSidebar"] .stButton button div {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    box-shadow: 0 6px 16px rgba(37,99,235,.30) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    transform: translateY(-1px);
}

/* Botón Secundario (Limpiar Caché) */
[data-testid="stSidebar"] .stButton.cache-btn button {
    background: linear-gradient(90deg, #1e3a5f, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(37,99,235,.15) !important;
    margin-top: 6px;
    font-size: 11px !important;
    padding: 8px !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stButton.cache-btn button:hover {
    box-shadow: 0 6px 16px rgba(37,99,235,.30) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}

/* Panel de Control título — forzar blanco */
.panel-control-header p {
    color: #ffffff !important;
}
/* ── Header ── */
.header-bar {
    background: linear-gradient(90deg, #1e3a8a 0%, #2563eb 100%) !important;
    padding: 12px 20px !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.12) !important;
    border: none !important;
}
.header-title-text {
    font-size: 17px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    margin: 0 !important;
    letter-spacing: -.4px !important;
}
.header-sub-text {
    font-size: 11px !important;
    color: #93c5fd !important;
    margin: 1px 0 0 !important;
    font-weight: 500 !important;
}

/* ── Sección ── */
.section-title {
    font-size: 11.5px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 0 0 8px 0 !important;
    padding-bottom: 4px !important;
    border-bottom: 1.5px solid #f1f5f9 !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
}

/* ── KPI Cards Glassmorphism Premium ── */
.kpi-card {
    border-radius: 10px !important;
    padding: 12px 16px !important;
    text-align: center !important;
    margin-bottom: 6px !important;
    border: 1px solid rgba(226, 232, 240, 0.8) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.04) !important;
}
.kpi-title {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
    margin: 0 0 4px !important;
}
.kpi-value {
    font-size: 21px !important;
    font-weight: 850 !important;
    margin: 0 !important;
    line-height: 1.1 !important;
    letter-spacing: -.5px !important;
}
.kpi-desc {
    font-size: 10.5px !important;
    margin: 4px 0 0 !important;
    font-weight: 500 !important;
    opacity: 0.9 !important;
}

/* ── Badges Dinámicos con Dots ── */
.zmg-badge {
    font-size: 8px !important;
    font-weight: 700 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
}
.badge-alto  { background: rgba(239, 68, 68, 0.10) !important; color: #ef4444 !important; border: 1px solid rgba(239, 68, 68, 0.20) !important; }
.badge-medio { background: rgba(245, 158, 11, 0.10) !important; color: #f59e0b !important; border: 1px solid rgba(245, 158, 11, 0.20) !important; }
.badge-bajo  { background: rgba(16, 185, 129, 0.10) !important; color: #10b981 !important; border: 1px solid rgba(16, 185, 129, 0.20) !important; }

.risk-dot {
    width: 5px !important;
    height: 5px !important;
    border-radius: 50% !important;
    display: inline-block !important;
}
.dot-alto { background-color: #ef4444 !important; }
.dot-medio { background-color: #f59e0b !important; }
.dot-bajo { background-color: #10b981 !important; }

/* ── Sleek Light Tables ── */
.compact-table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    font-size: 11.5px !important;
    background: #ffffff !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    margin-bottom: 6px !important;
}
.compact-table th {
    background: #f8fafc !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    text-align: left !important;
    padding: 8px 10px !important;
    border-bottom: 1.5px solid #e2e8f0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
.compact-table td {
    padding: 6px 10px !important;
    border-bottom: 1px solid #f1f5f9 !important;
    color: #334155 !important;
    vertical-align: middle !important;
    transition: background-color 0.15s ease !important;
}
.compact-table tr:last-child td {
    border-bottom: none !important;
}
.compact-table tr:hover td {
    background-color: #f8fafc !important;
}

@media (max-width: 768px) {
    .header-title-text { font-size: 14px !important; }
    .kpi-value { font-size: 17px !important; }
}
</style>
""", unsafe_allow_html=True)

# ── CARGA MODELOS ────────────────────────────────────────────
@st.cache_resource(ttl=1)
def cargar_modelos():
    reg = xgb.XGBRegressor()
    clf = xgb.XGBClassifier()
    reg.load_model(str(Path(BASE) / "safecity_model_reg.json"))
    clf.load_model(str(Path(BASE) / "safecity_model_clf.json"))
    with open(Path(BASE) / "safecity_params.json", encoding="utf-8") as f:
        return reg, clf, json.load(f)

@st.cache_data(ttl=10)
def cargar_colonias():
    # Generado con actualizar_colonias_siniestralidad.py (siniestralimap IMSS)
    p = Path(BASE) / "colonias_top100.csv"
    if p.is_file():
        try:
            df = pd.read_csv(p, encoding="utf-8")
        except Exception:
            df = pd.read_csv(p, encoding="latin-1", errors="replace")
        if "colonia" in df.columns:
            df["colonia"] = df["colonia"].astype(str)
        if "municipio" in df.columns:
            df["municipio"] = df["municipio"].astype(str)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=10)
def cargar_shares_colonias():
    p = Path(BASE) / "colonias_shares_detalle.csv"
    pm = Path(BASE) / "colonias_shares_mun.csv"
    
    if p.is_file():
        try:
            det = pd.read_csv(p, encoding="utf-8")
        except Exception:
            det = pd.read_csv(p, encoding="latin-1", errors="replace")
    else:
        det = pd.DataFrame()
        
    if pm.is_file():
        try:
            mun = pd.read_csv(pm, encoding="utf-8")
        except Exception:
            mun = pd.read_csv(pm, encoding="latin-1", errors="replace")
    else:
        mun = pd.DataFrame()
        
    if len(det) and "colonia" in det.columns:
        det["colonia"] = det["colonia"].astype(str)
    if len(det) and "municipio" in det.columns:
        det["municipio"] = det["municipio"].astype(str)
    if len(mun) and "municipio" in mun.columns:
        mun["municipio"] = mun["municipio"].astype(str)
        
    return det, mun

try:
    modelo_reg, modelo_clf, params = cargar_modelos()
    df_colonias = cargar_colonias()
    df_shares_det, df_shares_mun = cargar_shares_colonias()
    FEATURE_COLS  = params["feature_cols"]
    CLF_THRESHOLD = params.get("clf_threshold", 0.47)
    MUN_ENC_MAP   = params.get("mun_enc_map", {})
    FACTOR        = params.get("factor_dashboard", 1.0)
    modelo_col    = entrenar_modelo_intensidad(df_colonias, MUN_ENC_MAP) \
                    if len(df_colonias) else None
except Exception as e:
    st.error(f"Error cargando modelos: {e}"); st.stop()

# Forzar umbral correcto por si el cache devuelve valor viejo
if CLF_THRESHOLD < 0.40:
    CLF_THRESHOLD = 0.47
MEDIO_THRESHOLD = 0.35

# ── CONSTANTES ───────────────────────────────────────────────
MUNICIPIOS_BASE = ["Guadalajara","Zapopan","Tlaquepaque","Tonalá"]
OPCIONES_AREA   = ["ZMG — Todas las zonas"] + MUNICIPIOS_BASE

# Inicializar estado de sesión para el selectbox de área
if "area_sel_key" not in st.session_state:
    st.session_state.area_sel_key = "ZMG — Todas las zonas"

DIAS = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
DIA_NUM_MAP = {d:i for i,d in enumerate(DIAS)}
MESES_LABEL = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
               7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",
               11:"Noviembre",12:"Diciembre"}
FRANJAS = {
    0:"Madrugada  00–02h", 1:"Madrugada  03–05h",
    2:"Mañana     06–08h", 3:"Mañana     09–11h",
    4:"Tarde      12–14h", 5:"Tarde      15–17h",
    6:"Noche      18–20h", 7:"Noche      21–23h",
}
FRANJA_HORA_REP = {0:1,1:4,2:7,3:10,4:13,5:16,6:19,7:22}
COORD_MUN = {
    "Guadalajara":(20.6597,-103.3496), "Zapopan":(20.7214,-103.3912),
    "Tlaquepaque":(20.6364,-103.3122), "Tonalá":(20.6233,-103.2347),
}
ZOOM_MUN = {
    "Guadalajara":(20.6660,-103.3680,11.8),
    "Zapopan":    (20.7390,-103.4300,11.1),
    "Tlaquepaque":(20.6180,-103.3180,11.8),
    "Tonalá":     (20.6515,-103.2690,11.8),
}
POLIGONOS = {
    "Guadalajara":[[-103.415,20.710],[-103.380,20.715],[-103.345,20.705],
                   [-103.310,20.688],[-103.308,20.658],[-103.325,20.628],
                   [-103.360,20.618],[-103.400,20.625],[-103.425,20.648],
                   [-103.428,20.678],[-103.415,20.710]],
    "Zapopan":    [[-103.510,20.780],[-103.450,20.800],[-103.380,20.790],
                   [-103.340,20.760],[-103.340,20.720],[-103.380,20.715],
                   [-103.415,20.710],[-103.428,20.678],[-103.460,20.680],
                   [-103.500,20.710],[-103.520,20.745],[-103.510,20.780]],
    "Tlaquepaque":[[-103.360,20.618],[-103.325,20.628],[-103.308,20.658],
                   [-103.290,20.655],[-103.270,20.640],[-103.268,20.608],
                   [-103.290,20.585],[-103.330,20.578],[-103.360,20.590],
                   [-103.368,20.608],[-103.360,20.618]],
    "Tonalá":     [[-103.290,20.655],[-103.308,20.658],[-103.310,20.688],
                   [-103.295,20.695],[-103.265,20.688],[-103.235,20.665],
                   [-103.228,20.635],[-103.248,20.608],[-103.268,20.608],
                   [-103.270,20.640],[-103.290,20.655]],
}
NIVEL_COLOR = {"ALTO":"#ef4444","MEDIO":"#f59e0b","BAJO":"#10b981"}
NIVEL_EMOJI = {"ALTO":"","MEDIO":"","BAJO":""}
COLOR_MAP   = {
    "ALTO":  ("#ef4444","rgba(239,68,68,"),
    "MEDIO": ("#f59e0b","rgba(245,158,11,"),
    "BAJO":  ("#10b981","rgba(16,185,129,"),
}

# ── PREDICCIÓN ───────────────────────────────────────────────
# Features del clasificador v5 (incluye TIPO_DIA)
# Features del regresor v4 (sin TIPO_DIA)
FEAT_REG = [f for f in [
    'MES','DIA_NUM','FRANJA_NUM','MUN_ENC','MUN_RIESGO_RANK',
    'ES_FIN_SEMANA','ES_HORA_PICO','ES_NOCHE',
    'MES_SIN','MES_COS','CAUSA_ENC','TIPO_ENC'
] if f in FEATURE_COLS or True]   # regresor usa features originales

def predecir(mun, dia_str, franja_mod, mes):
    """
    Predicción con modelo v5 (clasificador ANOMALIA + regresor v4).
    TIPO_DIA es feature del clasificador pero no del regresor.
    """
    dia_num  = DIA_NUM_MAP.get(dia_str, 0)
    tipo_dia = int(dia_num >= 4)

    # Fila completa con todas las posibles features
    fila = {
        'MES':mes, 'DIA_NUM':dia_num, 'FRANJA_NUM':franja_mod,
        'TIPO_DIA':tipo_dia,
        'MUN_ENC':int(MUN_ENC_MAP.get(mun,0)),
        'MUN_RIESGO_RANK':
            {"Guadalajara":1,"Zapopan":2,"Tlaquepaque":3,"Tonalá":4}.get(mun,4),
        'ES_FIN_SEMANA':int(dia_num>=4),
        'ES_HORA_PICO':int(franja_mod in [1,2]),
        'ES_NOCHE':int(franja_mod==3),
        'MES_SIN':np.sin(2*np.pi*mes/12),
        'MES_COS':np.cos(2*np.pi*mes/12),
        'CAUSA_ENC':0,'TIPO_ENC':0,
    }

    # Regresor: features sin TIPO_DIA (modelo v4)
    FEAT_REG_ACTUAL = [f for f in [
        'MES','DIA_NUM','FRANJA_NUM','MUN_ENC','MUN_RIESGO_RANK',
        'ES_FIN_SEMANA','ES_HORA_PICO','ES_NOCHE',
        'MES_SIN','MES_COS','CAUSA_ENC','TIPO_ENC'
    ] if f in fila]
    X_reg = pd.DataFrame([fila])[FEAT_REG_ACTUAL]
    acc   = max(1, round(float(np.maximum(0, modelo_reg.predict(X_reg))[0])*FACTOR))

    # Clasificador: features del modelo v5 (puede incluir TIPO_DIA)
    cols_clf = [c for c in FEATURE_COLS if c in fila]
    X_clf = pd.DataFrame([fila])[cols_clf]
    prob  = float(modelo_clf.predict_proba(X_clf)[0,1])

    MEDIO_THR = MEDIO_THRESHOLD
    niv = ("ALTO"  if prob >= CLF_THRESHOLD else
           "MEDIO" if prob >= MEDIO_THR else "BAJO")
    return acc, prob, niv

def distribuir_ajustadores(disponibles: int, probs: dict) -> dict:
    """Distribuye disponibles entre municipios proporcional a su prob.
    El total nunca supera disponibles."""
    total_prob = sum(probs.values())
    if total_prob == 0:
        n = len(probs)
        return {k: disponibles // n for k in probs}
    asignados = {k: max(0, round(disponibles*(p/total_prob)))
                 for k,p in probs.items()}
    diff = disponibles - sum(asignados.values())
    if diff != 0:
        top = max(probs, key=probs.get)
        asignados[top] = max(0, asignados[top] + diff)
    return asignados

# Modelo v5 ya está balanceado — no requiere calibración post-hoc
RANGOS_CAL = None

# ── HEADER ───────────────────────────────────────────────────
st.markdown(f"""
<div class="header-bar">
  <div>
    <p class="header-title-text">SafeCity — Sistema de Predicción de Riesgo Vial</p>
  </div>
</div>""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''<div class="panel-control-header" style="background:linear-gradient(90deg,#0f2d52,#1a5fa8);padding:14px 16px;border-radius:10px;margin-bottom:12px"><p style="color:#ffffff !important;font-size:15px;font-weight:800;margin:0 !important;letter-spacing:-.2px">📋 Panel de Control</p></div>''', unsafe_allow_html=True)
    st.markdown("---")

    area_sel = st.selectbox(
        "Área", OPCIONES_AREA,
        key="area_sel_key"
    )
    dia_sel = st.selectbox(
        "Día", DIAS,
        index=_dia_idx   # día actual
    )
    mes_sel = st.selectbox(
        "Mes", list(MESES_LABEL.keys()),
        format_func=lambda x: MESES_LABEL[x],
        index=_mes_actual - 1   # mes actual
    )
    franja_dash = st.selectbox(
        "Franja Horaria", list(FRANJAS.keys()),
        format_func=lambda x: FRANJAS[x],
        index=_franja_idx   # franja actual según hora del sistema
    )
    ajust_disp = st.number_input(
        "Ajustadores Disponibles",
        min_value=1, max_value=500,
        value=35,   # 35 por defecto
        step=1,
        help="Total de ajustadores disponibles. Se distribuyen proporcionalmente entre zonas según el riesgo predicho."
    )

    st.markdown("---")
    st.markdown("### Configuración de Mapa")
    tipo_mapa = st.selectbox(
        "Visualización",
        ["Puntos de Riesgo", "Calor de Densidad", "Vista Híbrida"],
        index=0,
        help="Selecciona cómo visualizar la siniestralidad de las colonias."
    )
    estilo_mapa = st.selectbox(
        "Estilo del Mapa",
        ["Mapa Claro (Carto)", "Mapa Detallado (OSM)", "Mapa Contraste (Oscuro)"],
        index=0,
        help="Cambia el estilo visual base del mapa."
    )

    st.markdown("---")
    st.button("🔮 GENERAR PRONÓSTICO", use_container_width=True)

    # Botón Premium para Limpiar Caché
    st.markdown('<div class="cache-btn">', unsafe_allow_html=True)
    if st.button("LIMPIAR CACHÉ DEL SISTEMA", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.toast("Caché del sistema vaciada con éxito")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Info de calibración + última actualización (Tema Claro Premium)
    st.markdown(f"""
    <div style="margin-top:16px;padding:10px 12px;background:#ffffff;
                border-radius:6px;border:1px solid #e2e8f0;box-shadow: 0 1px 3px rgba(0,0,0,0.02)">
      <p style="color:#64748b;font-size:10px;font-weight:700;text-transform:uppercase;
                letter-spacing:.05em;margin:0 0 4px">Consulta actual</p>
      <p style="color:#0f172a;font-size:12px;font-weight:600;margin:0 0 8px">
        {_ahora.strftime('%d %b %Y · %H:%M')} h</p>

    </div>""", unsafe_allow_html=True)

# ── VARIABLES COMUNES ────────────────────────────────────────
area_activa   = area_sel
hora_rep      = FRANJA_HORA_REP[franja_dash]
franja_modelo = hora_a_franja_modelo(hora_rep)
es_zmg        = area_activa == "ZMG — Todas las zonas"

# Predecir los 4 municipios
resultados_mun = {}
for mun in MUNICIPIOS_BASE:
    acc, prob, niv = predecir(mun, dia_sel, franja_modelo, mes_sel)
    resultados_mun[mun] = {"acc":acc,"prob":prob,"nivel":niv,"ajust":0}

probs_mun   = {mun: v["prob"] for mun,v in resultados_mun.items()}
ajust_x_mun = distribuir_ajustadores(ajust_disp, probs_mun)
for mun in MUNICIPIOS_BASE:
    resultados_mun[mun]["ajust"] = ajust_x_mun[mun]

datos_mapa = [
    {"municipio":mun,
     "lat":COORD_MUN[mun][0],"lon":COORD_MUN[mun][1],
     "prob":v["prob"],"acc":v["acc"],"nivel":v["nivel"],
     "color":COLOR_MAP[v["nivel"]][0],
     "fill":COLOR_MAP[v["nivel"]][1]}
    for mun,v in resultados_mun.items()
]
df_mapa = pd.DataFrame(datos_mapa)

# Calcular predicciones para las colonias de todos los municipios (filtrando las que estén fuera de límites)
dplot_lista = []
for mun in MUNICIPIOS_BASE:
    sub_col = (df_colonias[df_colonias["municipio"]==mun].copy()
                 if len(df_colonias) and "municipio" in df_colonias.columns
                 else pd.DataFrame())
    if len(sub_col) > 0 and mun in POLIGONOS:
        sub_col = sub_col[sub_col.apply(lambda r: punto_en_poligono(r["lat"], r["lon"], POLIGONOS[mun]), axis=1)].reset_index(drop=True)
        
    v_mun = resultados_mun[mun]
    if len(sub_col) > 0 and modelo_col is not None:
        d_pred = predecir_por_colonia(
            sub_col,
            MUN_ENC_MAP,
            modelo_col,
            v_mun["prob"],
            v_mun["acc"],
            CLF_THRESHOLD,
            mes=mes_sel,
            dia_num=DIA_NUM_MAP.get(dia_sel, 0),
            franja_dash=franja_dash,
            shares_detalle=df_shares_det if len(df_shares_det) else None,
            shares_mun=df_shares_mun if len(df_shares_mun) else None,
            municipio=mun,
        )
        # Repartir los ajustadores asignados a este municipio a sus colonias
        pesos = d_pred["peso_col"].values.astype(float)
        if pesos.sum() > 0:
            d_pred["ajust_col"] = repartir_enteros(v_mun["ajust"], pesos)
        else:
            d_pred["ajust_col"] = 0
        dplot_lista.append(d_pred)
dplot_zmg = pd.concat(dplot_lista, ignore_index=True) if len(dplot_lista) > 0 else None

# ── KPI CARDS COMUNES A PANTALLA COMPLETA ───────────────────
sum_acc_zmg = sum(v["acc"] for v in resultados_mun.values())
prob_zmg  = float(np.mean([v["prob"] for v in resultados_mun.values()]))
nivel_zmg = ("ALTO"  if prob_zmg >= CLF_THRESHOLD else
             "MEDIO" if prob_zmg >= MEDIO_THRESHOLD else "BAJO")
nc_zmg    = NIVEL_COLOR[nivel_zmg]

if es_zmg:
    val_area = f"ZMG · RIESGO {nivel_zmg}"
    val_acc = f"{sum_acc_zmg} siniestros"
    val_ajust = f"{ajust_disp} asignados"
    nivel_actual = nivel_zmg
else:
    municipio_sel = area_activa
    v_sel = resultados_mun[municipio_sel]
    val_area = f"{municipio_sel.upper()} · RIESGO {v_sel['nivel']}"
    val_acc = f"{v_sel['acc']} siniestros"
    val_ajust = f"{v_sel['ajust']} asignados"
    nivel_actual = v_sel["nivel"]

# Calcular estilos dinámicos de colores para Card 1 basados en el nivel de riesgo actual
if nivel_actual == "ALTO":
    card1_style = "background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-left: 5px solid #ef4444; color: #7f1d1d;"
    card1_title_style = "color: #991b1b;"
    card1_desc_style = "color: #b91c1c;"
elif nivel_actual == "MEDIO":
    card1_style = "background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border-left: 5px solid #f59e0b; color: #78350f;"
    card1_title_style = "color: #92400e;"
    card1_desc_style = "color: #d97706;"
else:
    card1_style = "background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-left: 5px solid #10b981; color: #14532d;"
    card1_title_style = "color: #166534;"
    card1_desc_style = "color: #059669;"

card2_style = "background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-left: 5px solid #3b82f6; color: #1e3a8a;"
card2_title_style = "color: #1e40af;"
card2_desc_style = "color: #2563eb;"

card3_style = "background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); border-left: 5px solid #8b5cf6; color: #581c87;"
card3_title_style = "color: #6b21a8;"
card3_desc_style = "color: #7c3aed;"

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.markdown(f"""
<div class="kpi-card" style="{card1_style}">
  <p class="kpi-title" style="{card1_title_style}">Area y Nivel de Riesgo</p>
  <p class="kpi-value" style="color: inherit;">{val_area}</p>
  <p class="kpi-desc" style="{card1_desc_style}">Periodo: {dia_sel} · {FRANJAS[franja_dash].split("  ")[0].strip()}</p>
</div>""", unsafe_allow_html=True)
with kpi_col2:
    st.markdown(f"""
<div class="kpi-card" style="{card2_style}">
  <p class="kpi-title" style="{card2_title_style}">Siniestros Estimados</p>
  <p class="kpi-value" style="color: inherit;">{val_acc}</p>
  <p class="kpi-desc" style="{card2_desc_style}">Pronostico para la proxima franja horaria</p>
</div>""", unsafe_allow_html=True)
with kpi_col3:
    st.markdown(f"""
<div class="kpi-card" style="{card3_style}">
  <p class="kpi-title" style="{card3_title_style}">Ajustadores Recomendados</p>
  <p class="kpi-value" style="color: inherit;">{val_ajust} / {ajust_disp}</p>
  <p class="kpi-desc" style="{card3_desc_style}">Distribuidos proporcionalmente al riesgo</p>
</div>""", unsafe_allow_html=True)


# ── CÁLCULO DE DATOS PARA COMPONENTES ───────────────────────
if not es_zmg:
    municipio_sel = area_activa
    v_sel         = resultados_mun[municipio_sel]  # ya calibrado
    acc_est       = v_sel["acc"]
    prob_est      = v_sel["prob"]
    nivel         = v_sel["nivel"]
    ajust_total   = v_sel["ajust"]
    nc            = NIVEL_COLOR[nivel]

    # Colonias del municipio (filtrando las que estén fuera de límites)
    sub_col   = (df_colonias[df_colonias["municipio"]==municipio_sel].copy()
                 if len(df_colonias) and "municipio" in df_colonias.columns
                 else pd.DataFrame())
    if len(sub_col) > 0 and municipio_sel in POLIGONOS:
        sub_col = sub_col[sub_col.apply(lambda r: punto_en_poligono(r["lat"], r["lon"], POLIGONOS[municipio_sel]), axis=1)].reset_index(drop=True)
    dplot_mun = (
        predecir_por_colonia(
            sub_col,
            MUN_ENC_MAP,
            modelo_col,
            prob_est,
            acc_est,
            CLF_THRESHOLD,
            mes=mes_sel,
            dia_num=DIA_NUM_MAP.get(dia_sel, 0),
            franja_dash=franja_dash,
            shares_detalle=df_shares_det if len(df_shares_det) else None,
            shares_mun=df_shares_mun if len(df_shares_mun) else None,
            municipio=municipio_sel,
        )
        if len(sub_col) > 0 and modelo_col is not None
        else None
    )

    # Repartir ajustadores a nivel de colonia para el municipio seleccionado
    if dplot_mun is not None and len(dplot_mun) > 0:
        pesos_m = dplot_mun["peso_col"].values.astype(float)
        if pesos_m.sum() > 0:
            dplot_mun["ajust_col"] = repartir_enteros(ajust_total, pesos_m)
        else:
            dplot_mun["ajust_col"] = 0

    clat, clon, czoom = ZOOM_MUN[municipio_sel]

# ── CONSTRUCCIÓN DE FIGURAS DE FORMA REACTIVA ────────────────
# 1. Mapa Base: Renderizado dinámicamente vía Maplibre GL JS en la sección de layout.

# 2. Pronóstico Spline por Hora (Mantiene transiciones fluidas de los marcadores)
horas = list(range(1, 25))
if es_zmg:
    acc_h_z, prob_h_z = [], []
    for h in horas:
        fm = hora_a_franja_modelo(h)
        acc_h_z.append(int(np.mean([predecir(m,dia_sel,fm,mes_sel)[0] for m in MUNICIPIOS_BASE])))
        prob_h_z.append(float(np.mean([predecir(m,dia_sel,fm,mes_sel)[1] for m in MUNICIPIOS_BASE])))
    col_h_z = ["#ef4444" if p>=CLF_THRESHOLD else
                ("#f59e0b" if p>=MEDIO_THRESHOLD else "#10b981") for p in prob_h_z]

    fig_spline = go.Figure()
    fig_spline.add_trace(go.Scatter(
        x=horas, y=acc_h_z, mode="lines+markers",
        line=dict(color="#2563eb", width=3.5, shape="spline"),
        marker=dict(color=col_h_z, size=7, line=dict(width=1.2, color="white")),
        fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.08)",
        hovertemplate="<b>%{x}:00 h</b> — %{y} acc. promedio ZMG<extra></extra>"
    ))
    fig_spline.add_vline(x=hora_rep, line_dash="dash", line_color="#ef4444",
                     line_width=1.8, annotation_text=f" Actual: {hora_rep}:00h",
                     annotation_font_color="#ef4444", annotation_font_size=10,
                     annotation_position="top left")
    fig_spline.update_layout(
        transition=dict(duration=800, easing="cubic-in-out"),
        xaxis=dict(
            title=dict(text="Hora del día", font=dict(size=10, color="#64748b", family="Inter")), 
            tickvals=list(range(1,25,2)),
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(size=9, color="#64748b", family="Inter"),
            fixedrange=True
        ),
        yaxis=dict(
            title=dict(text="Siniestros (promedio)", font=dict(size=10, color="#64748b", family="Inter")), 
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(size=9, color="#64748b", family="Inter"),
            fixedrange=True
        ),
        margin=dict(l=15,r=15,t=10,b=15), height=150,
        paper_bgcolor="white", plot_bgcolor="white"
    )
else:
    acc_h, prob_h = [], []
    for h in horas:
        a,p,_ = predecir(municipio_sel, dia_sel, hora_a_franja_modelo(h), mes_sel)
        acc_h.append(a); prob_h.append(p)
    col_h = ["#ef4444" if p>=CLF_THRESHOLD else
              ("#f59e0b" if p>=MEDIO_THRESHOLD else "#10b981") for p in prob_h]

    fig_spline = go.Figure()
    fig_spline.add_trace(go.Scatter(
        x=horas, y=acc_h, mode="lines+markers",
        line=dict(color="#2563eb", width=3.5, shape="spline"),
        marker=dict(color=col_h, size=7, line=dict(width=1.2, color="white")),
        fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.08)",
        hovertemplate="<b>%{x}:00 h</b> — %{y} accidentes<extra></extra>"
    ))
    fig_spline.add_vline(x=hora_rep, line_dash="dash", line_color="#ef4444",
                    line_width=1.8, annotation_text=f" Actual: {hora_rep}:00h",
                    annotation_font_color="#ef4444", annotation_font_size=10,
                    annotation_position="top left")
    fig_spline.update_layout(
        transition=dict(duration=800, easing="cubic-in-out"),
        xaxis=dict(
            title=dict(text="Hora del día", font=dict(size=10, color="#64748b", family="Inter")), 
            tickvals=list(range(1,25,2)),
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(size=9, color="#64748b", family="Inter"),
            fixedrange=True
        ),
        yaxis=dict(
            title=dict(text="Siniestros estimados", font=dict(size=10, color="#64748b", family="Inter")), 
            gridcolor="#f1f5f9",
            linecolor="#cbd5e1",
            tickfont=dict(size=9, color="#64748b", family="Inter"),
            fixedrange=True
        ),
        margin=dict(l=15,r=15,t=10,b=15), height=150,
        paper_bgcolor="white", plot_bgcolor="white"
    )

# ── RENDERIZADO VISUAL UNIFICADO (Conserva el elemento DOM intacto para animar) ──
col_mapa, col_right = st.columns([1.45, 1.0])

with col_mapa:
    if es_zmg:
        map_title = "Mapa de Riesgo — Zona Metropolitana de Guadalajara"
    else:
        map_title = f"Mapa de Riesgo — {municipio_sel}"
    st.markdown(f'<p class="section-title">{map_title}</p>', unsafe_allow_html=True)
    
    # Renderizar el mapa de Maplibre GL JS con transiciones fluidas de cámara y glows neón
    if es_zmg:
        render_maplibre_map(
            20.672, -103.345, 10.8,
            "ZMG — Todas las zonas", dplot_zmg,
            estilo_mapa, tipo_mapa,
            dia_sel, mes_sel, franja_dash
        )
    else:
        render_maplibre_map(
            clat, clon, czoom,
            municipio_sel, dplot_mun,
            estilo_mapa, tipo_mapa,
            dia_sel, mes_sel, franja_dash
        )

    if es_zmg:
        spline_title = "Pronostico por Hora — ZMG (promedio 4 zonas)"
    else:
        spline_title = f"Pronostico por Hora — {municipio_sel}"
    st.markdown(f'<p class="section-title">{spline_title}</p>', unsafe_allow_html=True)
    
    # st.plotly_chart con key="pronostico_spline_global" desliza suavemente los puntos en pantalla
    st.plotly_chart(
        fig_spline, 
        use_container_width=True, 
        key="pronostico_spline_global", 
        config={'displayModeBar': False, 'responsive': True}
    )

with col_right:
    if es_zmg:
        rows_mun_html = ""
        for mun, v in resultados_mun.items():
            bcls = v["nivel"].lower()
            nc = NIVEL_COLOR[v["nivel"]]
            rows_mun_html += (
                f"<tr>"
                f"<td style='font-weight: 600;'>{mun}</td>"
                f"<td><span class='zmg-badge badge-{bcls}'><span class='risk-dot dot-{bcls}'></span>{v['nivel']}</span></td>"
                f"<td style='text-align: center;'>{v['acc']}</td>"
                f"<td style='text-align: center; font-weight: 700; color: {nc};'>{v['ajust']}</td>"
                f"</tr>"
            )
        
        st.markdown(f"""
<div class="dashboard-card">
<p class="section-title">Distribución por Municipio</p>
<table class="compact-table">
<thead>
<tr>
<th>Municipio</th>
<th>Riesgo</th>
<th style="text-align: center;">Accidentes</th>
<th style="text-align: center;">Ajustadores</th>
</tr>
</thead>
<tbody>
{rows_mun_html}
</tbody>
</table>
</div>
""", unsafe_allow_html=True)

        if dplot_zmg is not None and len(dplot_zmg) > 0:
            top_col = (
                dplot_zmg.copy()
                .sort_values("prob_col", ascending=False)
                .groupby(["colonia", "municipio"], as_index=False)
                .agg(
                    prob_col=("prob_col", "max"),
                    peso_col=("peso_col", "sum"),
                    heat_score=("heat_score", "max"),
                    nivel=("nivel", "first"),
                    ajust_col=("ajust_col", "max"),
                )
                .sort_values("prob_col", ascending=False)
                .head(8)
                .reset_index(drop=True)
            )

            rows_col_html = ""
            for i, row in top_col.iterrows():
                niv_c = str(row.get("nivel", "BAJO"))
                bcls = niv_c.lower()
                nc_c = NIVEL_COLOR.get(niv_c, "#6b7280")
                prob_c = float(row.get("prob_col", 0))
                aj_c = int(row.get("ajust_col", 0))
                nombre = f"{row['colonia']} ({row['municipio']})"
                rows_col_html += (
                    f"<tr>"
                    f"<td style='white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 140px;'>{nombre}</td>"
                    f"<td><span class='zmg-badge badge-{bcls}'><span class='risk-dot dot-{bcls}'></span>{niv_c}</span></td>"
                    f"<td style='text-align: center;'>{prob_c*100:.0f}%</td>"
                    f"<td style='text-align: center; font-weight: 700; color: {nc_c};'>{aj_c}</td>"
                    f"</tr>"
                )
            
            st.markdown(f"""
<div class="dashboard-card">
<p class="section-title">Colonias de Mayor Riesgo en la ZMG</p>
<table class="compact-table">
<thead>
<tr>
<th>Colonia (Municipio)</th>
<th>Riesgo</th>
<th style="text-align: center;">Prob.</th>
<th style="text-align: center;">Ajust.</th>
</tr>
</thead>
<tbody>
{rows_col_html}
</tbody>
</table>
</div>
""", unsafe_allow_html=True)

    else:
        if dplot_mun is not None and len(dplot_mun) > 0:
            top_col = (
                dplot_mun.copy()
                .sort_values("prob_col", ascending=False)
                .groupby("colonia", as_index=False)
                .agg(
                    prob_col=("prob_col", "max"),
                    peso_col=("peso_col", "sum"),
                    heat_score=("heat_score", "max"),
                    nivel=("nivel", "first"),
                    ajust_col=("ajust_col", "max"),
                )
                .sort_values("prob_col", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

            rows_col_html = ""
            for i, row in top_col.iterrows():
                niv_c  = str(row.get("nivel","BAJO"))
                bcls   = niv_c.lower()
                nc_c   = NIVEL_COLOR.get(niv_c,"#6b7280")
                prob_c = float(row.get("prob_col",0))
                aj_c   = int(row.get("ajust_col",0))
                nombre = str(row["colonia"])

                rows_col_html += (
                    f"<tr>"
                    f"<td style='white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 140px;'>{nombre}</td>"
                    f"<td><span class='zmg-badge badge-{bcls}'><span class='risk-dot dot-{bcls}'></span>{niv_c}</span></td>"
                    f"<td style='text-align: center;'>{prob_c*100:.0f}%</td>"
                    f"<td style='text-align: center; font-weight: 700; color: {nc_c};'>{aj_c}</td>"
                    f"</tr>"
                )

            st.markdown(f"""
<div class="dashboard-card">
<p class="section-title">Colonias de Mayor Riesgo en {municipio_sel}</p>
<table class="compact-table">
<thead>
<tr>
<th>Colonia</th>
<th>Riesgo</th>
<th style="text-align: center;">Prob.</th>
<th style="text-align: center;">Ajustadores</th>
</tr>
</thead>
<tbody>
{rows_col_html}
</tbody>
</table>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="dashboard-card">
<p class="section-title">Colonias de Mayor Riesgo en {municipio_sel}</p>
<p style="color: #475569; font-size: 11px; margin: 6px 0; font-weight: 500;">No hay datos de colonias disponibles para este municipio.</p>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<p style="text-align:center;color:#9ca3af;font-size:11px">
  SafeCity · XGBoost v4 · INEGI ATUS 2010–2024 · AUC-ROC 0.943 ·
  Recall 0.816 · Universidad Panamericana ·
  Última consulta: {_ahora.strftime('%d/%m/%Y %H:%M')}
</p>""", unsafe_allow_html=True)

# Las transiciones de cámara se manejan fluidamente a 60fps en el cliente vía Maplibre GL JS.
