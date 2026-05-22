# ============================================================
# Colonias — calor, ML (Gradient Boosting) y figuras de mapa
# Usado por 04_dashboard_safecity_v4.py
# ============================================================
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor

from colonias_temporal import (
    distribuir_ajustadores_por_peso,
    pesos_para_celdas,
    repartir_enteros,
)

# Umbral reutilizo el del clf municipal (MEDIO/ALTO)
def _nivel_riesgo(prob: float, clf_threshold: float) -> str:
    if prob >= clf_threshold:
        return "ALTO"
    if prob >= 0.35:
        return "MEDIO"
    return "BAJO"


NIVEL_COLORS = {
    "ALTO":  ("#ef4444", "rgba(239,68,68,"),
    "MEDIO": ("#f59e0b", "rgba(245,158,11,"),
    "BAJO":  ("#10b981", "rgba(16,185,129,"),
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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


def _opacity_por_chaos(score_0_1: float) -> float:
    """Más caos => más sólido (opaco). Menos caos => más tenue."""
    s = _clamp(float(score_0_1), 0.0, 1.0)
    return 0.10 + 0.85 * s


def _opacity_municipio(score_0_1: float) -> float:
    """
    Igual que el caos, pero con tope para no tapar el mapa (calles) debajo.
    """
    s = _clamp(float(score_0_1), 0.0, 1.0)
    return 0.08 + 0.34 * s  # max ~0.42


def entrenar_modelo_intensidad(df: pd.DataFrame, mun_enc_map: dict) -> GradientBoostingRegressor | None:
    """
    Modelo supervisado: concurrencia (o proxy) en función de posición y municipio.
    Con ~100 filas, max_depth bajo y pocos estimadores.
    """
    if df is None or len(df) < 8:
        return None
    d = df.copy()
    d["MUN_ENC"] = d["municipio"].map(lambda m: int(mun_enc_map.get(m, 0)))
    d["conc"] = d["concurrencia"].replace(0, np.nan)
    d["conc"] = d["conc"].fillna(d["rank_metro"].max() + 1 - d["rank_metro"])
    y = d["conc"].astype(float).values
    X = d[["lat", "lon", "MUN_ENC", "rank_metro"]].values
    m = GradientBoostingRegressor(
        n_estimators=64,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    m.fit(X, y)
    return m


def predecir_por_colonia(
    df_mun: pd.DataFrame,
    mun_enc_map: dict,
    modelo: GradientBoostingRegressor | None,
    prob_mun: float,
    acc_mun: int,
    clf_threshold: float,
    mes: int | None = None,
    dia_num: int | None = None,
    franja_dash: int | None = None,
    shares_detalle: pd.DataFrame | None = None,
    shares_mun: pd.DataFrame | None = None,
    municipio: str | None = None,
) -> pd.DataFrame:
    """
    Fase 2: reparte prob/acc/ajust por participación histórica (mes, día, franja)
    + intensidad espacial. La suma de acc_col ≈ acc_mun; prob_col ≤ prob_mun.
    """
    d = df_mun.copy()
    d["MUN_ENC"] = d["municipio"].map(lambda m: int(mun_enc_map.get(m, 0)))
    d["conc"] = d["concurrencia"].replace(0, np.nan)
    d["conc"] = d["conc"].fillna(d["rank_metro"].max() + 1 - d["rank_metro"])
    n = len(d)
    if n == 0:
        return d

    mun = municipio or str(d["municipio"].iloc[0])
    mes_u = int(mes) if mes is not None else 1
    dia_u = int(dia_num) if dia_num is not None else 0
    franja_u = int(franja_dash) if franja_dash is not None else 4

    pesos = pesos_para_celdas(
        d, shares_detalle, shares_mun, mun, mes_u, dia_u, franja_u
    )
    d["peso_col"] = pesos

    if modelo is not None:
        Xp = d[["lat", "lon", "MUN_ENC", "rank_metro"]].values
        pred = np.maximum(0, modelo.predict(Xp))
    else:
        pred = d["conc"].values.astype(float)
    pmin, pmax = float(np.min(pred)), float(np.max(pred))
    if pmax - pmin < 1e-9:
        heat_ml = np.ones(n) * 0.5
    else:
        heat_ml = (pred - pmin) / (pmax - pmin)

    # heat_score: mezcla participación temporal (70%) y ML espacial (30%)
    pmin_w, pmax_w = float(pesos.min()), float(pesos.max())
    heat_temp = (pesos - pmin_w) / (pmax_w - pmin_w + 1e-9)
    d["heat_score"] = np.clip(0.70 * heat_temp + 0.30 * heat_ml, 0.0, 1.0)

    w_max = float(pesos.max())
    rel = pesos / (w_max + 1e-9)
    d["prob_col"] = np.clip(prob_mun * (0.06 + 0.94 * rel), 0.0, prob_mun)

    d["acc_col"] = repartir_enteros(int(acc_mun), pesos)
    d["ajust_col"] = distribuir_ajustadores_por_peso(0, pesos)  # se rellena en dashboard

    d["nivel"] = d["prob_col"].apply(lambda p: _nivel_riesgo(p, clf_threshold))
    d["rank_local"] = d["prob_col"].rank(ascending=False, method="first").astype(int)
    d = d.sort_values("prob_col", ascending=False)
    return d


def fig_mapa_colonias(
    df_mapa: pd.DataFrame,
    polys: dict,
    dplot: pd.DataFrame,
    clat: float,
    clon: float,
    czoom: float,
    municipio_sel: str,
    clf_threshold: float,
    tipo_mapa: str = "Puntos de Riesgo",
    estilo_mapa: str = "Mapa Claro (Carto)",
) -> go.Figure:
    """
    Mantiene la figura del municipio (polígonos) y dibuja un sub-mapa:
    - Contornos municipales resaltados o difuminados según selección
    - Puntos interactivos con color, tamaño y opacidad proporcionales al riesgo individual
    - Capa de densidad de calor (Densitymapbox) opcional
    - Leyenda interactiva para filtrar niveles de riesgo en el mapa
    """
    fig = fig_mapa_municipios(df_mapa, polys, municipio_sel, clat, clon, czoom, estilo_mapa=estilo_mapa)

    if dplot is not None and len(dplot) > 0:
        dd = dplot.copy()
        
        # Filtrar colonias que estén dentro de los límites de su respectivo municipio
        def _colonia_dentro(row):
            m = row["municipio"]
            if m in polys:
                return punto_en_poligono(float(row["lat"]), float(row["lon"]), polys[m])
            return False
            
        dd = dd[dd.apply(_colonia_dentro, axis=1)].reset_index(drop=True)
        
        # Rellenar NaNs para evitar crashes de serialización
        dd["prob_col"] = dd["prob_col"].fillna(0.0)
        dd["acc_col"] = dd["acc_col"].fillna(0).astype(int)
        dd["heat_score"] = dd["heat_score"].fillna(0.0)
        dd["colonia"] = dd["colonia"].fillna("Desconocida").astype(str)
        dd["municipio"] = dd["municipio"].fillna("Desconocido").astype(str)
        
        # 1. Agregar capa de calor por densidad si se solicita
        if tipo_mapa in ["Calor de Densidad", "Vista Híbrida"]:
            fig.add_trace(
                go.Densitymapbox(
                    lat=dd["lat"].astype(float).tolist(),
                    lon=dd["lon"].astype(float).tolist(),
                    z=(dd["prob_col"].astype(float) * 100).tolist(),
                    radius=20,
                    colorscale=[
                        [0.0, "rgba(16,185,129,0)"],
                        [0.2, "rgba(16,185,129,0.35)"],
                        [0.5, "rgba(245,158,11,0.65)"],
                        [0.8, "rgba(239,68,68,0.85)"],
                        [1.0, "rgba(220,38,38,0.98)"]
                    ],
                    showscale=False,
                    name="Calor de Densidad",
                    hovertemplate=(
                        "<b>Área de Riesgo Densificado</b><br>"
                        "Densidad de Siniestros Relativa: <b>%{z:.1f}%</b><extra></extra>"
                    )
                )
            )

        # 2. Agregar puntos de colonias si se solicita
        if tipo_mapa in ["Puntos de Riesgo", "Vista Híbrida"]:
            niveles_config = [
                ("BAJO", "#10b981", "Bajo Riesgo"),
                ("MEDIO", "#f59e0b", "Medio Riesgo"),
                ("ALTO", "#ef4444", "Alto Riesgo")
            ]
            
            for nivel_c, c_hex, label_name in niveles_config:
                sub_dd = dd[dd["nivel"] == nivel_c]
                if len(sub_dd) == 0:
                    continue
                    
                # Calcular tamaños proporcionales al riesgo (convertido a listas nativas)
                if nivel_c == "ALTO":
                    opac = 0.85
                    sizes = (8.0 + 14.0 * sub_dd["prob_col"].astype(float)).tolist()
                elif nivel_c == "MEDIO":
                    opac = 0.70
                    sizes = (6.0 + 10.0 * sub_dd["prob_col"].astype(float)).tolist()
                else:
                    opac = 0.50
                    sizes = (4.0 + 6.0 * sub_dd["prob_col"].astype(float)).tolist()
                    
                fig.add_trace(
                    go.Scattermapbox(
                        lat=sub_dd["lat"].astype(float).tolist(),
                        lon=sub_dd["lon"].astype(float).tolist(),
                        mode="markers",
                        marker=dict(
                            size=sizes,
                            color=c_hex,
                            opacity=opac,
                        ),
                        name=label_name,
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "<span style='color:#94a3b8'>Municipio:</span> %{customdata[3]}<br>"
                            "<span style='color:#94a3b8'>Prob. Siniestro:</span> <b>%{customdata[0]:.1f}%</b><br>"
                            "<span style='color:#94a3b8'>Acc. Estimados:</span> <b>%{customdata[1]}</b><br>"
                            "<span style='color:#94a3b8'>Intensidad:</span> <b>%{customdata[2]:.0f}/100</b><extra></extra>"
                        ),
                        text=sub_dd["colonia"].tolist(),
                        customdata=list(zip(
                            (sub_dd["prob_col"].astype(float) * 100).tolist(),
                            sub_dd["acc_col"].astype(int).tolist(),
                            (sub_dd["heat_score"].astype(float) * 100).tolist(),
                            sub_dd["municipio"].tolist()
                        )),
                        showlegend=True,
                    )
                )

    return fig


def fig_tabla_ranking(dplot: pd.DataFrame, municipio_sel: str, top_n: int = 10) -> go.Figure:
    if dplot is None:
        return go.Figure()
    d = dplot.copy()
    if len(d) == 0:
        return go.Figure()

    # Agregar por colonia: una fila por colonia (la de mayor prob_col)
    d = d.sort_values("prob_col", ascending=False)
    d = d.groupby("colonia", as_index=False).first()
    d = d.sort_values("prob_col", ascending=False).head(top_n).reset_index(drop=True)
    d["rank_local"] = d.index + 1

    prob = (d["prob_col"].astype(float) * 100).round(1).map(lambda x: f"{x:.1f}%").tolist()
    intens = (d["heat_score"].astype(float) * 100).round(0).astype(int).map(lambda x: f"{x}/100").tolist()
    header_color = "#1e3a5f"
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>#</b>",
                        "<b>Colonia</b>",
                        "<b>Nivel</b>",
                        "<b>Prob.</b>",
                        "<b>Acc. est.</b>",
                        "<b>Intensidad</b>",
                    ],
                    fill_color=header_color,
                    font=dict(color="white", size=12),
                    align=["center", "left", "center", "center", "center", "center"],
                    height=28,
                ),
                cells=dict(
                    values=[
                        d["rank_local"].astype(int).tolist(),
                        d["colonia"].astype(str).tolist(),
                        d["nivel"].astype(str).tolist(),
                        prob,
                        d["acc_col"].astype(int).tolist(),
                        intens,
                    ],
                    fill_color="white",
                    font=dict(color="#0f172a", size=12),
                    align=["center", "left", "center", "center", "center", "center"],
                    height=24,
                ),
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=320,
        paper_bgcolor="white",
        title=dict(
            text=f"Top 10 colonias con mas incidencias — {municipio_sel}",
            font=dict(size=14, color="#1e3a5f"),
            x=0.01,
            y=0.98,
        ),
    )
    return fig


def fig_mapa_municipios(
    df_mapa: pd.DataFrame,
    polys: dict,
    municipio_sel: str,
    clat: float,
    clon: float,
    czoom: float,
    estilo_mapa: str = "Mapa Claro (Carto)",
) -> go.Figure:
    """Vista ZMG 4 polígonos con mejoras visuales y estilo de mapa base dinámico."""
    fig = go.Figure()
    is_zmg_mode = (municipio_sel == "ZMG — Todas las zonas")
    
    # 1. Dibujar polígonos
    for _, row in df_mapa.iterrows():
        poly = polys[row["municipio"]]
        es_sel = row["municipio"] == municipio_sel
        
        c, f = NIVEL_COLORS.get(str(row.get("nivel", "BAJO")), NIVEL_COLORS["BAJO"])
        
        if is_zmg_mode:
            opac = round(_opacity_municipio(float(row["prob"])), 2)
            fig.add_trace(
                go.Scattermapbox(
                    lat=[float(coord[1]) for coord in poly],
                    lon=[float(coord[0]) for coord in poly],
                    mode="lines",
                    fill="toself",
                    fillcolor=f"{f}{opac})",
                    line=dict(color=c, width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        elif es_sel:
            # Efecto Glow de contorno para el municipio seleccionado (tres capas concéntricas potenciadas)
            # Capa 1: Resplandor externo translúcido y extra grueso (Brillo Neón Exterior)
            fig.add_trace(
                go.Scattermapbox(
                    lat=[float(coord[1]) for coord in poly],
                    lon=[float(coord[0]) for coord in poly],
                    mode="lines",
                    line=dict(color="rgba(37, 99, 235, 0.25)", width=12.0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Capa 2: Resplandor medio (Brillo Neón Medio)
            fig.add_trace(
                go.Scattermapbox(
                    lat=[float(coord[1]) for coord in poly],
                    lon=[float(coord[0]) for coord in poly],
                    mode="lines",
                    line=dict(color="rgba(37, 99, 235, 0.45)", width=7.0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            # Capa 3: Contorno principal nítido y relleno iluminado
            fig.add_trace(
                go.Scattermapbox(
                    lat=[float(coord[1]) for coord in poly],
                    lon=[float(coord[0]) for coord in poly],
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(37, 99, 235, 0.16)",
                    line=dict(color="#2563eb", width=2.5),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:
            # Municipios no seleccionados
            fig.add_trace(
                go.Scattermapbox(
                    lat=[float(coord[1]) for coord in poly],
                    lon=[float(coord[0]) for coord in poly],
                    mode="lines",
                    fill="toself",
                    fillcolor=f"{f}0.06)",
                    line=dict(color=c, width=0.8),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        
    # 2. Dibujar marcadores municipales
    for _, row in df_mapa.iterrows():
        es_sel = row["municipio"] == municipio_sel
        mc, _ = NIVEL_COLORS.get(str(row.get("nivel", "BAJO")), NIVEL_COLORS["BAJO"])
        
        # En modo ZMG, los centros son sutiles para no tapar las colonias
        if is_zmg_mode:
            marker_size = 10
            marker_opacity = 0.5
            text_size = 9.5
            text_font_family = "Inter"
            mode_draw = "markers+text"
        else:
            # En modo municipio, todos los municipios se ven al mismo tiempo
            if es_sel:
                marker_size = 18 + float(row["prob"]) * 10
                marker_opacity = 0.75
                text_size = 11
                text_font_family = "Inter"
                mode_draw = "markers+text"
            else:
                marker_size = 8
                marker_opacity = 0.25
                text_size = 9
                text_font_family = "Inter"
                mode_draw = "markers+text"
            
        fig.add_trace(
            go.Scattermapbox(
                lat=[float(row["lat"])],
                lon=[float(row["lon"])],
                mode=mode_draw,
                marker=dict(
                    size=marker_size,
                    color=mc,
                    opacity=marker_opacity,
                ),
                text=[f"  {row['municipio']}"],
                textposition="middle right",
                textfont=dict(
                    color="#1e293b",  # Texto oscuro para legibilidad en fondo claro
                    size=text_size,
                    family=text_font_family,
                ),
                hovertemplate=(
                    f"<b>{row['municipio']}</b><br>Nivel General: <b>{row['nivel']}</b><br>"
                    f"Prob: {float(row['prob'])*100:.0f}%<br>Acc. est: {int(row['acc'])}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        
    # Halo de selección brillante en lugar de la estrella para evitar warnings del sprite de Mapbox
    if not is_zmg_mode and municipio_sel and municipio_sel in df_mapa["municipio"].values:
        sel = df_mapa[df_mapa["municipio"] == municipio_sel].iloc[0]
        # Halo concéntrico triple para efecto glow premium
        fig.add_trace(
            go.Scattermapbox(
                lat=[float(sel["lat"])],
                lon=[float(sel["lon"])],
                mode="markers",
                marker=dict(size=marker_size + 6, color="#2563eb", opacity=0.4),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=[float(sel["lat"])],
                lon=[float(sel["lon"])],
                mode="markers",
                marker=dict(size=marker_size + 14, color="#3b82f6", opacity=0.2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lat=[float(sel["lat"])],
                lon=[float(sel["lon"])],
                mode="markers",
                marker=dict(size=marker_size + 24, color="#60a5fa", opacity=0.08),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        
    # Map estilo_mapa to the official Plotly Mapbox style names
    style_mapping = {
        "Mapa Claro (Carto)": "carto-positron",
        "Mapa Detallado (OSM)": "open-street-map",
        "Mapa Contraste (Oscuro)": "carto-darkmatter"
    }
    mapbox_style = style_mapping.get(estilo_mapa, "carto-positron")

    fig.update_layout(
        transition=dict(duration=800, easing="cubic-in-out"),
        dragmode="pan",
        mapbox=dict(
            style=mapbox_style, 
            center=dict(lat=clat, lon=clon), 
            zoom=czoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=310,  # Reducido para que quepa en la pantalla
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.90)",  # Light theme legend background
            bordercolor="#e2e8f0",
            borderwidth=1.5,
            font=dict(size=10, color="#334155", family="Inter")
        )
    )
    return fig

