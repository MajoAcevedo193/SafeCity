"""
Participación histórica por colonia/celda según mes, día y franja (siniestralimap).
Fase 2: pesos que reparten prob, accidentes y ajustadores dentro del municipio.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

MESES_TXT = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

DIAS_TXT = {
    "lunes": 0,
    "martes": 1,
    "miercoles": 2,
    "miércoles": 2,
    "jueves": 3,
    "viernes": 4,
    "sabado": 5,
    "sábado": 5,
    "domingo": 6,
}


def _sin_acentos(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def mes_texto_a_num(mes: str) -> int | None:
    if mes is None or (isinstance(mes, float) and np.isnan(mes)):
        return None
    if isinstance(mes, (int, np.integer)):
        m = int(mes)
        return m if 1 <= m <= 12 else None
    key = _sin_acentos(str(mes).strip().lower())
    return MESES_TXT.get(key)


def dia_texto_a_num(dia: str) -> int | None:
    if dia is None or (isinstance(dia, float) and np.isnan(dia)):
        return None
    key = _sin_acentos(str(dia).strip().lower())
    return DIAS_TXT.get(key)


def rango_hora_a_franja(rango: str) -> int:
    """Franja 0–7 alineada al dashboard (FRANJAS 00–02 … 21–23)."""
    m = re.search(r"(\d{1,2})\s*:", str(rango))
    if not m:
        return 4
    h = int(m.group(1))
    if h <= 2:
        return 0
    if h <= 5:
        return 1
    if h <= 8:
        return 2
    if h <= 11:
        return 3
    if h <= 14:
        return 4
    if h <= 17:
        return 5
    if h <= 20:
        return 6
    return 7


def preparar_accidentes_temporales(df_acc: pd.DataFrame) -> pd.DataFrame:
    d = df_acc.copy()
    d["mes_num"] = d["mes"].map(mes_texto_a_num) if "mes" in d.columns else None
    if d["mes_num"].isna().all() and "anio" in d.columns:
        pass
    d["dia_num"] = d["dia_sem"].map(dia_texto_a_num) if "dia_sem" in d.columns else None
    d["franja"] = d["rango_hora"].map(rango_hora_a_franja) if "rango_hora" in d.columns else 4
    return d.dropna(subset=["mes_num", "dia_num"], how="any")


def construir_shares_detalle(df_acc: pd.DataFrame) -> pd.DataFrame:
    """
    Conteos por celda y contexto temporal.
    Columnas: municipio, lat, lon, colonia, mes, dia_num, franja, n_acc
    """
    d = preparar_accidentes_temporales(df_acc)
    d["mes"] = d["mes_num"].astype(int)
    d["dia_num"] = d["dia_num"].astype(int)
    d["franja"] = d["franja"].astype(int)

    agg = (
        d.groupby(
            ["municipio", "lat", "lon", "colonia", "mes", "dia_num", "franja"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_acc"})
    )
    return agg


def construir_shares_municipio(df_acc: pd.DataFrame) -> pd.DataFrame:
    """Conteo estático por celda (sin filtro temporal)."""
    return (
        df_acc.groupby(["municipio", "lat", "lon", "colonia"], as_index=False)
        .size()
        .rename(columns={"size": "n_acc"})
    )


def _conteo_contexto(
    shares_detalle: pd.DataFrame,
    municipio: str,
    mes: int | None,
    dia_num: int | None,
    franja: int | None,
) -> pd.Series:
    """Devuelve Series index (lat,lon) -> n para el contexto pedido, con fallbacks."""
    sd = shares_detalle
    sd = sd[sd["municipio"] == municipio]
    if len(sd) == 0:
        return pd.Series(dtype=float)

    def _sum(key_filter) -> pd.Series:
        sub = sd.loc[key_filter]
        if len(sub) == 0:
            return pd.Series(dtype=float)
        g = sub.groupby(["lat", "lon"])["n_acc"].sum()
        return g

    m = mes is not None
    d = dia_num is not None
    f = franja is not None

    if m and d and f:
        s = _sum((sd["mes"] == mes) & (sd["dia_num"] == dia_num) & (sd["franja"] == franja))
        if len(s):
            return s
    if m and f:
        s = _sum((sd["mes"] == mes) & (sd["franja"] == franja))
        if len(s):
            return s
    if d and f:
        s = _sum((sd["dia_num"] == dia_num) & (sd["franja"] == franja))
        if len(s):
            return s
    if m and d:
        s = _sum((sd["mes"] == mes) & (sd["dia_num"] == dia_num))
        if len(s):
            return s
    if m:
        s = _sum(sd["mes"] == mes)
        if len(s):
            return s
    if f:
        s = _sum(sd["franja"] == franja)
        if len(s):
            return s
    return sd.groupby(["lat", "lon"])["n_acc"].sum()


def pesos_para_celdas(
    df_celdas: pd.DataFrame,
    shares_detalle: pd.DataFrame | None,
    shares_mun: pd.DataFrame | None,
    municipio: str,
    mes: int,
    dia_num: int,
    franja: int,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Peso por fila de df_celdas (mismo municipio).
    Combina contexto temporal + concurrencia estática (suavizado).
    """
    n = len(df_celdas)
    if n == 0:
        return np.array([])

    keys = list(zip(df_celdas["lat"].round(2), df_celdas["lon"].round(2)))
    temp = np.zeros(n, dtype=float)
    if shares_detalle is not None and len(shares_detalle):
        conteo = _conteo_contexto(shares_detalle, municipio, mes, dia_num, franja)
        if len(conteo):
            for i, (la, lo) in enumerate(keys):
                temp[i] = float(conteo.get((la, lo), 0.0))

    base = df_celdas["concurrencia"].astype(float).values
    if base.sum() <= 0:
        base = np.ones(n)

    if temp.sum() > 0:
        temp = temp / temp.sum()
        base = base / base.sum()
        pesos = (1.0 - alpha) * temp + alpha * base
    else:
        pesos = base / base.sum()

    pesos = np.maximum(pesos, 1e-9)
    return pesos / pesos.sum()


def repartir_enteros(total: int, pesos: np.ndarray) -> np.ndarray:
    """Reparte `total` en enteros ≥ 0; la suma coincide con total."""
    total = int(max(0, total))
    n = len(pesos)
    if n == 0:
        return np.array([], dtype=int)
    if total == 0:
        return np.zeros(n, dtype=int)
    p = np.asarray(pesos, dtype=float)
    if p.sum() <= 0:
        p = np.ones(n) / n
    else:
        p = p / p.sum()
    raw = p * total
    out = np.floor(raw).astype(int)
    rem = total - int(out.sum())
    if rem > 0:
        frac = raw - out
        for idx in np.argsort(-frac)[:rem]:
            out[idx] += 1
    return out


def distribuir_ajustadores_por_peso(disponibles: int, pesos: np.ndarray) -> np.ndarray:
    """Igual que repartir_enteros pero mínimo 1 en celdas con peso > 0 si hay cupo."""
    disp = int(max(0, disponibles))
    n = len(pesos)
    if n == 0 or disp == 0:
        return np.zeros(n, dtype=int)
    base = repartir_enteros(disp, pesos)
    return base
