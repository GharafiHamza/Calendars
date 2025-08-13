# sentinel_coverage_app.py
import json
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
import base64

from shapely.ops import unary_union, polygonize
try:
    from shapely.make_valid import make_valid  # Shapely ≥ 2.0
except Exception:
    make_valid = None

st.set_page_config(page_title="", layout="wide")

# --- Clickable Logo + Title header ---
def _embed_clickable_logo():
    # Try a local copy first; fall back to your uploaded path
    from pathlib import Path
    logo_candidates = [Path("ats_logo.png"), Path("./ats_logo.png")]
    logo_path = next((p for p in logo_candidates if p.exists()), None)
    if not logo_path:
        return  # silently skip if not found

    b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
    logo_html = f"""
    <a href="https://www.ariastechsolutions.com/" target="_blank" rel="noopener">
        <img src="data:image/png;base64,{b64}" style="height:60px; display:block;" />
    </a>
    """
    left, right = st.columns([1, 6])
    with left:
        st.markdown(logo_html, unsafe_allow_html=True)
    with right:
        st.markdown("<h1 style='margin:0'>Sentinel-1 &amp; Sentinel-2 Coverage Viewer</h1>", unsafe_allow_html=True)

_embed_clickable_logo()

# st.title("Sentinel-1 & Sentinel-2 Coverage Viewer")

# --- Configuration ---
DEFAULT_FILES = {
    "S1A": "S1A_12day_reference_coverage_plan.geojson",
    "S1C": "S1C_12day_reference_coverage_plan.geojson",
    "S2A": "S2A_10day_reference_coverage_plan.geojson",
    "S2B": "S2B_10day_reference_coverage_plan.geojson",
    "S2C": "S2C_10day_reference_coverage_plan.geojson",
}
PERIODS = {"S1A": 12, "S1C": 12, "S2A": 10, "S2B": 10, "S2C": 10}
SAT_COLORS = {
    "S1A": "#e41a1c",
    "S1C": "#377eb8",
    "S2A": "#4daf4a",
    "S2B": "#984ea3",
    "S2C": "#ff7f00"
}
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

# ---- Sidebar: AOI selection ----
st.sidebar.header("Area of Interest (AOI)")
AOI_FILES = {
    "Qatari EEZ": "aois/qatari_eez.geojson",
    "Saudi Arabia EEZ": "aois/saudi_arabia_eez.geojson",
}
QATARI_FALLBACK = "./qatari_eez.geojson"

aoi_choice = st.sidebar.selectbox("Select AOI", options=["(None)"] + list(AOI_FILES.keys()), index=0)

def load_aoi(choice: str) -> Optional[gpd.GeoDataFrame]:
    if choice not in AOI_FILES:
        return None
    path = Path(AOI_FILES[choice])
    raw = None
    if path.exists():
        raw = path.read_bytes()
    elif choice == "Qatari EEZ" and Path(QATARI_FALLBACK).exists():
        raw = Path(QATARI_FALLBACK).read_bytes()
    else:
        up = st.sidebar.file_uploader(f"Upload {choice} (GeoJSON)", type=["geojson"], key=f"aoi_{choice}")
        if up is not None:
            raw = up.read()

    if raw is None:
        st.sidebar.warning(f"{choice} file not found. Upload it to proceed.")
        return None

    try:
        gj = json.loads(raw.decode("utf-8"))
        gdf = gpd.GeoDataFrame.from_features(gj.get("features", []), crs="EPSG:4326")
        try:
            gdf["geometry"] = gdf["geometry"].buffer(0)
        except Exception:
            pass
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        st.sidebar.error(f"Failed to read {choice}: {e}")
        return None

AOI_GDF = load_aoi(aoi_choice) if aoi_choice != "(None)" else None

# ---- Sidebar: satellite plan inputs ----
uploaded_files: Dict[str, Optional[bytes]] = {}
for sat, fname in DEFAULT_FILES.items():
    path = Path(fname)
    if path.exists():
        uploaded_files[sat] = path.read_bytes()
    else:
        uf = st.sidebar.file_uploader(f"Upload {sat} plan ({fname})", type=["geojson"], key=f"u_{sat}")
        uploaded_files[sat] = uf.read() if uf is not None else None
        if uploaded_files[sat] is None:
            st.sidebar.warning(f"Waiting for {fname}")

if any(v is None for v in uploaded_files.values()):
    st.info("Please provide all five GeoJSON files to continue.")
    st.stop()

# --- Helpers ---
def parse_date_ymd(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def extract_dates_from_properties(props: dict) -> List[datetime]:
    dates = []
    for v in (props or {}).values():
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        for hit in DATE_RE.findall(v):
            dt = parse_date_ymd(hit)
            if dt:
                dates.append(dt)
    return dates

def feature_has_target_date(props: dict, target_ymd: str) -> bool:
    for v in (props or {}).values():
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        for hit in DATE_RE.findall(v):
            if hit == target_ymd:
                return True
    return False

def _make_valid_geom(geom):
    if geom is None:
        return geom
    try:
        if make_valid is not None:
            g = make_valid(geom)
        else:
            g = geom.buffer(0)
            if not g.is_valid:
                g = unary_union(list(polygonize(geom)))
    except Exception:
        g = geom
    return g

@st.cache_data(show_spinner=False)
def load_reference_dates_and_gdfs(file_blobs: Dict[str, bytes]):
    ref_dates: Dict[str, List[datetime]] = {}
    gdfs: Dict[str, gpd.GeoDataFrame] = {}
    raw_feats: Dict[str, List[dict]] = {}
    for sat, blob in file_blobs.items():
        gj = json.loads(blob.decode("utf-8"))
        feats = gj.get("features") or []
        raw_feats[sat] = feats
        dates: List[datetime] = []
        for f in feats:
            dates.extend(extract_dates_from_properties(f.get("properties", {})))
        ref_dates[sat] = sorted(set(dates))
        gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
        try:
            gdf["geometry"] = gdf["geometry"].buffer(0)
        except Exception:
            pass
        try:
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            else:
                gdf = gdf.to_crs(epsg=4326)
        except Exception:
            pass
        gdf = gdf.reset_index(drop=True)
        gdf["feat_idx"] = gdf.index
        gdfs[sat] = gdf
    return ref_dates, gdfs, raw_feats

REFERENCE_DATES, GDFS, RAW_FEATS = load_reference_dates_and_gdfs(uploaded_files)

def same_phase(chosen: datetime, ref: datetime, period_days: int) -> bool:
    return ((chosen.date() - ref.date()).days % period_days) == 0

def get_reference_date(chosen: datetime, reference_dates: List[datetime], period_days: int) -> Optional[datetime]:
    if not reference_dates:
        return None
    matches = [r for r in reference_dates if same_phase(chosen, r, period_days)]
    if matches:
        matches.sort()
        for r in reversed(matches):
            if r <= chosen:
                return r
        return matches[0]
    anchor = max([r for r in reference_dates if r <= chosen], default=min(reference_dates))
    shift = (chosen.date() - anchor.date()).days % period_days
    return (chosen - timedelta(days=shift)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Pick a date")
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    chosen_str = st.date_input("Date", pd.to_datetime(today_str)).strftime("%Y-%m-%d")
    chosen_dt = parse_date_ymd(chosen_str)
    refs = {}
    for sat, dates in REFERENCE_DATES.items():
        ref_dt = get_reference_date(chosen_dt, dates, PERIODS[sat])
        refs[sat] = ref_dt.strftime("%Y-%m-%d") if ref_dt else None
    st.header("Basemap")
    basemap = st.radio(
        "Choose basemap",
        options=["OpenStreetMap", "Carto Light"],
        index=0
    )

# ---- Map setup ----
m = folium.Map(location=[20, 10], zoom_start=3, tiles=None, control_scale=True)
folium.TileLayer("openstreetmap", name="OpenStreetMap", show=(basemap == "OpenStreetMap")).add_to(m)
folium.TileLayer(
    tiles="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
    attr="CartoDB Positron", name="Carto Light",
    show=(basemap == "Carto Light")
).add_to(m)

# AOI Layer
if AOI_GDF is not None and not AOI_GDF.empty:
    folium.GeoJson(
        AOI_GDF,
        name="AOI",
        style_function=lambda x: {"color": "yellow", "weight": 2, "fillOpacity": 0.1}
    ).add_to(m)

# Coverage union for percentage
coverage_union = None

# Satellite Layers — only swaths intersecting AOI if AOI present
for sat, gdf in GDFS.items():
    ref_day = refs.get(sat)
    if not ref_day or gdf.empty:
        continue
    feats = RAW_FEATS[sat]
    keep_idxs = [i for i, f in enumerate(feats) if feature_has_target_date(f.get("properties", {}), ref_day)]
    sub = gdf[gdf["feat_idx"].isin(keep_idxs)].copy()
    if sub.empty:
        continue
    if AOI_GDF is not None and not AOI_GDF.empty:
        sub = sub[sub.intersects(AOI_GDF.unary_union)]
    if sub.empty:
        continue

    if AOI_GDF is not None and not AOI_GDF.empty:
        if coverage_union is None:
            coverage_union = sub.unary_union
        else:
            coverage_union = coverage_union.union(sub.unary_union)

    folium.GeoJson(
        sub,
        name=sat,
        style_function=lambda x, sat=sat: {
            "fillColor": SAT_COLORS.get(sat, "#3186cc"),
            "color": SAT_COLORS.get(sat, "#3186cc"),
            "weight": 1, "fillOpacity": 0.25
        }
    ).add_to(m)

# Zoom to AOI if present
if AOI_GDF is not None and not AOI_GDF.empty:
    minx, miny, maxx, maxy = AOI_GDF.total_bounds.tolist()
    m.fit_bounds([[miny, minx], [maxy, maxx]])

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, height=720, use_container_width=True)

# Coverage percentage (only if AOI present)
if AOI_GDF is not None and coverage_union is not None:
    try:
        aoi_area = AOI_GDF.unary_union.area
        covered_area = AOI_GDF.unary_union.intersection(coverage_union).area
        coverage_percent = round((covered_area / aoi_area) * 100, 2)
        st.markdown(f"**{coverage_percent}%**")
    except Exception:
        pass
