import streamlit as st
import networkx as nx
import pickle
import gzip
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString
import geopandas as gpd
from geopy.distance import geodesic
import pandas as pd
import os
from huggingface_hub import hf_hub_download

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Taipei Map", layout="wide")

# --- 設定區 ---
DATA_REPO_ID = "ZnCYLin/north-taiwan-map-data"  # 👈 請記得修改這裡
DATA_FILENAME = "north_taiwan_ready.pkl.gz"
CSV_FILENAME = "stations_master.csv"

# --- 2. 資料載入 ---
@st.cache_resource(show_spinner="正在從雲端下載地圖資料...")
def load_core_data():
    G_drive, G_walk, stations = None, None, None
    try:
        # 下載
        local_path = hf_hub_download(repo_id=DATA_REPO_ID, filename=DATA_FILENAME, repo_type="model")
        # 讀取
        with gzip.open(local_path, "rb") as f:
            G_raw = pickle.load(f)
        
        G_drive = G_raw 
        # 步行圖層過濾
        def filter_walk(u, v, k, d): return d.get('time_walk', 999999) < 1000
        G_walk = nx.subgraph_view(G_raw, filter_edge=filter_walk)
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

    if os.path.exists(CSV_FILENAME):
        try:
            stations = pd.read_csv(CSV_FILENAME)
            stations['unique_id'] = stations.apply(lambda row: f"{row['name']}_{row['line_id']}", axis=1)
            stations['node_id'] = stations['unique_id'].apply(lambda x: f"STATION_{x}")
        except: pass
    return G_drive, G_walk, stations

G_drive, G_walk, stations_df = load_core_data()

if G_drive is None:
    st.error("Data missing.")
    st.stop()

# --- 3. 邏輯類別 ---
def get_nearest_node(G, point):
    t_lat, t_lon = point
    best, min_d = None, 100.0
    for n, d in G.nodes(data=True):
        if 'y' not in d: continue
        dy, dx = d['y'] - t_lat, d['x'] - t_lon
        dist = dy*dy + dx*dx
        if dist < min_d: min_d, best = dist, n
    return best

class RailSystem:
    def __init__(self, df, year):
        self.stations, self.node_map, self.lines = {}, {}, []
        if df is None: return
        target_year = int(year)
        valid = ['Operating']
        if target_year >= 2028: valid.append('Construction')
        if target_year >= 2031: valid.append('Planning')

        active = df[df['status'].isin(valid)]
        for _, r in active.iterrows():
            uid = r['unique_id']
            self.stations[uid] = (r['lat'], r['lon'])
            self.node_map[uid] = r['node_id']

        colors = {"BL": "#0070BD", "R": "#E3002C", "G": "#008659", "O": "#F8B61C", "BR": "#C48C31", "Y": "#FDD935", "A": "#8246AF", "LB": "#6C9ED3"}
        self.rail_G = nx.Graph()

        # 簡單建立路網，略過複雜邏輯以確保穩定
        for lid, grp in active.groupby('line_id'):
            grp = grp.sort_values('sequence')
            ids = grp['unique_id'].tolist()
            coords = [self.stations[i] for i in ids]
            c = colors.get(lid, "gray")
            # 這裡只存純資料，不存物件
            self.lines.append({"coords": coords, "color": c})
            
            spd = 55.0 if lid.startswith(('A', 'TRA')) else 35.0
            for i in range(len(ids) - 1):
                u, v = ids[i], ids[i + 1]
                dist = geodesic(self.stations[u], self.stations[v]).km
                self.rail_G.add_edge(u, v, weight=dist*(60/spd)+0.5)

        # 轉乘邏輯 (簡化)
        uids = list(self.stations.keys())
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                u, v = uids[i], uids[j]
                if u.split('_')[-1] == v.split('_')[-1]: continue
                dist = geodesic(self.stations[u], self.stations[v]).meters
                if dist < 450: self.rail_G.add_edge(u, v, weight=5.0)

    def get_sources(self, start, limit, wait_time=0):
        entry = []
        for uid, pos in self.stations.items():
            dist = geodesic(start, pos).meters * 1.35
            t = dist / (4.0 * 1000 / 60) + wait_time
            if t < limit: entry.append((uid, t))
        if not entry: return []
        
        temp_G = self.rail_G.copy()
        temp_G.add_node("S")
        for u, t in entry: temp_G.add_edge("S", u, weight=t)
        
        paths = nx.single_source_dijkstra_path_length(temp_G, "S", cutoff=limit)
        res = []
        for uid, cost in paths.items():
            if uid == "S": continue
            rem = limit - cost - 3.0
            if rem > 0 and self.node_map[uid] in G_walk.nodes:
                res.append((self.node_map[uid], rem))
        return res

def compute(start, mode, limit, rs, detailed=False, wait_penalty=0):
    G = G_walk if mode in ['rail', 'walk'] else G_drive
    metric = 'time_walk' if mode in ['rail', 'walk'] else f'time_{mode}'
    
    targets = rs.get_sources(start, limit, wait_penalty) if mode == 'rail' else []
    if mode != 'rail':
        sn = get_nearest_node(G, start)
        if sn: targets = [(sn, limit)]

    all_pts = []
    edges = []
    for node, rem in targets:
        try:
            sub = nx.ego_graph(G, node, radius=rem, distance=metric)
            # 收集點座標
            pts = [Point(G.nodes[n]['x'], G.nodes[n]['y']) for n in sub.nodes]
            if pts: all_pts.extend(pts)
            # 詳細模式線條
            if detailed:
                for u, v, d in sub.edges(data=True):
                    if 'geometry' in d: edges.append(d['geometry'])
                    else: edges.append(LineString([Point(G.nodes[u]['x'], G.nodes[u]['y']), Point(G.nodes[v]['x'], G.nodes[v]['y'])]))
        except: pass

    if detailed and edges: return None, gpd.GeoDataFrame(geometry=edges, crs="EPSG:4326")
    if all_pts: 
        # 這裡做簡化
        return gpd.GeoSeries(all_pts).buffer(0.0015).union_all().simplify(0.0001), None
    return None, None

# --- 4. UI ---
if 'marker' not in st.session_state: st.session_state['marker'] = [25.0418, 121.5436]
if 'res' not in st.session_state: st.session_state['res'] = {}
if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False

defaults = {'year': '2025', 'limit': 30, 'wait_cost': 5, 'm_private': False, 'm_peak': True, 'm_rail': True, 'm_bike': False, 'm_walk': True, 'is_detailed': False}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

st.title("Taipei Isochrone Map (Safe Mode)")

rs = RailSystem(stations_df, st.session_state['year'])

if st.session_state['analyzed'] and not st.session_state['res']:
    with st.spinner("Calculating..."):
        res = {}
        modes = {'private': st.session_state['m_private'], 'rail': st.session_state['m_rail'], 'walk': st.session_state['m_walk']}
        for m, on in modes.items():
            if on:
                p, e = compute(st.session_state['marker'], m, st.session_state['limit'], rs, st.session_state['is_detailed'], st.session_state['wait_cost'])
                if p or e: res[m] = {'p': p, 'e': e}
        st.session_state['res'] = res

# --- 5. 地圖 (安全模式：移除 LayerControl 和 Icon) ---
# 建立基本地圖，不指定 tiles，使用預設值以避免序列化問題
m = folium.Map(location=st.session_state['marker'], zoom_start=13)

# 畫軌道 (純線條，安全)
for l in rs.lines:
    folium.PolyLine(l['coords'], color=l['color'], weight=2).add_to(m)

# 畫結果
colors = {'private': 'red', 'rail': 'blue', 'walk': 'green'}
if st.session_state['res']:
    for k, v in st.session_state['res'].items():
        c = colors.get(k, 'gray')
        # 畫多邊形
        if v['p']:
            poly = v['p']
            geoms = list(poly.geoms) if poly.geom_type == 'MultiPolygon' else [poly]
            for p in geoms:
                locs = [(y, x) for x, y in p.exterior.coords]
                holes = [[(y, x) for x, y in h.coords] for h in p.interiors]
                # 這裡只傳基本參數，絕對不要傳 lambda 或 style_function
                folium.Polygon(locations=locs, holes=holes, color=c, fill_color=c, fill_opacity=0.3, weight=0).add_to(m)
        
        # 畫細節線條
        if v['e']:
            for _, row in v['e'].iterrows():
                 # 簡化：只支援 LineString，避免複雜物件
                 if row.geometry.geom_type == 'LineString':
                     coords = [(y, x) for x, y in row.geometry.coords]
                     folium.PolyLine(coords, color=c, weight=1).add_to(m)

# 標記 (移除 Icon 參數，使用預設藍色圖釘)
folium.Marker(st.session_state['marker']).add_to(m)

# ⚠️ 注意：移除了 folium.LayerControl()，因為它經常導致序列化崩潰

# 渲染地圖
try:
    map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked"])
except Exception as e:
    st.error(f"Map Error: {e}")
    map_data = None

# --- 控制項 ---
if not st.session_state['analyzed'] and map_data and map_data.get('last_clicked'):
    lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
    if geodesic((lat, lon), st.session_state['marker']).meters > 10:
        st.session_state['marker'] = [lat, lon]
        st.rerun()

with st.expander("Settings", expanded=True):
    st.button("Start Analysis", type="primary", on_click=lambda: st.session_state.update({'analyzed': True}))
    st.button("Reset", on_click=lambda: st.session_state.update({'analyzed': False, 'res': {}}))
    st.toggle("Rail", key='m_rail')
    st.toggle("Private", key='m_private')
    st.toggle("Detailed", key='is_detailed')
    st.slider("Time", 10, 60, key='limit')
