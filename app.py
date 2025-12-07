import streamlit as st
import networkx as nx
import pickle
import gzip
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely import wkt
import geopandas as gpd
from geopy.distance import geodesic
import pandas as pd
import os
from huggingface_hub import hf_hub_download

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Debug Mode Map (Fixed)", layout="wide")

# --- 設定區 ---
DATA_REPO_ID = "ZnCYLin/north-taiwan-map-data" 
DATA_FILENAME = "north_taiwan_ready.pkl.gz"
CSV_FILENAME = "stations_master.csv"

# --- 2. 資料載入 ---
@st.cache_resource(show_spinner="正在下載地圖...")
def load_core_data():
    G_drive, G_walk, stations = None, None, None
    try:
        print(f"📥 下載 {DATA_FILENAME}...")
        local_path = hf_hub_download(repo_id=DATA_REPO_ID, filename=DATA_FILENAME, repo_type="model")
        
        print("🚀 讀取 Pickle...")
        with gzip.open(local_path, "rb") as f:
            G_raw = pickle.load(f)

        G_drive = G_raw 
        
        # ⚠️ [關鍵修復] NetworkX 的 subgraph_view 對於 MultiGraph 只傳 (u, v, k)
        # 我們必須自己從 G_raw 裡面去查屬性，不能期待它傳 d 進來
        def filter_walk(u, v, k):
            # 修正寫法：自己去查資料
            edge_data = G_raw[u][v][k] 
            return edge_data.get('time_walk', 999999) < 1000
            
        G_walk = nx.subgraph_view(G_raw, filter_edge=filter_walk)

    except Exception as e:
        st.error(f"❌ 資料載入嚴重錯誤: {e}")
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
    st.error("❌ 無法載入地圖資料，請檢查 Repo ID")
    st.stop()

# --- 3. 核心邏輯 (Debug 版：移除 try-except) ---
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
        self.stations = {}
        self.node_map = {}
        self.lines = []
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
        # 簡單建立路網
        for lid, grp in active.groupby('line_id'):
            grp = grp.sort_values('sequence')
            ids = grp['unique_id'].tolist()
            # 存線條 (只存座標)
            self.lines.append({"coords": [self.stations[i] for i in ids], "color": colors.get(lid, "gray")})

            spd = 55.0 if lid.startswith(('A', 'TRA')) else 35.0
            for i in range(len(ids) - 1):
                u, v = ids[i], ids[i + 1]
                dist = geodesic(self.stations[u], self.stations[v]).km
                self.rail_G.add_edge(u, v, weight=dist*(60/spd)+0.5)

    def get_sources(self, start, limit, wait_time=0):
        entry = []
        detour_factor = 1.35
        for uid, pos in self.stations.items():
            d_straight = geodesic(start, pos).meters
            if d_straight > 2000: continue
            t = (d_straight * detour_factor) / (4.0 * 1000 / 60) + wait_time
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
    if mode in ['rail', 'walk']:
        G = G_walk
        metric = 'time_walk'
    else:
        G = G_drive
        metric = f'time_{mode}'
    
    targets = []
    if mode == 'rail':
        targets = rs.get_sources(start, limit, wait_penalty)
    else:
        sn = get_nearest_node(G, start)
        if sn: targets = [(sn, limit)]
        else: st.warning(f"⚠️ 找不到最近的節點！模式: {mode}")

    if not targets: return None, None

    all_pts = []
    
    # Debug: 移除 try-except
    for node, rem in targets:
        # 這裡現在應該不會報錯了，因為 filter_walk 修好了
        sub = nx.ego_graph(G, node, radius=rem, distance=metric)
        pts = [Point(G.nodes[n]['x'], G.nodes[n]['y']) for n in sub.nodes]
        if pts: all_pts.extend(pts)

    if all_pts:
        radius = 0.0030 if 'private' in mode else 0.0015
        return gpd.GeoSeries(all_pts).buffer(radius).union_all().simplify(0.0001), None
        
    return None, None

# --- 4. UI ---
if 'marker' not in st.session_state: st.session_state['marker'] = [25.0418, 121.5436]
if 'res' not in st.session_state: st.session_state['res'] = {}
if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False

# Debug 設定：預設全開
defaults = {'year': '2025', 'limit': 30, 'wait_cost': 5, 
            'm_private': True, 'm_peak': True, 'm_rail': True, 
            'm_bike': True, 'm_walk': True, 'is_detailed': False}

for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

st.title("🚧 Debug Mode (Fix): 應該要全綠了！")

rs = RailSystem(stations_df, st.session_state['year'])

current_modes = {
    'private': st.session_state['m_private'],
    'private_peak': st.session_state['m_peak'],
    'rail': st.session_state['m_rail'],
    'bike': st.session_state['m_bike'],
    'walk': st.session_state['m_walk']
}

if st.button("🚀 開始測試 (Run Test)", type="primary"):
    st.session_state['analyzed'] = True
    st.session_state['res'] = {}
    
    with st.spinner("測試中..."):
        res = {}
        for m_key, on in current_modes.items():
            if on:
                st.write(f"▶️ 正在計算: **{m_key}** ...")
                try:
                    p, e = compute(st.session_state['marker'], m_key, st.session_state['limit'], rs)
                    if p: 
                        res[m_key] = {'p': p, 'e': e}
                        st.success(f"✅ {m_key} 計算成功！")
                    else:
                        st.warning(f"⚠️ {m_key} 回傳了空結果 (None)")
                except Exception as e:
                    st.error(f"❌ {m_key} 發生錯誤: {e}")
                    st.exception(e)
        st.session_state['res'] = res

# --- 5. 地圖 ---
m = folium.Map(location=st.session_state['marker'], zoom_start=13)
colors = {'private': '#E74C3C', 'private_peak': '#922B21', 'rail': '#0070BD', 'bike': '#F39C12', 'walk': '#2ECC71'}

if st.session_state['res']:
    for k, v in st.session_state['res'].items():
        if k not in colors: continue
        if v['p']:
            poly_geom = v['p']
            geoms = list(poly_geom.geoms) if isinstance(poly_geom, MultiPolygon) else [poly_geom] if isinstance(poly_geom, Polygon) else []
            for p in geoms:
                locations = [(y, x) for x, y in p.exterior.coords]
                holes = [[(y, x) for x, y in h.coords] for h in p.interiors]
                folium.Polygon(locations=locations, holes=holes, color=colors[k], fill_color=colors[k], fill_opacity=0.3, weight=0).add_to(m)

folium.Marker(st.session_state['marker']).add_to(m)

try:
    map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked"])
except Exception as e:
    st.error(f"Map Error: {e}")
    map_data = None

if map_data and map_data.get('last_clicked'):
    lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
    if geodesic((lat, lon), st.session_state['marker']).meters > 10:
        st.session_state['marker'] = [lat, lon]
        st.rerun()

# 顯示開關
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.toggle("Private", key='m_private')
with c2: st.toggle("Peak", key='m_peak')
with c3: st.toggle("Rail", key='m_rail')
with c4: st.toggle("Bike", key='m_bike')
with c5: st.toggle("Walk", key='m_walk')
