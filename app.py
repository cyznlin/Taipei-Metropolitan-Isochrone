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
st.set_page_config(page_title="Debug Mode Map", layout="wide")

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

        # 這裡不需自動修復，因為你的檢查腳本說檔案是好的
        G_drive = G_raw 
        
        # 建立步行圖層 (使用 subgraph_view)
        # ⚠️ 注意：這裡可能會因為某些路段缺 time_walk 而導致過濾出錯，我們加強檢查
        def filter_walk(u, v, k, d):
            return d.get('time_walk', 999999) < 1000
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
        
        # 為了 Debug，這裡我們不建立複雜的 rail_G，只保留基本列表
        self.rail_sources = [] 
        # (這裡省略複雜的軌道演算法，專注於解決為什麼私有運具跑不出來的問題)

    # 簡化版：只回傳空，因為重點在修復 Private
    def get_sources(self, start, limit, wait_time=0):
        return []

def compute(start, mode, limit, rs, detailed=False, wait_penalty=0):
    # 1. 決定使用哪張圖
    if mode in ['rail', 'walk']:
        G = G_walk
        metric = 'time_walk'
    else:
        G = G_drive
        metric = f'time_{mode}'
    
    # Debug: 顯示正在使用哪個 Metric
    # st.write(f"🔍 [Debug] 模式: {mode}, 使用權重欄位: {metric}")

    targets = []
    if mode == 'rail':
        targets = [] # 暫時略過 rail
    else:
        sn = get_nearest_node(G, start)
        if sn: 
            targets = [(sn, limit)]
        else:
            st.warning(f"⚠️ 找不到最近的節點！模式: {mode}")

    if not targets: return None, None

    all_pts = []
    
    # ⚠️ 關鍵修改：這裡移除了 try-except，讓錯誤直接爆出來
    for node, rem in targets:
        # 使用 networkx 的 ego_graph
        # 如果這裡報錯 KeyError，代表有些路段缺少了該 metric
        sub = nx.ego_graph(G, node, radius=rem, distance=metric)
        
        pts = [Point(G.nodes[n]['x'], G.nodes[n]['y']) for n in sub.nodes]
        if pts: all_pts.extend(pts)
        
        # Debug: 顯示找到了多少點
        # st.write(f"📊 [Debug] {mode} 找到 {len(pts)} 個可達節點")

    if all_pts:
        radius = 0.0030 if 'private' in mode else 0.0015
        return gpd.GeoSeries(all_pts).buffer(radius).union_all().simplify(0.0001), None
        
    return None, None

# --- 4. UI ---
if 'marker' not in st.session_state: st.session_state['marker'] = [25.0418, 121.5436]
if 'res' not in st.session_state: st.session_state['res'] = {}
if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False

# ⚠️ Debug 設定：預設開啟私有運具，強制檢查
defaults = {'year': '2025', 'limit': 30, 'wait_cost': 5, 
            'm_private': True, 'm_peak': False, 'm_rail': False, 
            'm_bike': False, 'm_walk': True, 'is_detailed': False}

for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

st.title("🚧 Debug Mode: 請按開始分析")
st.write("此模式會顯示詳細錯誤訊息，請觀察畫面是否有紅色報錯。")

rs = RailSystem(stations_df, st.session_state['year'])

# 這裡我們只測試主要的幾個模式
current_modes = {
    'private': st.session_state['m_private'],
    'private_peak': st.session_state['m_peak'],
    'walk': st.session_state['m_walk']
}

if st.button("🚀 開始除錯分析 (Start Debug)", type="primary"):
    st.session_state['analyzed'] = True
    st.session_state['res'] = {} # 清空舊結果
    
    with st.spinner("正在暴力運算..."):
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
                    # 這行會把詳細的 Python 錯誤印出來，非常重要
                    st.exception(e) 
        st.session_state['res'] = res

# --- 5. 地圖 ---
m = folium.Map(location=st.session_state['marker'], zoom_start=13)

colors = {'private': '#E74C3C', 'private_peak': '#922B21', 'walk': '#2ECC71'}

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

# 顯示設定開關 (方便你測試)
c1, c2, c3 = st.columns(3)
with c1: st.toggle("Private (私有)", key='m_private')
with c2: st.toggle("Peak (尖峰)", key='m_peak')
with c3: st.toggle("Walk (步行)", key='m_walk')
