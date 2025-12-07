import streamlit as st
import networkx as nx
import pickle
import gzip
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import geopandas as gpd
from geopy.distance import geodesic
import pandas as pd
import os
from huggingface_hub import hf_hub_download

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Taipei Metropolitan Area Isochrone Map", layout="wide")

# --- 設定區 (請確認 Repo ID) ---
DATA_REPO_ID = "ZnCYLin/north-taiwan-map-data" 
DATA_FILENAME = "north_taiwan_ready.pkl.gz"
CSV_FILENAME = "stations_master.csv"

# --- 2. 資料載入 (加入自動修復邏輯) ---
@st.cache_resource(show_spinner="正在從雲端下載並修復地圖資料...")
def load_core_data():
    G_drive, G_walk, stations = None, None, None
    try:
        # 1. 下載
        print(f"📥 正在下載 {DATA_FILENAME}...")
        local_path = hf_hub_download(repo_id=DATA_REPO_ID, filename=DATA_FILENAME, repo_type="model")
        
        # 2. 讀取
        print("🚀 載入地圖結構中...")
        with gzip.open(local_path, "rb") as f:
            G_raw = pickle.load(f)

        # 3. 🛡️ 自動檢測與修復 (關鍵修復)
        # 檢查第一條邊，看是否有必要的權重屬性
        test_edge = next(iter(G_raw.edges(data=True)))
        attrs = test_edge[2] # 取得屬性字典
        
        # 如果缺資料 (例如缺 time_private 或 time_walk)，就當場補算
        needed_keys = ['time_private', 'time_private_peak', 'time_bike', 'time_walk']
        is_missing_data = any(k not in attrs for k in needed_keys)

        if is_missing_data:
            print("⚠️ 偵測到地圖檔缺少部分權重資料，正在自動修復 (Calculation Repair)...")
            
            # 確保是 MultiDiGraph
            if not isinstance(G_raw, nx.MultiDiGraph):
                G_raw = nx.MultiDiGraph(G_raw)

            for u, v, k, d in G_raw.edges(keys=True, data=True):
                length = float(d.get('length', 50))
                raw_speed = float(d.get('speed_kph', 30))
                if raw_speed <= 0: raw_speed = 30.0

                is_hwy = raw_speed >= 80
                
                # 補算開車 (一般)
                speed_normal = 80.0 if is_hwy else 30.0
                d['time_private'] = length / (speed_normal * 1000 / 60)

                # 補算開車 (尖峰)
                speed_peak = 35.0 if is_hwy else 15.0
                d['time_private_peak'] = length / (speed_peak * 1000 / 60)

                # 補算單車 (假設均速 10km/h)
                d['time_bike'] = length / (10.0 * 1000 / 60)

                # 補算步行 (若為高速公路則設為無限大)
                if is_hwy:
                    d['time_walk'] = 999999
                else:
                    d['time_walk'] = length / (4.0 * 1000 / 60)

            # 確保座標格式正確
            for n, d in G_raw.nodes(data=True):
                d['x'] = float(d.get('x', 0))
                d['y'] = float(d.get('y', 0))
                
            print("✅ 自動修復完成！")
        else:
            print("✅ 地圖資料完整，無需修復。")

        # 4. 分拆圖層
        G_drive = G_raw 
        
        # 步行圖層過濾 (必須確保 time_walk 存在，剛剛的修復保證了這一點)
        def filter_walk(u, v, k, d):
            return d.get('time_walk', 999999) < 1000
        G_walk = nx.subgraph_view(G_raw, filter_edge=filter_walk)

    except Exception as e:
        st.error(f"地圖載入失敗: {e}")
        return None, None, None

    # B. 載入 CSV
    if os.path.exists(CSV_FILENAME):
        try:
            stations = pd.read_csv(CSV_FILENAME)
            stations['unique_id'] = stations.apply(lambda row: f"{row['name']}_{row['line_id']}", axis=1)
            stations['node_id'] = stations['unique_id'].apply(lambda x: f"STATION_{x}")
        except: pass

    return G_drive, G_walk, stations

G_drive, G_walk, stations_df = load_core_data()

if G_drive is None:
    st.error("❌ 系統資料缺失！請確認 Hugging Face Model Repo 設定正確。")
    st.stop()

# --- 3. 核心類別 (RailSystem) ---
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

        target_year = int(year)
        valid = ['Operating']
        if target_year >= 2028: valid.append('Construction')
        if target_year >= 2031: valid.append('Planning')

        if df is not None:
            active = df[df['status'].isin(valid)]
            for _, r in active.iterrows():
                uid = r['unique_id']
                self.stations[uid] = (r['lat'], r['lon'])
                self.node_map[uid] = r['node_id']

            colors = {"BL": "#0070BD", "R": "#E3002C", "R_NewBeitou": "#E3002C", "R_East": "#E3002C",
                      "G": "#008659", "G_Xiaobitan": "#008659", "O": "#F8B61C",
                      "O_Luzhou": "#F8B61C", "BR": "#C48C31", "Y": "#FDD935",
                      "Y_North": "#FDD935", "Y_South": "#FDD935", "A": "#8246AF",
                      "LG_1": "#D1E231", "LG_2": "#D1E231", "LB": "#6C9ED3",
                      "G_Taoyuan": "#008659", "TRA_West": "#333333", "TRA_East": "#333333"}

            self.rail_G = nx.Graph()

            for lid, grp in active.groupby('line_id'):
                grp = grp.sort_values('sequence')
                ids = grp['unique_id'].tolist()
                coords = [self.stations[i] for i in ids]
                is_future = any(s != 'Operating' for s in grp['status'])
                dash = "5, 5" if is_future else None
                self.lines.append({"coords": coords, "color": colors.get(lid, "gray"), "dash": dash, "weight": 3})

                spd = 55.0 if lid.startswith(('A', 'TRA')) else 35.0
                for i in range(len(ids) - 1):
                    u, v = ids[i], ids[i + 1]
                    dist = geodesic(self.stations[u], self.stations[v]).km
                    w = dist * (60 / spd) + 0.5
                    self.rail_G.add_edge(u, v, weight=w)

            uids = list(self.stations.keys())
            for i in range(len(uids)):
                for j in range(i + 1, len(uids)):
                    u, v = uids[i], uids[j]
                    line_u = u.split('_')[-1]
                    line_v = v.split('_')[-1]
                    if line_u == line_v: continue

                    dist = geodesic(self.stations[u], self.stations[v]).meters
                    if dist < 80:
                        w = 0.5
                        if u.split('_')[0] == v.split('_')[0]: w = 0.0
                        self.rail_G.add_edge(u, v, weight=w)
                    elif dist < 450:
                        w = 5.0
                        if "A1" in u and "台北" in v: w = 12.0
                        elif "BR" in u or "Y" in u: w = 7.0
                        self.rail_G.add_edge(u, v, weight=w)
                        self.lines.append({"coords": [self.stations[u], self.stations[v]], "color": "#666", "dash": "2, 2", "weight": 1})

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
    G = G_walk if mode in ['rail', 'walk'] else G_drive
    metric = 'time_walk' if mode in ['rail', 'walk'] else f'time_{mode}'
    
    targets = []
    if mode == 'rail':
        targets = rs.get_sources(start, limit, wait_penalty)
    else:
        sn = get_nearest_node(G, start)
        if sn: targets = [(sn, limit)]

    if not targets: return None, None

    all_pts = []
    edges = []
    for node, rem in targets:
        try:
            sub = nx.ego_graph(G, node, radius=rem, distance=metric)
            if detailed:
                lines = []
                for u, v, d in sub.edges(data=True):
                    if 'geometry' in d: lines.append(d['geometry'])
                    else: lines.append(LineString([Point(G.nodes[u]['x'], G.nodes[u]['y']), Point(G.nodes[v]['x'], G.nodes[v]['y'])]))
                if lines: edges.append(gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326"))
            else:
                pts = [Point(G.nodes[n]['x'], G.nodes[n]['y']) for n in sub.nodes]
                if pts: all_pts.extend(pts)
        except: pass

    if detailed: return None, edges
    else:
        if all_pts:
            radius = 0.0030 if 'private' in mode else 0.0015
            return gpd.GeoSeries(all_pts).buffer(radius).union_all().simplify(0.0001), None
    return None, None

# --- 4. UI 邏輯 ---
if 'marker' not in st.session_state: st.session_state['marker'] = [25.0418, 121.5436]
if 'res' not in st.session_state: st.session_state['res'] = {}
if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False

defaults = {'year': '2025', 'limit': 30, 'wait_cost': 5, 'm_private': False, 'm_peak': True, 'm_rail': True, 'm_bike': False, 'm_walk': True, 'is_detailed': False}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

st.title("北北基桃等時圈 Taipei Metropolitan Area Isochrone Map")
st.info("💡 **操作順序：** 1. 點擊地圖選擇地點 → 2. 設定下方參數 → 3. 點擊「開始分析」")

rs = RailSystem(stations_df, st.session_state['year'])

current_modes = {
    'private': st.session_state['m_private'],
    'private_peak': st.session_state['m_peak'],
    'rail': st.session_state['m_rail'],
    'bike': st.session_state['m_bike'],
    'walk': st.session_state['m_walk']
}

if st.session_state['analyzed'] and not st.session_state['res']:
    with st.spinner("正在計算可及範圍..."):
        res = {}
        for m_key, on in current_modes.items():
            if on:
                p, e = compute(st.session_state['marker'], m_key, st.session_state['limit'], rs, st.session_state['is_detailed'], st.session_state['wait_cost'])
                if p or e: res[m_key] = {'p': p, 'e': e}
        st.session_state['res'] = res

# --- 5. 地圖繪製 (安全模式) ---
m = folium.Map(location=st.session_state['marker'], zoom_start=13)

# 1. 畫軌道
for l in rs.lines:
    folium.PolyLine(l['coords'], color=l['color'], weight=l.get('weight', 2), dash_array=l.get('dash')).add_to(m)
for uid, pos in rs.stations.items():
    folium.CircleMarker(pos, radius=1.5, color='black', fill=True).add_to(m)

# 2. 畫結果
colors = {'private': '#E74C3C', 'private_peak': '#922B21', 'rail': '#0070BD', 'bike': '#F39C12', 'walk': '#2ECC71'}
area_stats = {}

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
            
            try:
                area = gpd.GeoSeries([poly_geom], crs="EPSG:4326").to_crs(epsg=3857).area[0] / 1e6
                area_stats[k] = area
            except: pass

        if st.session_state['is_detailed'] and v['e']:
            for gdf in v['e']:
                for _, row in gdf.iterrows():
                    geom = row.geometry
                    lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
                    for line in lines:
                        folium.PolyLine([(y, x) for x, y in line.coords], color=colors[k], weight=1.2, opacity=0.8).add_to(m)

folium.Marker(st.session_state['marker']).add_to(m)

# 3. 顯示統計
if area_stats:
    st.markdown("### 📊 可及範圍統計")
    cols = st.columns(len(area_stats))
    labels = {'private': '私有', 'private_peak': '尖峰', 'rail': '軌道', 'bike': '單車', 'walk': '步行'}
    for idx, (k, val) in enumerate(area_stats.items()):
        if idx < len(cols):
            with cols[idx]:
                st.metric(label=labels.get(k, k), value=f"{val:.1f} km²")

# 4. 渲染
try:
    map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked"])
except Exception as e:
    st.error(f"地圖渲染錯誤: {e}")
    map_data = None

# --- 6. 控制面板 ---
if not st.session_state['analyzed'] and map_data and map_data.get('last_clicked'):
    lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
    if geodesic((lat, lon), st.session_state['marker']).meters > 10:
        st.session_state['marker'] = [lat, lon]
        st.rerun()

status_txt = "✅ 完成" if st.session_state['analyzed'] else "⚙️ 設定"
selected_labels = [k for k, v in current_modes.items() if v]
mode_summary = "/".join(selected_labels) if selected_labels else "未選"
expander_label = f"{status_txt} ({st.session_state['year']}年 | {st.session_state['limit']}分 | {mode_summary})"

with st.expander(expander_label, expanded=not st.session_state['analyzed']):
    c1, c2, c3 = st.columns(3)
    with c1: st.select_slider("📅 年份", options=['2025', '2028', '2031'], key='year')
    with c2: st.slider("⏱️ 時間", 5, 60, key='limit')
    with c3: st.slider("⏳ 進站", 0, 15, key='wait_cost', help="轉乘/進出站成本")
    st.write("---")
    r1, r2, r3 = st.columns(3)
    with r1: st.toggle("🚗 私有運具（35/80 kph）", key='m_private')
    with r2: st.toggle("🚗 私有運具尖峰（15/30 kph）", key='m_peak')
    with r3: st.toggle("🚆 軌道運輸＋步行", key='m_rail')
    r4, r5, r6 = st.columns(3)
    with r4: st.toggle("🚲 單車", key='m_bike')
    with r5: st.toggle("🚶 純步行", key='m_walk')
    with r6: st.toggle("🐢 路徑細節", key='is_detailed')

b1, b2 = st.columns([2, 1])
with b1:
    if not st.session_state['analyzed']:
        if st.button("🚀 開始分析", type="primary", use_container_width=True):
            st.session_state['analyzed'] = True
            st.rerun()
    else: st.button("✅ 分析完成", disabled=True, use_container_width=True)
with b2:
    if st.button("🔄 重置", use_container_width=True):
        st.session_state['analyzed'] = False
        st.session_state['res'] = {}
        st.rerun()
