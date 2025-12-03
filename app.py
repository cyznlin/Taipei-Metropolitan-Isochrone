import streamlit as st
import networkx as nx
import pickle
import gzip
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString, MultiPoint
from shapely.ops import unary_union
import geopandas as gpd
from huggingface_hub import hf_hub_download
from geopy.distance import geodesic
from jinja2 import Template
import pandas as pd
import os

st.set_page_config(page_title="Taipei Metropolitan Area Isochrone Map", layout="wide")

DATA_REPO_ID = "你的帳號/north-taiwan-map-data"  # 👈 請修改這裡
DATA_FILENAME = "north_taiwan.pkl.gz"
CSV_FILENAME = os.path.join("stations_master.csv")


# --- 1. 資料載入 (防崩潰保護) ---
@st.cache_resource(show_spinner="正在從雲端下載地圖資料...")
def load_core_data():
    G_drive, G_walk, stations = None, None, None
    
    try:
        # 1. 自動下載並取得檔案路徑 (HF 會自動快取，不會每次都重抓，速度很快)
        print(f"📥 正在從 {DATA_REPO_ID} 下載地圖...")
        local_path = hf_hub_download(
            repo_id=DATA_REPO_ID,
            filename=DATA_FILENAME,
            repo_type="model"  # 指定是 Model 倉庫
        )
        print(f"✅ 下載完成，路徑: {local_path}")

        # 2. 讀取壓縮檔
        print("🚀 載入地圖結構中...")
        with gzip.open(local_path, "rb") as f:
            G_raw = pickle.load(f)

        # 3. 處理座標 (維持原本邏輯)
        for n, d in G_raw.nodes(data=True):
            d['x'] = float(d.get('x', 0))
            d['y'] = float(d.get('y', 0))

            # 建立 G_drive
            G_drive = G_raw.to_undirected()
            for u, v, k, d in G_drive.edges(keys=True, data=True):
                length = float(d.get('length', 50))

                # 1. 取得原始速限
                raw_speed = float(d.get('speed_kph', 30))
                if raw_speed <= 0: raw_speed = 30.0

                # 2. 依照您的規則設定速度參數 (km/h)
                is_hwy = raw_speed >= 80

                if is_hwy:
                    speed_normal = 80.0
                    speed_peak = 35.0
                else:
                    speed_normal = 30.0
                    speed_peak = 15.0

                # 3. 計算時間權重 (分鐘)
                # 修正點：這裡的 Key 必須跟 UI 的 mode 對應
                d['time_private'] = length / (speed_normal * 1000 / 60)
                d['time_private_peak'] = length / (
                            speed_peak * 1000 / 60)  # 原本是 'time_peak'，修正為 'time_private_peak'

                # 4. 腳踏車
                d['time_bike'] = length / (10.0 * 1000 / 60)

            # 建立 G_walk
                # 建立 G_walk (修正版：雙向通行，但嚴格排除高速公路)
            G_walk = G_raw.to_undirected()
            # 先收集需要移除的高速公路邊 (避免在迭代中修改字典)
            remove_edges = []
            for u, v, k, d in G_walk.edges(keys=True, data=True):
                raw_speed = float(d.get('speed_kph', 30))

                # 邏輯：如果該路段原本速限 >= 80 (快速道路/高速公路)，行人禁行 -> 移除
                if raw_speed >= 80:
                    remove_edges.append((u, v, k))
                else:
                    # 其餘路段 (平面道路)，一律以 4 km/h 計算
                    length_walk = float(d.get('length', 50))
                    d['time_walk'] = length_walk / (4.0 * 1000 / 60)

            # 執行移除
            G_walk.remove_edges_from(remove_edges)

    except Exception as e:
        st.error(f"路網錯誤: {e}")

    # B. 載入 CSV
    if os.path.exists(CSV_FILENAME):
        try:
            stations = pd.read_csv(CSV_FILENAME)
            stations['unique_id'] = stations.apply(lambda row: f"{row['name']}_{row['line_id']}", axis=1)
            stations['node_id'] = stations['unique_id'].apply(lambda x: f"STATION_{x}")
        except Exception as e:
            st.error(f"地圖載入失敗: {e}")
            return None, None, None

    return G_drive, G_walk, stations


G_drive, G_walk, stations_df = load_core_data()

# [關鍵修正] 若資料未載入，直接停止，避免後面 RailSystem 報錯
if G_drive is None or stations_df is None:
    st.error("❌ 系統資料缺失！請確認 `build_database.py` 與 `prepare_map.py` 已正確執行且檔案存在。")
    st.stop()


# --- 2. 輔助與邏輯 ---
def get_nearest_node(G, point):
    t_lat, t_lon = point
    best, min_d = None, 100.0
    for n, d in G.nodes(data=True):
        if 'y' not in d: continue
        dy, dx = d['y'] - t_lat, d['x'] - t_lon
        if dy * dy + dx * dx < min_d:
            min_d, best = dy * dy + dx * dx, n
    return best

class RailSystem:
    def __init__(self, df, year):
        self.stations = {}
        self.node_map = {}
        self.lines = []

        # [修正] 這裡加入轉型：將 UI 傳入的字串 (例如 '2025') 轉為整數
        target_year = int(year)

        valid = ['Operating']
        # [修正] 使用轉型後的整數進行比較
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

            # 1. 建立站間連接 (同路線)
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
                    # 時間 = 距離 / 速度 + 停靠損耗(0.5分)
                    w = dist * (60 / spd) + 0.5
                    self.rail_G.add_edge(u, v, weight=w)

            # 2. 建立轉乘連接 (跨路線)
            uids = list(self.stations.keys())
            # 為了效能，這裡可以用 spatial index，但目前 station 數量少，雙迴圈尚可
            for i in range(len(uids)):
                for j in range(i + 1, len(uids)):
                    u, v = uids[i], uids[j]

                    # 相同路線 ID 的前綴相同 (例如 O 和 O_Luzhou 視為不同線，可以進入判斷)
                    # 但如果完全同名 (e.g. O_Luzhou, O_Luzhou) 前面 groupby 處理過了
                    line_u = u.split('_')[-1]  # 假設 unique_id 是 Name_LineID
                    line_v = v.split('_')[-1]
                    if line_u == line_v: continue

                    dist = geodesic(self.stations[u], self.stations[v]).meters

                    # [邏輯修復]
                    # 情況 A: 距離 < 80m -> 視為同一站體 (平行轉乘/共構) -> 0 分鐘
                    # 這能解決「大橋頭(O)」與「大橋頭(O_Luzhou)」座標極近但沒連起來的問題
                    if dist < 80:
                        w = 0.5  # 給一點點換月台成本，避免演算法過度樂觀，若是平行轉乘可設 0

                        # 特例：大橋頭、八堵、北投(紅線轉新北投) 這種同月台或極短轉乘
                        name_u = u.split('_')[0]
                        name_v = v.split('_')[0]
                        if name_u == name_v: w = 0.0

                        self.rail_G.add_edge(u, v, weight=w)

                    # 情況 B: 距離 < 450m -> 站外轉乘/長通道 -> 加上懲罰
                    elif dist < 450:
                        w = 5.0  # 基礎轉乘時間

                        # 特殊轉乘加權
                        if "A1" in u and "台北" in v:
                            w = 12.0  # 北車機捷很遠
                        elif "BR" in u or "Y" in u:
                            w = 7.0  # 文湖/環狀線 轉乘通常久

                        self.rail_G.add_edge(u, v, weight=w)

                        # 畫出轉乘連線
                        self.lines.append({
                            "coords": [self.stations[u], self.stations[v]],
                            "color": "#666", "dash": "2, 2", "weight": 1
                        })

    def get_sources(self, start, limit, wait_time=0):
        entry = []
        # 曲折係數 (Tortuosity Factor): 城市道路非直線，通常乘以 1.35 估算實際步距
        detour_factor = 1.35
        walk_speed_kph = 4.0

        for uid, pos in self.stations.items():
            # 先用直線篩選 (稍微放寬到 2km)
            d_straight = geodesic(start, pos).meters
            if d_straight > 2000: continue

            # [真實化] 估算實際步行距離
            d_real = d_straight * detour_factor

            # 時間 = 走路時間 + 進站等車固定成本 (wait_time)
            t_walk = d_real / (walk_speed_kph * 1000 / 60)
            t_total = t_walk + wait_time

            if t_total < limit: entry.append((uid, t_total))

        if not entry: return []

        temp_G = self.rail_G.copy()
        temp_G.add_node("S")
        for u, t in entry: temp_G.add_edge("S", u, weight=t)

        # ... (前段代碼不變)
        paths = nx.single_source_dijkstra_path_length(temp_G, "S", cutoff=limit)
        res = []

        # 定義出站成本 (您可以設為固定值 3.0，或是沿用 wait_time)
        exit_cost = 3.0

        for uid, cost in paths.items():
            if uid == "S": continue

            # 計算列車到站時的剩餘時間
            rem_arrival = limit - cost

            # 修正邏輯：
            # 1. 剩餘時間必須大於出站成本 (否則出不了站)
            if rem_arrival > exit_cost:

                # 2. 【關鍵修正】實際可走路的時間 = 到站剩餘時間 - 出站成本
                rem_walk = rem_arrival - exit_cost

                nid = self.node_map[uid]
                if nid in G_walk.nodes:
                    res.append((nid, rem_walk))  # 這裡要傳出扣掉後的 rem_walk

        return res

def compute(start, mode, limit, rs, detailed=False, wait_penalty=0):
    polys, edges = [], []
    if mode in ['rail', 'walk']:
        G = G_walk
        metric = 'time_walk'
    else:
        G = G_drive
        metric = f'time_{mode}'  # 注意：請確保這裡產生的 key (如 time_private_peak) 與 G_drive 裡的 key 一致！

    targets = []
    if mode == 'rail':
        # 將等待時間傳入
        targets = rs.get_sources(start, limit, wait_penalty)
    else:
        sn = get_nearest_node(G, start)
        if sn: targets = [(sn, limit)]

    if not targets: return None, None

    all_pts = []

    for node, rem in targets:
        try:
            sub = nx.ego_graph(G, node, radius=rem, distance=metric)

            if detailed:
                # --- 詳細模式邏輯 (保持不變) ---
                lines = []
                for u, v, d in sub.edges(data=True):
                    if 'geometry' in d and isinstance(d['geometry'], LineString):
                        lines.append(d['geometry'])
                    else:
                        lines.append(LineString(
                            [Point(G.nodes[u]['x'], G.nodes[u]['y']), Point(G.nodes[v]['x'], G.nodes[v]['y'])]))
                if lines:
                    merged = unary_union(lines)
                    polys.append(merged.buffer(0.00025, resolution=2))
                    edges.append(gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326"))
            else:
                # --- 快速模式優化 (修改處) ---
                # 收集所有可到達的節點座標
                pts = [Point(G.nodes[n]['x'], G.nodes[n]['y']) for n in sub.nodes]
                if pts: all_pts.extend(pts)

        except Exception as e:
            # print(e) # 除錯用
            pass


    if detailed:
        return unary_union(polys) if polys else None, edges
    else:
        # 針對不同模式給予不同的 Buffer 大小
        if all_pts:
            # 預設半徑 (走路用)
            radius = 0.0015

            # 如果是開車 (private / private_peak)，稍微加大半徑以填補節點間隙
            # 開車速度快，節點在時間軸上的跨度大，需要較大的筆刷來塗滿
            if 'private' in mode:
                radius = 0.0030  # 加倍至約 300m

            return gpd.GeoSeries(all_pts).buffer(radius).union_all(), None

    return None, None


# --- 5. UI 與 流程控制 (Mobile Optimized V3) ---
if 'marker' not in st.session_state: st.session_state['marker'] = [25.0418, 121.5436]
if 'res' not in st.session_state: st.session_state['res'] = {}
if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False

# 初始化 Session State (確保 dynamic label 取得到值)
defaults = {
    'year': '2025',  # 必須是字串，因為 Select Slider 的 options 是字串列表
    'limit': 30,
    'wait_cost': 5,
    'm_private': False,
    'm_peak': True,
    'm_rail': True,
    'm_bike': False,
    'm_walk': True,
    'is_detailed': False
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

st.title("北北基桃等時圈 Taipei Metropolitan Area Isochrone Map")

# [新功能] 操作指引 (置頂)
st.info("💡 **操作順序：** 1. 點擊地圖選擇地點 → 2. 設定下方參數 (年份/模式) → 3. 點擊「開始分析」")

# --- 4. 運算邏輯 (Map 需要先有結果才能畫) ---
# 根據 Session State 中的 'year' 初始化 RailSystem
rs = RailSystem(stations_df, st.session_state['year'])

# 根據 Session State 中的 'modes' 構建字典
current_modes = {
    'private': st.session_state['m_private'],
    'private_peak': st.session_state['m_peak'],
    'rail': st.session_state['m_rail'],
    'bike': st.session_state['m_bike'],
    'walk': st.session_state['m_walk']
}

# 執行計算 (如果 analyzed 為 True)
if st.session_state['analyzed'] and not st.session_state['res']:
    with st.spinner("正在計算可及範圍..."):
        res = {}
        for m, on in current_modes.items():
            if on:
                p, e = compute(
                    st.session_state['marker'], m,
                    st.session_state['limit'], rs,
                    st.session_state['is_detailed'],
                    st.session_state['wait_cost']
                )
                if p: res[m] = {'p': p, 'e': e}
        st.session_state['res'] = res

# --- 5. 繪製地圖 (置頂) ---
m = folium.Map(location=st.session_state['marker'], zoom_start=13, tiles="CartoDB positron")

# 軌道底圖
fg_rail = folium.FeatureGroup(name="軌道系統", show=True)
for l in rs.lines:
    folium.PolyLine(l['coords'], color=l['color'], weight=l.get('weight', 2), dash_array=l.get('dash')).add_to(fg_rail)
for uid, pos in rs.stations.items():
    folium.CircleMarker(pos, radius=1.5, color='black').add_to(fg_rail)
fg_rail.add_to(m)

# 結果圖層
colors = {'private': '#E74C3C', 'private_peak': '#922B21', 'rail': '#0070BD', 'bike': '#F39C12', 'walk': '#2ECC71'}
order = ['private', 'private_peak', 'bike', 'rail', 'walk']

if st.session_state['res']:
    for k in order:
        if k in st.session_state['res']:
            v = st.session_state['res'][k]
            fg = folium.FeatureGroup(name=k)
            if st.session_state['is_detailed'] and v['e']:
                for gdf in v['e']:
                    folium.GeoJson(gdf, style_function=lambda x, c=colors[k]: {'color': c, 'weight': 1.2,
                                                                               'opacity': 0.8}).add_to(fg)
            folium.GeoJson(v['p'], style_function=lambda x, c=colors[k]: {'fillColor': c, 'color': c, 'weight': 0,
                                                                          'fillOpacity': 0.3}).add_to(fg)
            fg.add_to(m)

# 懸浮統計
stats = ""
for k in order:
    if k in st.session_state['res']:
        a = gpd.GeoSeries([st.session_state['res'][k]['p']], crs="EPSG:4326").to_crs(epsg=3857).area[0] / 1e6
        label = {'private': '私有', 'private_peak': '私有尖峰', 'rail': '運輸', 'bike': '單車', 'walk': '步行'}[k]
        stats += f"<div style='color:{colors[k]}; font-size: 14px;'><b>{label}</b>: {a:.1f} km²</div>"
if stats:
    macro = folium.MacroElement()
    macro._template = Template(f"""
        {{% macro html(this, kwargs) %}}
        <div style="position: fixed; top: 10px; right: 10px; width: 140px; background: rgba(255,255,255,0.9); padding: 8px; border-radius: 5px; z-index:9999; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            <b>範圍 ({st.session_state['limit']}分)</b><hr style="margin:5px 0;">{stats}
        </div>
        {{% endmacro %}}
    """)
    m.get_root().add_child(macro)

folium.LayerControl().add_to(m)
folium.Marker(st.session_state['marker'], icon=folium.Icon(color="black", icon="home")).add_to(m)

map_data = st_folium(m, width=None, height=500)  # 手機版地圖高度適中

# 點擊更新 (僅未分析時)
if not st.session_state['analyzed'] and map_data['last_clicked']:
    lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
    if geodesic((lat, lon), st.session_state['marker']).meters > 10:
        st.session_state['marker'] = [lat, lon]
        st.rerun()

# --- 6. 設定面板 (置底) ---
# 動態摘要文字
selected_labels = []
if st.session_state['m_private']: selected_labels.append("私有")
if st.session_state['m_peak']: selected_labels.append("尖峰")
if st.session_state['m_rail']: selected_labels.append("軌道")
if st.session_state['m_bike']: selected_labels.append("單車")
if st.session_state['m_walk']: selected_labels.append("步行")
mode_summary = "/".join(selected_labels) if selected_labels else "未選"

status_txt = "✅ 完成" if st.session_state['analyzed'] else "⚙️ 設定"
detail_txt = " | 🐢精細" if st.session_state['is_detailed'] else ""

# [修正] 總結加入年份顯示
expander_label = f"{status_txt} ({st.session_state['year']}年 | {st.session_state['limit']}分 | {mode_summary}{detail_txt})"
show_expander = not st.session_state['analyzed']

with st.expander(expander_label, expanded=show_expander):
    # 核心參數
    c1, c2, c3 = st.columns(3)
    with c1:
        # [修正] 使用字串列表作為 options，確保每個年份都被標示出來
        st.select_slider("📅 年份", options=['2025', '2028', '2031'], key='year')
    with c2: st.slider("⏱️ 時間", 10, 60, key='limit')
    with c3: st.slider("⏳ 進站", 0, 15, key='wait_cost', help="轉乘/進出站成本")

    st.write("---")

    # 交通工具
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1: st.toggle("🚗 私有運具（35/80 kph）", key='m_private')
    with r1_c2: st.toggle("🚗 私有運具尖峰（15/30 kph）", key='m_peak')
    with r1_c3: st.toggle("🚆 軌道運輸＋步行", key='m_rail')

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1: st.toggle("🚲 單車", key='m_bike')
    with r2_c2: st.toggle("🚶 純步行", key='m_walk')

    st.write("---")

    # 渲染細節
    d_col1, d_col2 = st.columns([1, 2])
    with d_col1: st.toggle("🐢 路徑細節", key='is_detailed')
    with d_col2: st.caption("⚠️ 勾選後將繪製真實路徑幾何，運算與繪圖速度較慢。")

# 動作按鈕 (置底)
btn_c1, btn_c2 = st.columns([2, 1])
with btn_c1:
    if not st.session_state['analyzed']:
        if st.button("🚀 開始分析", type="primary", use_container_width=True):
            st.session_state['analyzed'] = True
            st.rerun()
    else:
        st.button(f"✅ 正在運算 {st.session_state['limit']} 分鐘可達範圍", disabled=True, use_container_width=True)

with btn_c2:
    if st.button("🔄 重置", type="secondary", use_container_width=True):
        st.session_state['analyzed'] = False
        st.session_state['res'] = {}
        st.rerun()
