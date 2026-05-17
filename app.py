from flask import Flask, render_template, jsonify, request
import heapq
import numpy as np
import math

app = Flask(__name__)

# ── 위치 데이터 ──────────────────────────────────────────────
LOCATIONS = {
    "A": (37.5846, 126.9680, "국립서울맹학교", "출발"),
    "B": (37.5758, 126.9735, "경복궁역", ""),
    "G": (37.5808, 126.9698, "통인시장", ""),
    "H": (37.5794, 126.9723, "경복궁 영추문", ""),
    "I": (37.5740, 126.9760, "정부서울청사", ""),
    "K": (37.5730, 126.9790, "종로구청", ""),
    "M": (37.5820, 126.9700, "종로장애인복지관", ""),
    "N": (37.5779, 126.9749, "국립고궁박물관", ""),
    "O": (37.5775, 126.9835, "조계사", ""),
    "P": (37.5840, 126.9750, "청와대 앞길", ""),
    "Q": (37.5815, 126.9765, "국립민속박물관", ""),
    "R": (37.5760, 126.9765, "광화문 광장", ""),
    "S": (37.5735, 126.9795, "교보문고", ""),
    "T": (37.5750, 126.9815, "인사동 쌈지길", ""),
    "U": (37.5730, 126.9890, "익선동 한옥마을", ""),
    "E": (37.5765, 126.9850, "안국역", "도착"),
}

# 추상 그래프용 2D 좌표 (캔버스 기반, 픽셀)
GRAPH_POS = {
    "A": (120, 60),
    "M": (220, 80),
    "P": (340, 60),
    "Q": (460, 90),
    "G": (150, 160),
    "H": (270, 160),
    "N": (390, 160),
    "B": (130, 280),
    "R": (260, 270),
    "T": (390, 270),
    "O": (510, 190),
    "I": (160, 380),
    "S": (290, 380),
    "K": (200, 460),
    "U": (430, 400),
    "E": (530, 360),
}


def haversine(p1, p2):
    R = 6371
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_edges(active_nodes, k=3):
    """각 노드에서 가장 가까운 k개 노드와 연결"""
    edges = {}
    edge_list = []
    for n1 in active_nodes:
        dists = []
        for n2 in active_nodes:
            if n1 == n2:
                continue
            d_km = haversine(LOCATIONS[n1][:2], LOCATIONS[n2][:2])
            time_val = max(1, int(d_km * 15) + 2)
            dists.append((time_val, n2))
        dists.sort()
        for i in range(min(k, len(dists))):
            u, v = n1, dists[i][1]
            key = tuple(sorted((u, v)))
            if key not in edges:
                edges[key] = dists[i][0]
    for (u, v), w in edges.items():
        edge_list.append({"u": u, "v": v, "w": w})
    return edge_list


def dijkstra(active_nodes, edge_list):
    graph = {n: {} for n in active_nodes}
    for e in edge_list:
        graph[e["u"]][e["v"]] = e["w"]
        graph[e["v"]][e["u"]] = e["w"]

    dist = {n: float("inf") for n in active_nodes}
    dist["A"] = 0
    prev = {n: None for n in active_nodes}
    visited = set()
    queue = [(0, "A")]
    steps = []

    # 초기 상태
    steps.append({
        "phase": "init",
        "confirmed": None,
        "dist": {k: (v if v != float("inf") else None) for k, v in dist.items()},
        "visited": [],
        "current": None,
    })

    while queue:
        d, u = heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)

        prev_dist = dict(dist)
        updated = []

        for v, w in graph[u].items():
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(queue, (dist[v], v))
                updated.append(v)

        steps.append({
            "phase": "confirm",
            "confirmed": u,
            "dist": {k: (v if v != float("inf") else None) for k, v in dist.items()},
            "visited": list(visited),
            "current": u,
            "updated": updated,
        })

    # 최단 경로 역추적
    path = []
    cur = "E"
    while cur:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    if path[0] != "A":
        path = []

    total = dist.get("E")

    # ── 역추적 단계 추가 ──────────────────────────────────────
    if path:
        # 역추적 시작 안내 스텝
        steps.append({
            "phase": "traceback_start",
            "confirmed": None,
            "dist": {k: (v if v != float("inf") else None) for k, v in dist.items()},
            "visited": list(visited),
            "current": None,
            "updated": [],
            "reveal_edges": [],
            "reveal_nodes": [],
        })

        # E←...←A 방향으로 한 간선씩 하이라이트
        reveal_edges = []
        reveal_nodes = [path[-1]]
        for i in range(len(path) - 1, 0, -1):
            node_from = path[i - 1]
            node_to   = path[i]
            reveal_edges.append([node_from, node_to])
            reveal_nodes.append(node_from)
            steps.append({
                "phase": "traceback",
                "confirmed": node_from,
                "dist": {k: (v if v != float("inf") else None) for k, v in dist.items()},
                "visited": list(visited),
                "current": node_from,
                "updated": [],
                "reveal_edges": [list(e) for e in reveal_edges],
                "reveal_nodes": list(reveal_nodes),
            })

        # 최종 완성 스텝
        steps.append({
            "phase": "done",
            "confirmed": None,
            "dist": {k: (v if v != float("inf") else None) for k, v in dist.items()},
            "visited": list(visited),
            "current": None,
            "updated": [],
            "reveal_edges": [list(e) for e in reveal_edges],
            "reveal_nodes": list(set(path)),
        })

    return steps, path, (total if total != float("inf") else None), prev


# ── Routes ───────────────────────────────────────────────────

@app.route("/")
def index():
    nodes = {
        k: {
            "lat": v[0], "lon": v[1],
            "name": v[2], "tag": v[3],
            "gx": GRAPH_POS[k][0],
            "gy": GRAPH_POS[k][1],
        }
        for k, v in LOCATIONS.items()
    }
    return render_template("index.html", nodes=nodes)


@app.route("/api/run", methods=["POST"])
def run():
    data = request.get_json()
    selected = data.get("selected", [])

    # A, E는 항상 포함
    active = list(set(["A", "E"] + selected))
    edge_list = build_edges(active)
    steps, path, total, prev = dijkstra(active, edge_list)

    return jsonify({
        "active": active,
        "edges": edge_list,
        "steps": steps,
        "path": path,
        "total": total,
        "prev": {k: v for k, v in prev.items() if v is not None},
        "locations": {
            k: {
                "name": LOCATIONS[k][2],
                "lat": LOCATIONS[k][0],
                "lon": LOCATIONS[k][1],
                "gx": GRAPH_POS[k][0],
                "gy": GRAPH_POS[k][1],
            }
            for k in active
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
