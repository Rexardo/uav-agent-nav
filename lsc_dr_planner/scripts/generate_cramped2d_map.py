import csv
import json

# =========================
# 基本设置
# =========================
RES = 0.5          # 一个字符格 = 0.5 m
MAP_W = 30.0       # 地图宽度 30 m
MAP_H = 15.0       # 地图高度 15 m

Z = 1.25
SX = 0.5
SY = 0.5
SZ = 2.5

MAP_CSV = "world/cramped2d/cramped2d_1.csv"
MISSION_JSON = "missions/cramped2d/maze10_1.json"


# =========================
# 地图字符画
#
# # = 障碍物
# . = 空地
#
# 注意：
# 1. 每一行必须是 61 个字符
# 2. 一共必须是 31 行
# 3. 第一行是地图最上方 y=15m
# 4. 最后一行是地图最下方 y=0m
# =========================

LAYOUT = [
    "#############################################################",
    "#..................................................####.....#",
    "#....#######....##########.........#########.......####.....#",
    "#....#######..##...#######.........####################.....#",
    "#....#########.....#######.........####################.....#",
    "#....#######.......#########.......#########....#####........",
    "#....#######..............###...................#####.......#",
    ".....#######..##########..###........####..##########.......#",
    "#.............##########..###........####..##########.......#",
    "########......##########..###.......####...##########.......#",
    "#.............##########.......##...####....########...######",
    "#.............................##............................#",
    "#.....##########......####...##.........####.......###......#",
    "#.....##########......####..##.####.....####.......###......#",
    "#.....###.....##......####.##..####....#####.......###......#",
    "..............##...............####...###.....##............#",
    "#....................................###.......##............",
    "#.....###..############################........#######......#",
    "#.....###..############################........#######......#",
    "#.....####...##########################........#########....#",
    "#........##............###.....................##########...#",
    "#.........##...........###..............#####..##############",
    "#..........##..####....###.##################.............###",
    "#...........##.####....###.##################...............#",
    "#....####......####........##################..###.####.....#",
    ".....####.....................................###..####.....#",
    "#....####.......########........########.#######..#####......",
    "#....####.......########........########..........###########",
    "#....####.......########........########.......##############",
    "#...............########............................#########",
    "#############################################################",
]

def print_layout_lengths():
    print("===== LAYOUT row lengths =====")
    for i, line in enumerate(LAYOUT):
        print(f"row {i:02d}: length={len(line)}")

def check_layout():
    rows = len(LAYOUT)
    cols = len(LAYOUT[0])

    for i, line in enumerate(LAYOUT):
        if len(line) != cols:
            raise ValueError(
                f"第 {i} 行长度是 {len(line)}，但第一行长度是 {cols}。每一行长度必须一样。"
            )
        for ch in line:
            if ch not in ["#", "."]:
                raise ValueError(f"地图里只能用 # 和 .，但发现了 {ch}")

    expected_cols = int(MAP_W / RES) + 1
    expected_rows = int(MAP_H / RES) + 1

    if cols != expected_cols:
        raise ValueError(f"列数应该是 {expected_cols}，但现在是 {cols}")

    if rows != expected_rows:
        raise ValueError(f"行数应该是 {expected_rows}，但现在是 {rows}")

    return rows, cols


def generate_map():
    rows_count, cols_count = check_layout()

    obstacles = []

    for row_idx, line in enumerate(LAYOUT):
        for col_idx, ch in enumerate(line):
            if ch == "#":
                x = col_idx * RES
                y = (rows_count - 1 - row_idx) * RES
                obstacles.append([x, y, Z, SX, SY, SZ])

    with open(MAP_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(obstacles)

    print("Saved map:", MAP_CSV)
    print("Map size:", MAP_W, "m x", MAP_H, "m")
    print("Grid size:", cols_count, "cols x", rows_count, "rows")
    print("Obstacle count:", len(obstacles))


def update_mission():
    rows_count, cols_count = check_layout()

    with open(MISSION_JSON, "r") as f:
        data = json.load(f)

    # 世界范围比障碍物地图更大一点，给左侧起点和右侧终点留空间
    data["world"] = [
        {
            "dimension": [
                -1.5,      # 左侧给起点留空间
                0.0,       # 下方不留绕行空间
                0.0,
                MAP_W + 1.5,  # 右侧给终点留空间
                MAP_H,        # 上方不留绕行空间
                2.5,
            ]
        }
    ]

    # 20 架无人机：左边一列出发，右边一列到达
    start_x = -1.0
    goal_x = MAP_W + 1.0

    # y 从 1.0 到 14.0 均匀排开
    # 20 台无人机之间间距约 0.68 m，比 0.5 m 更安全
    y_min = 1.0
    y_max = MAP_H - 1.0
    ys = [y_min + (y_max - y_min) * i / 19 for i in range(20)]

    agents = []
    for i, y in enumerate(ys, start=1):
        agents.append(
            {
                "type": "crazyflie",
                "cid": i,
                "start": [start_x, y, 1],
                "goal": [goal_x, y, 1],
            }
        )

    data["agents"] = agents

    with open(MISSION_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print("Updated mission:", MISSION_JSON)
    print("Total agents:", len(agents))
    print("Start x:", start_x)
    print("Goal x:", goal_x)
    print("Y range:", y_min, "to", y_max)


if __name__ == "__main__":
    print_layout_lengths()
    generate_map()
    update_mission()