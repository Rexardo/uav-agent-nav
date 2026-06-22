import copy
import heapq
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.path import Path as MplPath

from python_motion_planning.common import Grid, TYPES, Visualizer2D
from python_motion_planning.path_planner import AStar, Dijkstra, RRT, RRTStar, ThetaStar

class PRM:
    # 【修复1】修正了 sample_num 的拼写
    def __init__(self, map_, start, goal, sample_num=1000, max_dist=12.0):
        self.map_ = map_
        self.start = start
        self.goal = goal
        self.sample_num = sample_num
        self.max_dist = max_dist

    def plan(self):
        width, height = self.map_.type_map.shape

        # Pick points randomly
        nodes = [self.start, self.goal]
        # 【修复1】配套修正
        while len(nodes) < self.sample_num + 2:
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            val = self.map_.type_map[x, y]
            if val != TYPES.OBSTACLE and val != TYPES.INFLATION:
                nodes.append((x, y))

        nodes = list(set(nodes))
        start_idx = nodes.index(self.start)
        goal_idx = nodes.index(self.goal)

        # Build Road map
        nodes_arr = np.array(nodes)
        graph = {i: [] for i in range(len(nodes))}

        for i, p1 in enumerate(nodes):
            # Find neighbor
            dists = np.linalg.norm(nodes_arr - p1, axis=1)
            neighbor_indices = np.where((dists > 0) & (dists <= self.max_dist))[0]

            for j in neighbor_indices:
                p2 = nodes[j]
                # Collision test
                if self._is_collision_free(p1, p2):
                    graph[i].append((j, dists[j]))

        # Find road by using A*
        open_set = []
        heapq.heappush(open_set, (0, 0, start_idx))
        came_from = {}
        g_score = {start_idx: 0}
        expand = {}

        success = False
        path = []

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == goal_idx:
                success = True
                path = [nodes[current]]
                while current in came_from:
                    current = came_from[current]
                    path.append(nodes[current])
                path = path[::-1]
                break

            # Record explored nodes to draw animation
            expand[nodes[current]] = True

            for neighbor, cost in graph[current]:
                tentative_g = current_g + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    # Euler Distance
                    h = np.hypot(nodes[neighbor][0] - nodes[goal_idx][0],
                                 nodes[neighbor][1] - nodes[goal_idx][1])
                    # 【修复2】把大括号 {} 改成了小括号 ()，将其变为元组
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, neighbor))

        # Calculate cost
        length = 0.0
        if success:
            for i in range(len(path) - 1):
                length += np.hypot(path[i][0]-path[i+1][0], path[i][1]-path[i+1][1])
        
        return path, {
            "success": success,
            "length": length,
            "cost": length,
            "expand": expand
        }

    def _is_collision_free(self, p1, p2):
        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = int(dist * 2)
        if steps == 0:
            return True
        
        xs = np.linspace(p1[0], p2[0], steps)
        ys = np.linspace(p1[1], p2[1], steps)

        for x, y in zip(xs, ys):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.map_.type_map.shape[0] and 0 <= iy < self.map_.type_map.shape[1]:
                val = self.map_.type_map[ix, iy]
                if val == TYPES.OBSTACLE or val == TYPES.INFLATION:
                    return False
        return True

def make_cmap_dict():
    """自定义颜色：把 CUSTOM 当成目标采集区域来显示。"""
    return {
        TYPES.FREE: "#ffffff",
        TYPES.OBSTACLE: "#000000",
        TYPES.START: "#00aa00",
        TYPES.GOAL: "#1155cc",
        TYPES.INFLATION: "#d9d9d9",
        TYPES.EXPAND: "#8ecae6",
        TYPES.CUSTOM: "#ffb000",
    }

def add_rectangle_obstacle(map_, xmin, xmax, ymin, ymax):
    map_.type_map[xmin:xmax, ymin:ymax] = TYPES.OBSTACLE

def add_polygon_obstacle(map_, vertices):
    polygon = MplPath(vertices)
    x_values = [p[0] for p in vertices]
    y_values = [p[1] for p in vertices]

    xmin = max(int(np.floor(min(x_values))), 0)
    xmax = min(int(np.ceil(max(x_values))) + 1, map_.type_map.shape[0])
    ymin = max(int(np.floor(min(y_values))), 0)
    ymax = min(int(np.ceil(max(y_values))) + 1, map_.type_map.shape[1])

    xs = np.arange(xmin, xmax)
    ys = np.arange(ymin, ymax)
    xx, yy = np.meshgrid(xs + 0.5, ys + 0.5, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])

    inside = polygon.contains_points(points).reshape(xx.shape)
    local_map = map_.type_map[xmin:xmax, ymin:ymax]
    local_map[inside] = TYPES.OBSTACLE

def add_target_area(map_, xmin, xmax, ymin, ymax):
    """目标采集区域只用于任务标记，不作为障碍物。"""
    map_.type_map[xmin:xmax, ymin:ymax] = TYPES.CUSTOM

def create_uav_grid_map():
    width = 70
    height = 45
    resolution = 1.0
    safety_buffer = 1

    map_ = Grid(bounds=[[0, width], [0, height]], resolution=resolution)
    map_.fill_boundary_with_obstacles()

    rectangle_obstacles = [
        (13, 21, 8, 18),
        (13, 21, 29, 38),
        (31, 39, 14, 28),
        (50, 58, 8, 19),
        (51, 59, 29, 37),
    ]
    for obs in rectangle_obstacles:
        add_rectangle_obstacle(map_, *obs)

    polygon_obstacles = [
        [(25, 5), (34, 7), (36, 13), (29, 17), (23, 12)],
        [(42, 22), (49, 24), (50, 31), (44, 35), (38, 29)],
        [(5, 22), (11, 19), (17, 23), (14, 28), (7, 28)],
    ]
    for vertices in polygon_obstacles:
        add_polygon_obstacle(map_, vertices)

    map_.inflate_obstacles(radius=safety_buffer)

    # 目标采集区域
    add_target_area(map_, xmin=56, xmax=67, ymin=20, ymax=27)

    start = (5, 6)
    task_point = (61, 23)  # 新增：黄色区域中心的巡检途径点
    goal = (65, 39)

    map_.type_map[start] = TYPES.START
    map_.type_map[goal] = TYPES.GOAL

    return map_, start, task_point, goal

def run_planner(algo_name, map_, start, goal):
    """统一接口：根据 algo_name 调用不同的底层算法"""
    if algo_name == "AStar":
        planner = AStar(map_=map_, start=start, goal=goal)
    elif algo_name == "Dijkstra":
        planner = Dijkstra(map_=map_, start=start, goal=goal)
    elif algo_name == "RRT":
        planner = RRT(map_=map_, start=start, goal=goal, max_dist=5.0, 
                      max_sample_step=30000, goal_sample_rate=0.10, 
                      discrete=True, use_faiss=True)
    elif algo_name == "RRTStar":
        planner = RRTStar(map_=map_, start=start, goal=goal, max_dist=5.0, 
                          max_sample_step=30000, goal_sample_rate=0.10, 
                          discrete=True, use_faiss=True, rewire_radius=10.0)
    elif algo_name == "ThetaStar":
        planner = ThetaStar(map_=map_, start=start, goal=goal)
    elif algo_name == "PRM":
        planner = PRM(map_=map_, start=start, goal=goal, sample_num=1500, max_dist=10.0)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    return planner.plan()

def plan_with_task(algo_name, map_, start, task_point, goal):
    """
    分段规划逻辑：
    Phase 1: Start -> Task Point
    Phase 2: Task Point -> Goal
    """
    print(f"[{algo_name}] Planning Phase 1: Start to Task Area...")
    path1, info1 = run_planner(algo_name, map_, start, task_point)
    if not info1["success"]:
        return [], {"success": False}

    print(f"[{algo_name}] Planning Phase 2: Task Area to Goal...")
    path2, info2 = run_planner(algo_name, map_, task_point, goal)
    if not info2["success"]:
        return [], {"success": False}

    # 合并路径：去掉 path2 的第一个点，防止在 task_point 重复停留
    final_path = path1 + path2[1:]

    # 直接合并两个阶段的展开节点字典，保留 (x, y) 坐标作为 key
    merged_expand = {**info1["expand"], **info2["expand"]}

    merged_info = {
        "success": True,
        "length": info1["length"] + info2["length"],
        "cost": info1["cost"] + info2["cost"],
        "expand": merged_expand
    }

    return final_path, merged_info

def save_final_path_figure(algo_name, map_, path, output_path):
    vis = Visualizer2D(figsize=(10, 8), cmap_dict=make_cmap_dict())
    vis.plot_grid_map(map_, equal=True)
    vis.plot_path(path, style="-", color="red", linewidth=3, label=f"{algo_name} path")
    vis.set_title(f"{algo_name} Final Path (With Inspection Task)")
    vis.legend()
    vis.savefig(output_path, dpi=300, bbox_inches="tight")
    vis.close()

def save_process_figure(algo_name, map_, path, path_info, output_path):
    process_map = copy.deepcopy(map_)
    process_map.fill_expands(path_info["expand"])

    vis = Visualizer2D(figsize=(10, 8), cmap_dict=make_cmap_dict())
    vis.plot_grid_map(process_map, equal=True)
    vis.plot_path(path, style="-", color="red", linewidth=3, label=f"{algo_name} path")
    vis.set_title(f"{algo_name} Search Process")
    vis.legend()
    vis.savefig(output_path, dpi=300, bbox_inches="tight")
    vis.close()

def map_points_to_world_xy(map_, points):
    xs, ys = [], []
    for p in points:
        x, y = map_.map_to_world(p)
        xs.append(x)
        ys.append(y)
    return xs, ys

def save_search_animation(algo_name, map_, path, path_info, output_path):
    expand_points = list(path_info["expand"].keys()) 
    expand_x, expand_y = map_points_to_world_xy(map_, expand_points)
    path_x, path_y = map_points_to_world_xy(map_, path)

    vis = Visualizer2D(figsize=(10, 8), cmap_dict=make_cmap_dict())
    vis.plot_grid_map(map_, equal=True)

    expand_scatter = vis.ax.scatter([], [], s=10, c="#8ecae6", alpha=0.75, label="Expanded nodes", zorder=40)
    path_line, = vis.ax.plot([], [], "-", color="red", linewidth=3, label=f"{algo_name} path", zorder=50)
    vis.ax.legend(loc="upper right")

    search_frames = 160
    pause_frames = 60
    total_frames = search_frames + pause_frames

    if len(expand_points) > 0:
        frame_indices = np.linspace(0, len(expand_points), search_frames, dtype=int)
    else:
        frame_indices = np.zeros(search_frames, dtype=int)

    def update(frame_id):
        if frame_id < search_frames:
            k = frame_indices[frame_id]
            is_final_phase = False
        else:
            k = frame_indices[-1]
            is_final_phase = True

        if k > 0:
            offsets = np.column_stack([expand_x[:k], expand_y[:k]])
        else:
            offsets = np.empty((0, 2))

        expand_scatter.set_offsets(offsets)

        if is_final_phase or frame_id == search_frames - 1:
            path_line.set_data(path_x, path_y)
            vis.ax.set_title(f"{algo_name} Animation: Final Path Found")
        else:
            path_line.set_data([], [])
            vis.ax.set_title(f"{algo_name} Animation: Expanded {k} nodes")

        return expand_scatter, path_line

    ani = animation.FuncAnimation(vis.fig, update, frames=total_frames, interval=50, blit=False, repeat=False)
    ani.save(output_path, writer=PillowWriter(fps=20))
    vis.close()

def main():
    output_dir = Path("grid_map_unified_task_results")
    output_dir.mkdir(exist_ok=True)

    map_, start, task_point, goal = create_uav_grid_map()

    # 这里包含了所有的算法
    algorithms_to_run = ["AStar", "Dijkstra", "RRT", "RRTStar", "ThetaStar", "PRM"]

    for algo in algorithms_to_run:
        print(f"\n{'='*40}\nRunning Algorithm: {algo}\n{'='*40}")
        
        path, path_info = plan_with_task(algo, map_, start, task_point, goal)

        if not path_info["success"]:
            print(f"{algo} failed to find a path.")
            continue

        print(f"Success: {path_info['success']}")
        print(f"Path length: {path_info['length']:.2f}")
        print(f"Path cost: {path_info['cost']:.2f}")
        print(f"Number of path points: {len(path)}")
        print(f"Number of expanded nodes: {len(path_info['expand'])}")

        # 统一保存对应算法的图像
        save_process_figure(algo, map_, path, path_info, output_dir / f"{algo.lower()}_process.png")
        save_final_path_figure(algo, map_, path, output_dir / f"{algo.lower()}_final_path.png")
        save_search_animation(algo, map_, path, path_info, output_dir / f"{algo.lower()}_animation.gif")
        print(f"Files saved for {algo} in '{output_dir.name}' folder.")

if __name__ == "__main__":
    main()