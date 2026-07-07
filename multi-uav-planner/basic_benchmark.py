import math
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

from a_star import astar
from map_generator import generate_test_map

# ==========================================
# Static map and UAV parameters
# ==========================================
class StaticMap:
    def __init__(self, width, height, circles=None, rectangles=None, inflation_radius=0.0):
        self.width = width
        self.height = height
        self.circles = circles if circles else []
        self.rectangles = rectangles if rectangles else []
        self.inflation_radius = inflation_radius  

        self.grid_map = self._generate_grid()

    def _generate_grid(self):
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                for obs_x, obs_y, obs_r in self.circles:
                    if math.hypot(x - obs_x, y - obs_y) <= (obs_r + self.inflation_radius):
                        grid[y][x] = 1
                        break
                if grid[y][x] == 0:
                    for rx, ry, rw, rh in self.rectangles:
                        if (rx - self.inflation_radius) <= x <= (rx + rw + self.inflation_radius) and \
                           (ry - self.inflation_radius) <= y <= (ry + rh + self.inflation_radius):
                            grid[y][x] = 1
                            break
        return grid


def make_center_inspection_region(width, height, region_size=30):
    x_min = (width - region_size) // 2
    y_min = (height - region_size) // 2
    return [(x_min, y_min, region_size, region_size)]


def is_in_inspection_regions(x, y, inspection_regions):
    for rx, ry, rw, rh in inspection_regions:
        if rx <= x < rx + rw and ry <= y < ry + rh:
            return True
    return False


def compute_reachable_mask(static_map, starts):
    reachable = np.zeros((static_map.height, static_map.width), dtype=bool)
    queue = deque()

    for start in starts:
        sx, sy = int(round(start[0])), int(round(start[1]))
        if 0 <= sx < static_map.width and 0 <= sy < static_map.height:
            if static_map.grid_map[sy][sx] == 0 and not reachable[sy, sx]:
                reachable[sy, sx] = True
                queue.append((sx, sy))

    motions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in motions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < static_map.width and 0 <= ny < static_map.height):
                continue
            if reachable[ny, nx]:
                continue
            if static_map.grid_map[ny][nx] == 1:
                continue
            reachable[ny, nx] = True
            queue.append((nx, ny))
    return reachable


def generate_reachable_inspection_waypoints(
    static_map, uav_starts, inspection_regions, coverage_radius=3.0, waypoint_spacing=None
):
    if waypoint_spacing is None:
        waypoint_spacing = max(1, int(round(coverage_radius * 1.5)))

    reachable = compute_reachable_mask(static_map, uav_starts)
    waypoints = []

    for y in range(static_map.height):
        for x in range(static_map.width):
            if x % waypoint_spacing != 0 or y % waypoint_spacing != 0:
                continue
            if not is_in_inspection_regions(x, y, inspection_regions):
                continue
            if static_map.grid_map[y][x] == 1:
                continue
            if not reachable[y, x]:
                continue
            waypoints.append((x, y))

    return waypoints, reachable, waypoint_spacing


def nearest_neighbor_order(start, waypoints):
    if not waypoints:
        return []
    remaining = [tuple(wp) for wp in waypoints]
    ordered = []
    current = np.array(start, dtype=float)

    while remaining:
        best_idx = min(
            range(len(remaining)),
            key=lambda i: np.linalg.norm(np.array(remaining[i], dtype=float) - current)
        )
        next_wp = remaining.pop(best_idx)
        ordered.append(next_wp)
        current = np.array(next_wp, dtype=float)
    return ordered


def assign_waypoints_to_uavs(static_map, waypoints, uav_tasks):
    starts = [start for start, _ in uav_tasks]
    per_uav_reachable = [compute_reachable_mask(static_map, [start]) for start in starts]

    assignments = {i + 1: [] for i in range(len(uav_tasks))}
    unassigned = []

    for wp in waypoints:
        wx, wy = wp
        candidate_ids = [i for i, reachable in enumerate(per_uav_reachable) if reachable[wy, wx]]

        if not candidate_ids:
            unassigned.append(wp)
            continue

        best_i = min(
            candidate_ids,
            key=lambda i: np.linalg.norm(np.array(wp, dtype=float) - np.array(starts[i], dtype=float)) + 20.0 * len(assignments[i + 1])
        )
        assignments[best_i + 1].append(wp)

    for uav_id, wps in assignments.items():
        start = starts[uav_id - 1]
        assignments[uav_id] = nearest_neighbor_order(start, wps)
    return assignments, unassigned


class InspectionMap:
    def __init__(self, static_map, inspection_regions, coverage_radius):
        self.static_map = static_map
        self.inspection_regions = inspection_regions
        self.coverage_radius = coverage_radius
        self.required = np.zeros((static_map.height, static_map.width), dtype=bool)
        self.covered = np.zeros((static_map.height, static_map.width), dtype=bool)

        for y in range(static_map.height):
            for x in range(static_map.width):
                if not is_in_inspection_regions(x, y, inspection_regions):
                    continue
                if static_map.grid_map[y][x] == 1:
                    continue
                self.required[y, x] = True

    def update_coverage(self, pos):
        x0, y0 = float(pos[0]), float(pos[1])
        r = float(self.coverage_radius)
        r_int = int(math.ceil(r))

        x_min = max(0, int(math.floor(x0 - r_int)))
        x_max = min(self.static_map.width - 1, int(math.ceil(x0 + r_int)))
        y_min = max(0, int(math.floor(y0 - r_int)))
        y_max = min(self.static_map.height - 1, int(math.ceil(y0 + r_int)))

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if not self.required[y, x]:
                    continue
                if math.hypot(x - x0, y - y0) <= r:
                    self.covered[y, x] = True

    def coverage_ratio(self):
        total = int(np.sum(self.required))
        if total == 0:
            return 1.0
        covered_count = int(np.sum(self.required & self.covered))
        return covered_count / total


class UAV:
    def __init__(self, uav_id, start, goal, safe_radius, comm_range, horizon, waypoints=None, coverage_radius=3.0):
        self.id = uav_id
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.safe_radius = safe_radius
        self.comm_range = comm_range
        self.horizon = horizon 
        self.coverage_radius = coverage_radius 
        self.waypoints = [np.array(wp, dtype=float) for wp in (waypoints if waypoints else [])]
        self.current_wp_idx = 0
        self.inspection_finished = len(self.waypoints) == 0
        self.current_path = [] 
        self.velocity = np.array([0.0, 0.0])
        self.is_reached = False
        self.max_speed = 1.0
        self.path_length = 0.0      
        self.wait_steps = 0         
        self.is_yielding = False    
        self.yield_timer = 0
        self.history = []           

    def get_distance(self, other_pos):
        return np.linalg.norm(self.pos - other_pos)

    def get_current_target(self):
        self.update_mission_status()
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return self.goal

    def update_mission_status(self):
        waypoint_reach_threshold = 0.5
        while self.current_wp_idx < len(self.waypoints):
            target = self.waypoints[self.current_wp_idx]
            if np.linalg.norm(self.pos - target) <= waypoint_reach_threshold:
                self.current_wp_idx += 1
            else:
                break
        self.inspection_finished = self.current_wp_idx >= len(self.waypoints)
        if self.inspection_finished and np.linalg.norm(self.pos - self.goal) < 0.5:
            self.is_reached = True

    def plan_reference_path(self, static_map):
        self.update_mission_status()
        if self.is_reached or self.is_yielding:
            return

        target = self.get_current_target()
        start_grid = (max(0, min(static_map.width - 1, int(round(self.pos[0])))),
                      max(0, min(static_map.height - 1, int(round(self.pos[1])))))
        goal_grid = (max(0, min(static_map.width - 1, int(round(target[0])))),
                     max(0, min(static_map.height - 1, int(round(target[1])))))

        full_path = astar(start_grid, goal_grid, static_map.grid_map)
        self.current_path = []

        if not full_path:
            self.current_path = [self.pos.copy() for _ in range(self.horizon)]
            return

        if len(full_path) > 1 and full_path[0] == start_grid:
            full_path = full_path[1:]

        for i in range(self.horizon):
            if i < len(full_path):
                self.current_path.append(np.array([float(full_path[i][0]), float(full_path[i][1])]))
            else:
                self.current_path.append(target.copy())

    def communicate(self, all_uavs):
        neighbors_info = []
        for other in all_uavs:
            if other.id != self.id and self.get_distance(other.pos) <= self.comm_range:
                neighbors_info.append({
                    'id': other.id, 'pos': other.pos, 'path': other.current_path,
                    'safe_radius': other.safe_radius, 'is_reached': other.is_reached
                })
        return neighbors_info

    def resolve_conflicts_and_replan(self, neighbors_info, static_map):
        if self.is_reached or not self.current_path:
            return False
        if self.is_yielding:
            self.yield_timer -= 1
            if self.yield_timer <= 0:
                self.is_yielding = False
            return False

        conflict_detected = False
        conflict_neighbor = None

        for step, my_next_pos in enumerate(self.current_path):
            for neighbor in neighbors_info:
                if neighbor.get('is_reached', False):
                    continue
                neighbor_next_pos = neighbor['path'][step] if step < len(neighbor['path']) else (neighbor['pos'] if not neighbor['path'] else neighbor['path'][-1])
                if np.linalg.norm(my_next_pos - neighbor_next_pos) < (self.safe_radius + neighbor['safe_radius'] + 0.5):
                    conflict_detected = True
                    conflict_neighbor = neighbor
                    break
            if conflict_detected:
                break

        if conflict_detected:
            self.wait_steps += 1
            if self.wait_steps > 3:
                if self.id > conflict_neighbor['id']:
                    parking_spot = self._find_parking_spot(static_map, neighbors_info, conflict_neighbor)
                    if parking_spot is not None:
                        self.is_yielding = True
                        self.yield_timer = 3  
                        self.wait_steps = 0
                        self.current_path = []
                        dir_to_park = parking_spot - self.pos
                        dist = np.linalg.norm(dir_to_park)
                        dir_norm = dir_to_park / dist if dist > 0 else np.array([0.0, 0.0])
                        for i in range(self.horizon):
                            if i < self.yield_timer:
                                self.current_path.append(self.pos + dir_norm * min(self.max_speed * (i + 1), dist))
                            else:
                                self.current_path.append(parking_spot.copy())
                        return True
                    else:
                        self.current_path = [self.pos.copy() for _ in range(self.horizon)]
                else:
                    self.current_path = [self.pos.copy() for _ in range(self.horizon)]
            else:
                self.current_path = [self.pos.copy() for _ in range(self.horizon)]
        else:
            self.wait_steps = 0
        return False

    def _find_parking_spot(self, static_map, neighbors_info, conflict_neighbor):
        forbidden_grids = set()
        for neighbor in neighbors_info:
            if neighbor.get('is_reached', False):
                continue
            forbidden_grids.add((int(round(neighbor['pos'][0])), int(round(neighbor['pos'][1]))))
            for p in neighbor['path']:
                px, py = int(round(p[0])), int(round(p[1]))
                forbidden_grids.add((px, py))
                if neighbor['id'] == conflict_neighbor['id']:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        forbidden_grids.add((px + dx, py + dy))

        history_grids = set((int(round(p[0])), int(round(p[1]))) for p in self.history)
        start_grid = (int(round(self.pos[0])), int(round(self.pos[1])))
        queue, visited, candidates, search_count = [start_grid], {start_grid}, [], 0
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while queue and search_count < 150:
            curr = queue.pop(0)
            search_count += 1
            if 0 <= curr[0] < static_map.width and 0 <= curr[1] < static_map.height:
                if curr not in forbidden_grids and static_map.grid_map[curr[1]][curr[0]] == 0:
                    if curr != start_grid:
                        candidates.append(curr)
                        if len(candidates) >= 15:
                            break
            for m in motions:
                nx, ny = curr[0] + m[0], curr[1] + m[1]
                if 0 <= nx < static_map.width and 0 <= ny < static_map.height and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        if not candidates:
            return None

        best_spot, max_score = None, -float('inf')
        for spot in candidates:
            dist_to_conflict = math.hypot(spot[0] - conflict_neighbor['pos'][0], spot[1] - conflict_neighbor['pos'][1])
            min_dist_to_uavs = min([math.hypot(spot[0] - n['pos'][0], spot[1] - n['pos'][1]) for n in neighbors_info if not n.get('is_reached', False)] + [float('inf')])
            if min_dist_to_uavs == float('inf'): min_dist_to_uavs = 0.0
            
            min_dist_to_obs = float('inf')
            for cx, cy, r in static_map.circles:
                min_dist_to_obs = min(min_dist_to_obs, math.hypot(spot[0] - cx, spot[1] - cy) - r)
            for rx, ry, rw, rh in static_map.rectangles:
                min_dist_to_obs = min(min_dist_to_obs, math.hypot(max(rx - spot[0], 0, spot[0] - (rx + rw)), max(ry - spot[1], 0, spot[1] - (ry + rh))))
            if min_dist_to_obs == float('inf'): min_dist_to_obs = 0.0

            score = dist_to_conflict * 1.0 + min_dist_to_uavs * 1.5 + min_dist_to_obs * 2.0 - (100.0 if spot in history_grids else 0.0)
            if score > max_score:
                max_score, best_spot = score, spot
        return np.array([float(best_spot[0]), float(best_spot[1])])

    def step_forward(self):
        self.update_mission_status()
        if self.is_reached: return
        if self.current_path:
            self.history.append(self.pos.copy())
            if len(self.history) > 10: self.history.pop(0)
            next_step = self.current_path.pop(0)
            self.velocity = next_step - self.pos
            self.path_length += np.linalg.norm(self.velocity)
            self.pos = next_step
        else:
            self.velocity = np.array([0.0, 0.0])
        self.update_mission_status()


# ==========================================
# run_simulation (返回数据字典，包含总步数)
# ==========================================
def run_simulation(
    num_uavs=8, width=50, height=50, num_obstacles=50, map_seed=42, inflation_radius=0.8,
    max_logical_steps=1000, coverage_radius=3.0, center_region_size=30, waypoint_spacing=None,
    verbose=False
):
    test_circles, test_rectangles, _ = generate_test_map(width=width, height=height, num_obstacles=num_obstacles, seed=map_seed)
    env_map = StaticMap(width, height, circles=test_circles, rectangles=test_rectangles, inflation_radius=inflation_radius)

    num_uavs = max(1, min(8, num_uavs)) 
    all_uav_tasks = [
        ([2, 2], [2, 2]), ([width - 3, 2], [width - 3, 2]), ([2, 5], [2, 2]), ([width - 3, 5], [width - 3, 2]),                 
        ([5, 2], [2, 2]), ([width - 6, 2], [width - 3, 2]), ([5, 5], [2, 2]), ([width - 6, 5], [width - 3, 2]),                 
    ]
    uav_tasks = all_uav_tasks[:num_uavs]
    inspection_regions = make_center_inspection_region(width, height, center_region_size)
    uav_starts = [start for start, _ in uav_tasks]

    inspection_waypoints, _, _ = generate_reachable_inspection_waypoints(
        static_map=env_map, uav_starts=uav_starts, inspection_regions=inspection_regions,
        coverage_radius=coverage_radius, waypoint_spacing=waypoint_spacing
    )
    waypoint_assignments, _ = assign_waypoints_to_uavs(env_map, inspection_waypoints, uav_tasks)
    inspection_map = InspectionMap(env_map, inspection_regions, coverage_radius)

    uavs = [UAV(i, start, goal, safe_radius=0.8, comm_range=7.0, horizon=5, waypoints=waypoint_assignments[i], coverage_radius=coverage_radius)
            for i, (start, goal) in enumerate(uav_tasks, start=1)]

    for uav in uavs:
        inspection_map.update_coverage(uav.pos)

    total_deadlocks = 0  
    total_steps = max_logical_steps  # 默认使用最大步数
    
    if verbose:
        print(f"  [Map {map_seed}] 启动仿真: UAV数量={num_uavs}...")

    for t in range(max_logical_steps):
        for uav in uavs: uav.plan_reference_path(env_map)
        for uav in uavs:
            if uav.resolve_conflicts_and_replan(uav.communicate(uavs), env_map):
                total_deadlocks += 1
        for uav in uavs:
            uav.step_forward()
            inspection_map.update_coverage(uav.pos)

        # 当所有无人机任务结束并成功到达终点时
        if all(uav.is_reached for uav in uavs):
            total_steps = t + 1  # 记录实际消耗的总步数
            break

    coverage = inspection_map.coverage_ratio() * 100.0
    total_path = sum(uav.path_length for uav in uavs)
    avg_path = total_path / num_uavs if num_uavs > 0 else 0

    if verbose:
        print(f"    -> 完成! 覆盖率:{coverage:.1f}%, 死锁数:{total_deadlocks}, 总步数:{total_steps}, 平均路径:{avg_path:.1f}")

    return {
        "coverage": coverage,
        "deadlocks": total_deadlocks,
        "total_path": total_path,
        "avg_path": avg_path,
        "total_steps": total_steps   # <=== 新增返回总步数
    }


# ==========================================
# Benchmark & Plotting
# ==========================================
def run_benchmark_and_plot(num_maps=5, max_uavs=8):
    print(f"\n==============================================")
    print(f"🚀 开始批量基准测试 (Benchmark)")
    print(f"   共计 {num_maps} 张随机地图 | 每张地图跑 1 到 {max_uavs} 架无人机")
    print(f"==============================================\n")

    # Data structure to store results
    metrics = {
        uav_num: {"coverage": [], "deadlocks": [], "total_path": [], "avg_path": [], "total_steps": []} 
        for uav_num in range(1, max_uavs + 1)
    }

    for step in range(num_maps):
        current_seed = random.randint(0, 999999)
        print(f">>> 正在测试 Map {step + 1}/{num_maps} (Seed: {current_seed})")
        
        for num_uavs in range(1, max_uavs + 1):
            results = run_simulation(num_uavs=num_uavs, map_seed=current_seed, verbose=False)
            
            metrics[num_uavs]["coverage"].append(results["coverage"])
            metrics[num_uavs]["deadlocks"].append(results["deadlocks"])
            metrics[num_uavs]["total_path"].append(results["total_path"])
            metrics[num_uavs]["avg_path"].append(results["avg_path"])
            metrics[num_uavs]["total_steps"].append(results["total_steps"])  # <=== 保存步数数据

    print("\n✅ 所有测试运行完毕，正在生成图表...")

    x_axis = list(range(1, max_uavs + 1))
    
    mean_coverage = [np.mean(metrics[u]["coverage"]) for u in x_axis]
    std_coverage = [np.std(metrics[u]["coverage"]) for u in x_axis]
    
    mean_deadlocks = [np.mean(metrics[u]["deadlocks"]) for u in x_axis]
    std_deadlocks = [np.std(metrics[u]["deadlocks"]) for u in x_axis]
    
    mean_total_path = [np.mean(metrics[u]["total_path"]) for u in x_axis]
    std_total_path = [np.std(metrics[u]["total_path"]) for u in x_axis]
    
    mean_avg_path = [np.mean(metrics[u]["avg_path"]) for u in x_axis]
    std_avg_path = [np.std(metrics[u]["avg_path"]) for u in x_axis]

    mean_steps = [np.mean(metrics[u]["total_steps"]) for u in x_axis]        # <=== 聚合步数均值
    std_steps = [np.std(metrics[u]["total_steps"]) for u in x_axis]          # <=== 聚合步数标准差

    # Plotting 
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"MAPF Benchmark Results (Averaged over {num_maps} Random Maps)", fontsize=16)

    # 1. Coverage
    axs[0, 0].plot(x_axis, mean_coverage, '-o', color='blue')
    axs[0, 0].set_title('Task Coverage Ratio (%)')
    axs[0, 0].set_xlabel('Number of UAVs')
    axs[0, 0].set_ylabel('Coverage (%)')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Total Steps (Makespan)  <=== 新增步数折线图
    axs[0, 1].plot(x_axis, mean_steps, '-o', color='orange')
    axs[0, 1].set_title('Total Steps (Time)')
    axs[0, 1].set_xlabel('Number of UAVs')
    axs[0, 1].set_ylabel('Steps')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Total Path Length
    axs[1, 0].plot(x_axis, mean_total_path, '-o', color='green')
    axs[1, 0].set_title('Total Path Length')
    axs[1, 0].set_xlabel('Number of UAVs')
    axs[1, 0].set_ylabel('Distance')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 4. Average Path Length
    axs[1, 1].plot(x_axis, mean_avg_path, '-o', color='purple')
    axs[1, 1].set_title('Average Path Length per UAV')
    axs[1, 1].set_xlabel('Number of UAVs')
    axs[1, 1].set_ylabel('Distance')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("benchmark_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # 默认跑 5 张随机地图，你可以根据需要调整这里的参数进行测试
    run_benchmark_and_plot(num_maps=100, max_uavs=8)