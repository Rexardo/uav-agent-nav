import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# 从你的原始文件中导入依赖
from map_generator import generate_test_map
from a_star import astar
from apf_dynamic import apf
from multi_path_plan_with_cover_task import (
    StaticMap,
    make_center_inspection_region,
    generate_reachable_inspection_waypoints,
    assign_waypoints_to_uavs,
    InspectionMap
)

# ==========================================
# 统一测试用的 UAV 类 (支持三种算法模式)
# ==========================================
class EvalUAV:
    def __init__(self, uav_id, start, goal, safe_radius, comm_range, horizon, waypoints, coverage_radius, mode='astar'):
        self.id = uav_id
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.safe_radius = safe_radius
        self.comm_range = comm_range
        self.horizon = horizon
        self.coverage_radius = coverage_radius
        self.mode = mode  # 'astar', 'apf', 'hybrid'

        # 巡检任务属性
        self.waypoints = [np.array(wp, dtype=float) for wp in (waypoints if waypoints else [])]
        self.current_wp_idx = 0
        self.inspection_finished = len(self.waypoints) == 0
        self.max_speed = 1.0

        # 运动与状态
        self.current_path = []
        self.velocity = np.array([0.0, 0.0])
        self.is_reached = False
        self.history = []

        # 死锁与避让统计
        self.wait_steps = 0
        self.is_yielding = False
        self.yield_timer = 0
        self.total_yield_count = 0  # 核心统计指标：避让触发次数

        # Hybrid 模式专属属性
        self.global_path = []
        self.global_path_index = 0
        self.global_lookahead = 8
        self.global_replan_interval = 35
        self.last_global_replan = -10**9
        self.local_target = self.goal.copy()

    def get_distance(self, other_pos):
        return np.linalg.norm(self.pos - other_pos)

    def _clip_to_grid(self, point, static_map):
        grid = (int(round(point[0])), int(round(point[1])))
        return (
            max(0, min(static_map.width - 1, grid[0])),
            max(0, min(static_map.height - 1, grid[1]))
        )

    def _history_grid_cells(self):
        return {(int(round(p[0])), int(round(p[1]))) for p in self.history}

    def update_mission_status(self):
        waypoint_reach_threshold = 0.5
        while self.current_wp_idx < len(self.waypoints):
            target = self.waypoints[self.current_wp_idx]
            if np.linalg.norm(self.pos - target) <= waypoint_reach_threshold:
                self.current_wp_idx += 1
            else:
                break
        self.inspection_finished = self.current_wp_idx >= len(self.waypoints)
        if self.inspection_finished and np.linalg.norm(self.pos - self.goal) < 0.1:
            self.is_reached = True

    def get_current_target(self):
        self.update_mission_status()
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return self.goal

    # ==========================================
    # 核心规划切换逻辑
    # ==========================================
    def plan_reference_path(self, static_map, neighbors_info=None, logical_step=0):
        if self.is_reached or self.is_yielding:
            return

        target = self.get_current_target()
        start_grid = self._clip_to_grid(self.pos, static_map)
        target_grid = self._clip_to_grid(target, static_map)

        if self.mode == 'astar':
            full_path = astar(start_grid, target_grid, static_map.grid_map)
            self.current_path = []
            if not full_path:
                self.current_path = [self.pos.copy() for _ in range(self.horizon)]
                return
            if len(full_path) > 1 and full_path[0] == start_grid:
                full_path = full_path[1:]
            for i in range(self.horizon):
                if i < len(full_path):
                    self.current_path.append(np.array([full_path[i][0], full_path[i][1]], dtype=float))
                else:
                    self.current_path.append(target.copy())

        elif self.mode == 'apf':
            full_path = apf(
                start_grid, target_grid, static_map.grid_map,
                dynamic_obstacles=neighbors_info, own_safe_radius=self.safe_radius, own_id=self.id,
                k_att=1.0, k_rep=80.0, influence_radius=5.0,
                k_dynamic=180.0, dynamic_influence_radius=self.comm_range, max_iterations=200, return_partial=True
            )
            self._fill_current_path_from_grid_path(full_path, start_grid)

        elif self.mode == 'hybrid':
            # 需要判断是否更换了目标点，如果换了强制重规划全局路径
            target_changed = (np.linalg.norm(self.local_target - target) > 1.0)
            if target_changed or (not self.global_path) or (logical_step - self.last_global_replan >= self.global_replan_interval):
                path = astar(start_grid, target_grid, static_map.grid_map)
                if path:
                    self.global_path = path
                    self.global_path_index = 0
                    self.last_global_replan = logical_step
                self.local_target = target.copy()

            # 选取全局路径上的前瞻点
            lookahead_target = target
            if self.global_path:
                nearest_idx = min(range(len(self.global_path)), key=lambda i: math.hypot(self.pos[0] - self.global_path[i][0], self.pos[1] - self.global_path[i][1]))
                self.global_path_index = max(self.global_path_index, nearest_idx)
                target_idx = min(self.global_path_index + self.global_lookahead, len(self.global_path) - 1)
                tx, ty = self.global_path[target_idx]
                lookahead_target = np.array([float(tx), float(ty)])

            lookahead_grid = self._clip_to_grid(lookahead_target, static_map)
            full_path = apf(
                start_grid, lookahead_grid, static_map.grid_map,
                dynamic_obstacles=neighbors_info, own_safe_radius=self.safe_radius, own_id=self.id,
                k_att=1.2, k_rep=55.0, influence_radius=3.5,
                k_dynamic=260.0, dynamic_influence_radius=self.comm_range, max_iterations=200, return_partial=True
            )
            if not self._fill_current_path_from_grid_path(full_path, start_grid):
                self.current_path = [self.pos.copy() for _ in range(self.horizon)]

    def _fill_current_path_from_grid_path(self, full_path, start_grid=None):
        self.current_path = []
        if not full_path:
            self.current_path = [self.pos.copy() for _ in range(self.horizon)]
            return False
        if start_grid is not None and len(full_path) > 1 and full_path[0] == start_grid:
            full_path = full_path[1:]
        if not full_path:
            self.current_path = [self.pos.copy() for _ in range(self.horizon)]
            return False
        last_point = np.array([full_path[-1][0], full_path[-1][1]], dtype=float)
        for i in range(self.horizon):
            if i < len(full_path):
                self.current_path.append(np.array([full_path[i][0], full_path[i][1]], dtype=float))
            else:
                self.current_path.append(last_point.copy())
        return True

    def communicate(self, all_uavs):
        neighbors_info = []
        for other in all_uavs:
            if other.id != self.id and self.get_distance(other.pos) <= self.comm_range:
                neighbors_info.append({
                    'id': other.id, 'pos': other.pos.copy(),
                    'path': [p.copy() for p in other.current_path], 'safe_radius': other.safe_radius
                })
        return neighbors_info

    def resolve_conflicts_and_replan(self, neighbors_info, static_map):
        if self.is_reached or not self.current_path: return
        if self.is_yielding:
            self.yield_timer -= 1
            if self.yield_timer <= 0: self.is_yielding = False
            return

        conflict_detected = False
        conflict_neighbor = None

        for step, my_next_pos in enumerate(self.current_path):
            for neighbor in neighbors_info:
                neighbor_next_pos = neighbor['path'][step] if step < len(neighbor['path']) else (neighbor['pos'] if not neighbor['path'] else neighbor['path'][-1])
                dist = np.linalg.norm(my_next_pos - neighbor_next_pos)
                safe_dist = self.safe_radius + neighbor['safe_radius'] + 0.5
                if dist < safe_dist:
                    conflict_detected = True
                    conflict_neighbor = neighbor
                    break
            if conflict_detected: break

        if conflict_detected:
            # 简化版优先级：ID小的优先
            if conflict_neighbor is not None and self.id < conflict_neighbor['id']:
                self.wait_steps = 0
                return

            self.wait_steps += 1
            if self.wait_steps > 3:
                parking_spot = self._find_parking_spot(static_map, neighbors_info, conflict_neighbor)
                if parking_spot is not None:
                    self.is_yielding = True
                    self.total_yield_count += 1  # 记录发生了一次避让死锁处理
                    self.yield_timer = max(3, self.horizon)
                    self.wait_steps = 0

                    if self.mode == 'astar':
                        # A* 模式：直线去停车点
                        self.current_path = []
                        dir_to_park = parking_spot - self.pos
                        dist = np.linalg.norm(dir_to_park)
                        dir_norm = dir_to_park / dist if dist > 0 else np.array([0.0, 0.0])
                        for i in range(self.horizon):
                            if i < self.yield_timer:
                                self.current_path.append(self.pos + dir_norm * min(self.max_speed * (i + 1), dist))
                            else:
                                self.current_path.append(parking_spot.copy())
                    else:
                        # APF / Hybrid 模式：使用 APF 规划去停车点
                        start_grid = self._clip_to_grid(self.pos, static_map)
                        target_grid = self._clip_to_grid(parking_spot, static_map)
                        full_path = apf(start_grid, target_grid, static_map.grid_map, dynamic_obstacles=neighbors_info, own_safe_radius=self.safe_radius, own_id=self.id)
                        if not self._fill_current_path_from_grid_path(full_path, start_grid):
                            self.current_path = [self.pos.copy() for _ in range(self.horizon)]
                else:
                    self.current_path = [self.pos.copy() for _ in range(self.horizon)]
            else:
                self.current_path = [self.pos.copy() for _ in range(self.horizon)]
        else:
            self.wait_steps = 0

    def _find_parking_spot(self, static_map, neighbors_info, conflict_neighbor):
        # 复用你提供的简单 BFS 找停车位逻辑 (精简版)
        forbidden_grids = set()
        for neighbor in neighbors_info:
            forbidden_grids.add((int(round(neighbor['pos'][0])), int(round(neighbor['pos'][1]))))
            for p in neighbor['path']:
                forbidden_grids.add((int(round(p[0])), int(round(p[1]))))

        start_grid = self._clip_to_grid(self.pos, static_map)
        queue = [start_grid]
        visited = {start_grid}
        candidates = []
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        search_count = 0
        while queue and search_count < 150:
            curr = queue.pop(0)
            search_count += 1
            if 0 <= curr[0] < static_map.width and 0 <= curr[1] < static_map.height:
                if curr not in forbidden_grids and static_map.grid_map[curr[1]][curr[0]] == 0 and curr != start_grid:
                    candidates.append(curr)
                    if len(candidates) >= 15: break
            for dx, dy in motions:
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < static_map.width and 0 <= ny < static_map.height and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        if not candidates: return None
        # 简单取距离当前点最近的安全点作为候选
        best_spot = min(candidates, key=lambda c: math.hypot(c[0] - self.pos[0], c[1] - self.pos[1]))
        return np.array([float(best_spot[0]), float(best_spot[1])])

    def step_forward(self):
        self.update_mission_status()
        if self.is_reached: return
        if self.current_path:
            self.history.append(self.pos.copy())
            if len(self.history) > 12: self.history.pop(0)
            next_step = self.current_path.pop(0)
            
            delta = next_step - self.pos
            dist = np.linalg.norm(delta)
            if dist > self.max_speed and dist > 1e-9:
                next_step = self.pos + delta / dist * self.max_speed
            
            self.velocity = next_step - self.pos
            self.pos = next_step
        self.update_mission_status()

# ==========================================
# 无头模式单次仿真运行
# ==========================================
def run_headless_simulation(mode, seed, width=50, height=50, max_logical_steps=400, coverage_radius=3.0):
    test_circles, test_rectangles, _ = generate_test_map(width=width, height=height, num_obstacles=40, seed=seed)
    env_map = StaticMap(width, height, circles=test_circles, rectangles=test_rectangles, inflation_radius=0.8)

    uav_tasks = [
        ([2, 4], [width - 2, height - 4]), ([4, 2], [width - 4, height - 2]),
        ([2, height - 4], [width - 2, 4]), ([4, height - 2], [width - 4, 2]),
        ([width - 2, 4], [2, height - 4]), ([width - 4, 2], [4, height - 2]),
        ([width - 2, height - 4], [2, 4]), ([width - 4, height - 2], [4, 2]),
    ]
    
    inspection_regions = make_center_inspection_region(width, height, 30)
    uav_starts = [start for start, _ in uav_tasks]
    inspection_waypoints, _, _ = generate_reachable_inspection_waypoints(env_map, uav_starts, inspection_regions, coverage_radius)
    waypoint_assignments, _ = assign_waypoints_to_uavs(env_map, inspection_waypoints, uav_tasks)
    inspection_map = InspectionMap(env_map, inspection_regions, coverage_radius)

    uavs = [
        EvalUAV(
            uav_id=i, start=start, goal=goal, safe_radius=0.8, comm_range=7.0, horizon=5,
            waypoints=waypoint_assignments[i], coverage_radius=coverage_radius, mode=mode
        )
        for i, (start, goal) in enumerate(uav_tasks, start=1)
    ]

    for uav in uavs:
        inspection_map.update_coverage(uav.pos)

    # 开始逻辑循环
    for t in range(max_logical_steps):
        neighbor_infos = {uav.id: uav.communicate(uavs) for uav in uavs}
        
        for uav in uavs:
            uav.plan_reference_path(env_map, neighbor_infos[uav.id], logical_step=t)

        for uav in uavs:
            neighbors_info = uav.communicate(uavs)
            uav.resolve_conflicts_and_replan(neighbors_info, env_map)

        for uav in uavs:
            uav.step_forward()
            inspection_map.update_coverage(uav.pos)

        if all(uav.is_reached for uav in uavs):
            return {
                "success": True,
                "steps": t,
                "yields": sum(u.total_yield_count for u in uavs),
                "coverage": inspection_map.coverage_ratio()
            }

    # 失败/死局
    return {
        "success": False,
        "steps": max_logical_steps,
        "yields": sum(u.total_yield_count for u in uavs),
        "coverage": inspection_map.coverage_ratio()
    }

# ==========================================
# 主程序：批量测试与可视化绘图
# ==========================================
if __name__ == "__main__":
    NUM_MAPS = 100  # 测试地图数量
    MODES = ['astar', 'apf', 'hybrid']
    
    results = {mode: {'success': 0, 'steps': [], 'yields': [], 'coverages': []} for mode in MODES}
    
    print(f"🚀 开始多算法基准测试... 共跑 {NUM_MAPS} 张地图。 (这可能需要几分钟，请耐心等待)")
    start_time = time.time()

    for seed in range(1, NUM_MAPS + 1):
        if seed % 10 == 0:
            print(f"正在测试地图 {seed}/{NUM_MAPS} ...")
        
        for mode in MODES:
            res = run_headless_simulation(mode=mode, seed=seed)
            if res["success"]:
                results[mode]['success'] += 1
                results[mode]['steps'].append(res["steps"])
            
            # 不论成功与否，记录避让次数和覆盖率
            results[mode]['yields'].append(res["yields"])
            results[mode]['coverages'].append(res["coverage"])

    total_time = time.time() - start_time
    print(f"✅ 测试完成！耗时: {total_time:.1f} 秒")

    # ==========================
    # 绘制统计图表
    # ==========================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-UAV Inspection Planning Benchmark (100 Random Maps)', fontsize=16)

    # 颜色配置
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    labels = ['Pure A*', 'Dynamic APF', 'Hybrid A*+APF']

    # 1. 成功率 (柱状图)
    ax = axes[0, 0]
    success_rates = [results[m]['success'] for m in MODES]
    ax.bar(labels, success_rates, color=colors, edgecolor='black')
    ax.set_title('Success Rate (No Unresolvable Deadlocks)')
    ax.set_ylabel('Successful Runs (out of 100)')
    ax.set_ylim(0, 105)
    for i, v in enumerate(success_rates):
        ax.text(i, v + 2, f"{v}%", ha='center', fontweight='bold')

    # 2. 死锁处理/避让次数 (箱线图)
    ax = axes[0, 1]
    yield_data = [results[m]['yields'] for m in MODES]
    bplot1 = ax.boxplot(yield_data, patch_artist=True, labels=labels)
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Deadlock Resolution Triggers (is_yielding calls)')
    ax.set_ylabel('Total Yield Count per Run')

    # 3. 任务完成步长 (箱线图，仅统计成功局)
    ax = axes[1, 0]
    step_data = [results[m]['steps'] if results[m]['steps'] else [0] for m in MODES]
    bplot2 = ax.boxplot(step_data, patch_artist=True, labels=labels)
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Total Logical Steps (Makespan) - Successful runs only')
    ax.set_ylabel('Steps')

    # 4. 覆盖率 (箱线图)
    ax = axes[1, 1]
    cov_data = [[c * 100 for c in results[m]['coverages']] for m in MODES]
    bplot3 = ax.boxplot(cov_data, patch_artist=True, labels=labels)
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title('Final Inspection Coverage Ratio (%)')
    ax.set_ylabel('Coverage %')
    ax.set_ylim(0, 105)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()