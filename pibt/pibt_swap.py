import math
import argparse
import random
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import splprep, splev

# Import from your existing files
from map_generator import generate_test_map
from pibt_core import PIBTStepStats, DistanceTable, neighbors, project_goal_to_reachable


# ==========================================
# 纯净版 PIBT Planner (移除了随机噪声)
# ==========================================
class SafePIBTStepPlanner:
    def __init__(self, free_grid, goals, seed=0, safe_distance=1.5):
        self.grid = np.asarray(free_grid, dtype=bool)
        self.goals = list(goals)
        self.distances = [DistanceTable(self.grid, goal) for goal in goals]
        self.rng = random.Random(seed)
        self.stats = PIBTStepStats()
        self.safe_distance = safe_distance  
        
        self.occupied_now = {}
        self.occupied_next = {}
        self.next_config = []
        self.current_config = []

    def step(self, current, priorities):
        self.current_config = list(current)
        self.next_config = [None] * len(current)
        self.occupied_now = {coord: index for index, coord in enumerate(current)}
        self.occupied_next = {}

        # 按优先级降序分配
        order = sorted(range(len(current)), key=lambda i: (priorities[i], -i), reverse=True)
        for index in order:
            if self.next_config[index] is None:
                self._assign(index)

        result = [coord if coord is not None else current[i] for i, coord in enumerate(self.next_config)]
        self.stats.waits = sum(start == end for start, end in zip(current, result))
        return result, self.stats

    def _assign(self, index: int) -> bool:
        current = self.current_config[index]
        candidates = [current] + neighbors(self.grid, current)
        self.rng.shuffle(candidates) # 仅用于打乱同分项

        # 确定性打分：启发式距离 + 安全距离惩罚 + 原地停留轻微惩罚
        def candidate_score(c):
            base_dist = self.distances[index].get(c)
            penalty = 0.0
            
            for other_idx, next_pos in enumerate(self.next_config):
                if next_pos is not None and other_idx != index:
                    dist = math.hypot(c[0] - next_pos[0], c[1] - next_pos[1])
                    if dist < self.safe_distance:
                        penalty += 0.8  
            
            if c == current:
                penalty += 0.5
            
            return base_dist + penalty

        candidates.sort(key=candidate_score)

        for candidate in candidates:
            if candidate in self.occupied_next:
                continue
            occupant = self.occupied_now.get(candidate)
            
            # 防止直接物理换位 (Swap)
            if occupant is not None and self.next_config[occupant] == current:
                continue

            self.next_config[index] = candidate
            self.occupied_next[candidate] = index
            
            # 优先级继承
            if occupant is not None and occupant != index and self.next_config[occupant] is None:
                self.stats.requests += 1
                self.stats.priority_inheritances += 1
                accepted = self._assign(occupant)
                self.stats.responses += 1
                if not accepted:
                    self.stats.backtracks += 1
                    if self.occupied_next.get(candidate) == index:
                        del self.occupied_next[candidate]
                    self.next_config[index] = None
                    continue
            return True

        self.next_config[index] = current
        self.occupied_next[current] = index
        return False


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


# ==========================================
# Inspection / coverage utilities
# ==========================================
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


def generate_reachable_inspection_waypoints(static_map, uav_starts, inspection_regions, coverage_radius=3.0, waypoint_spacing=None):
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
        best_idx = min(range(len(remaining)), key=lambda i: np.linalg.norm(np.array(remaining[i], dtype=float) - current))
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

        best_i = min(candidate_ids, key=lambda i: np.linalg.norm(np.array(wp, dtype=float) - np.array(starts[i], dtype=float)) + 20.0 * len(assignments[i + 1]))
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


class UAV_PIBT:
    def __init__(self, uav_id, start, goal, safe_radius, coverage_radius=3.0, waypoints=None):
        self.id = uav_id
        self.pos = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        self.safe_radius = safe_radius
        self.coverage_radius = coverage_radius 

        self.waypoints = [(int(wp[0]), int(wp[1])) for wp in (waypoints if waypoints else [])]
        self.current_wp_idx = 0
        self.is_reached = False
        
        self.path_length = 0.0      
        self.full_path = [self.pos]

    def get_current_target(self):
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return self.goal

    def update_mission_status(self):
        while self.current_wp_idx < len(self.waypoints):
            target = self.waypoints[self.current_wp_idx]
            if self.pos == target:
                self.current_wp_idx += 1
            else:
                break
        if self.current_wp_idx >= len(self.waypoints) and self.pos == self.goal:
            self.is_reached = True


def smooth_trajectory(path, num_points=300, smooth_factor=3.0):
    if len(path) < 3:
        return path
    filtered_path = [path[0]]
    for p in path[1:]:
        if np.linalg.norm(np.array(p) - np.array(filtered_path[-1])) > 0.1:
            filtered_path.append(p)
    k = 3 if len(filtered_path) >= 4 else (1 if len(filtered_path) >= 2 else 0)
    if k < 2:
        return filtered_path
    x = [p[0] for p in filtered_path]
    y = [p[1] for p in filtered_path]
    tck, u = splprep([x, y], s=smooth_factor, k=k)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return list(zip(x_new, y_new))


# ==========================================
# main PIBT Simulation
# ==========================================
def run_simulation(
    num_uavs=8,               
    width=50,
    height=50,
    num_obstacles=50,
    map_seed=None,
    inflation_radius=0.8,
    max_logical_steps=500,
    render_frames_per_step=5,
    coverage_radius=3.0,          
    inspection_regions=None,      
    center_region_size=30,        
    waypoint_spacing=None,        
    safe_distance=1.5,            
):
    test_circles, test_rectangles, density = generate_test_map(width=width, height=height, num_obstacles=num_obstacles, seed=map_seed)

    print("Map generated successfully.")
    print(f"  width={width}, height={height}")
    print(f"  obstacles={num_obstacles}, density={density:.2%}, seed={map_seed}")
    print(f"  UAV Safe Distance Margin={safe_distance}")
    
    env_map = StaticMap(width, height, circles=test_circles, rectangles=test_rectangles, inflation_radius=inflation_radius)
    free_grid = (np.array(env_map.grid_map) == 0)

    num_uavs = max(1, min(8, num_uavs))
    
    all_uav_tasks = [
        ([2, 2], [2, 2]), 
        ([width - 3, 2], [width - 3, 2]), 
        ([2, 5], [2, 5]), 
        ([width - 3, 5], [width - 3, 5]),
        ([5, 2], [5, 2]), 
        ([width - 6, 2], [width - 6, 2]), 
        ([5, 5], [5, 5]), 
        ([width - 6, 5], [width - 6, 5]),
    ]
    uav_tasks = all_uav_tasks[:num_uavs]
    
    if inspection_regions is None:
        inspection_regions = make_center_inspection_region(width, height, center_region_size)

    uav_starts = [start for start, _ in uav_tasks]

    inspection_waypoints, _, used_spacing = generate_reachable_inspection_waypoints(
        static_map=env_map, uav_starts=uav_starts, inspection_regions=inspection_regions,
        coverage_radius=coverage_radius, waypoint_spacing=waypoint_spacing,
    )

    waypoint_assignments, unassigned_waypoints = assign_waypoints_to_uavs(
        static_map=env_map, waypoints=inspection_waypoints, uav_tasks=uav_tasks,
    )

    inspection_map = InspectionMap(
        static_map=env_map, inspection_regions=inspection_regions, coverage_radius=coverage_radius,
    )

    uavs = [
        UAV_PIBT(
            uav_id=i, start=start, goal=goal, safe_radius=0.8,
            coverage_radius=coverage_radius, waypoints=waypoint_assignments[i]
        )
        for i, (start, goal) in enumerate(uav_tasks, start=1)
    ]

    for uav in uavs:
        inspection_map.update_coverage(uav.pos)

    plt.ion()
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 8))

    total_resolutions = 0  
    sim_finished = False

    for t in range(max_logical_steps):
        # 1. 状态更新
        for uav in uavs:
            uav.update_mission_status()

        # 2. 【核心修改】检测并执行任务交换 (Task Swapping)
        # 仅当两架无人机靠近，且交换目标能让总体曼哈顿距离下降时执行
        for i in range(len(uavs)):
            for j in range(i + 1, len(uavs)):
                u1 = uavs[i]
                u2 = uavs[j]
                
                # 如果两人都已经完成任务，则忽略
                if u1.is_reached and u2.is_reached:
                    continue
                
                # 曼哈顿距离判断是否处于相遇/死锁边缘的邻域 (距离 <= 4)
                dist_between = abs(u1.pos[0] - u2.pos[0]) + abs(u1.pos[1] - u2.pos[1])
                if dist_between <= 4:
                    t1 = u1.get_current_target()
                    t2 = u2.get_current_target()
                    
                    # 当前前往各自目标的距离之和
                    dist_orig = abs(u1.pos[0] - t1[0]) + abs(u1.pos[1] - t1[1]) + \
                                abs(u2.pos[0] - t2[0]) + abs(u2.pos[1] - t2[1])
                    
                    # 如果交换目标后的距离之和更小
                    dist_swap = abs(u1.pos[0] - t2[0]) + abs(u1.pos[1] - t2[1]) + \
                                abs(u2.pos[0] - t1[0]) + abs(u2.pos[1] - t1[1])
                                
                    if dist_swap < dist_orig:
                        # 执行全局任务交换 (剩余 Waypoints 和 最终 Goal)
                        rem_wp1 = u1.waypoints[u1.current_wp_idx:]
                        rem_wp2 = u2.waypoints[u2.current_wp_idx:]
                        
                        u1.waypoints = rem_wp2
                        u2.waypoints = rem_wp1
                        u1.current_wp_idx = 0
                        u2.current_wp_idx = 0
                        
                        u1.goal, u2.goal = u2.goal, u1.goal
                        
                        # 重置完成状态并刷新
                        u1.is_reached = False
                        u2.is_reached = False
                        u1.update_mission_status()
                        u2.update_mission_status()
                        
                        print(f"[Step {t}] Task Swapped between UAV {u1.id} and UAV {u2.id}")

        # 3. 构建 PIBT 输入
        current_positions = []
        targets = []
        priorities = []

        for uav in uavs:
            current_positions.append(uav.pos)

            if uav.is_reached:
                target = uav.pos
                priority = 0.0
            else:
                target_wp = uav.get_current_target()
                target, _ = project_goal_to_reachable(free_grid, uav.pos, target_wp)
                
                # 移除了所有的噪声。纯确定性的优先级。
                priority = float(abs(uav.pos[0] - target[0]) + abs(uav.pos[1] - target[1]))
                
            targets.append(target)
            priorities.append(priority)

        if all(uav.is_reached for uav in uavs):
            print(f"\n All UAVs have completed the inspection task and successfully returned to the starting point. (Total steps: {t})")
            sim_finished = True
            break

        # 执行单步确定性 PIBT
        pibt = SafePIBTStepPlanner(free_grid, targets, seed=map_seed + t if map_seed else t, safe_distance=safe_distance)
        next_positions, stats = pibt.step(current_positions, priorities)
        total_resolutions += stats.backtracks
        
        old_positions = {uav.id: np.array(uav.pos, dtype=float) for uav in uavs}

        for i, uav in enumerate(uavs):
            if uav.pos != next_positions[i]:
                uav.path_length += np.linalg.norm(np.array(uav.pos) - np.array(next_positions[i]))
                uav.pos = next_positions[i]
                uav.full_path.append(uav.pos)
            inspection_map.update_coverage(uav.pos)

        current_coverage = inspection_map.coverage_ratio()

        for f in range(render_frames_per_step):
            ax.clear()

            covered_display = np.full((env_map.height, env_map.width), np.nan)
            covered_display[inspection_map.required & inspection_map.covered] = 1.0
            ax.imshow(covered_display, origin='lower', extent=[0, env_map.width, 0, env_map.height], alpha=0.25, vmin=0, vmax=1)

            for rx, ry, rw, rh in inspection_regions:
                ax.add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='green', facecolor='none', linestyle='--'))

            for rx, ry, rw, rh in env_map.rectangles:
                ax.add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5))
            for cx, cy, r in env_map.circles:
                ax.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=0.5))

            if inspection_waypoints:
                wp_x = [wp[0] for wp in inspection_waypoints]
                wp_y = [wp[1] for wp in inspection_waypoints]
                ax.plot(wp_x, wp_y, '.', color='green', markersize=3, alpha=0.45)

            alpha = (f + 1) / render_frames_per_step

            for uav in uavs:
                color = cmap((uav.id - 1) % 20)
                interp_pos = old_positions[uav.id] * (1 - alpha) + np.array(uav.pos, dtype=float) * alpha
                target = uav.get_current_target()

                ax.add_patch(plt.Circle((interp_pos[0], interp_pos[1]), uav.coverage_radius, color=color, alpha=0.08))
                ax.add_patch(plt.Circle((interp_pos[0], interp_pos[1]), uav.safe_radius, color=color, alpha=0.15))
                ax.plot(interp_pos[0], interp_pos[1], 'o', color=color, markersize=5)
                ax.text(interp_pos[0] + 0.5, interp_pos[1] + 0.5, f'UAV{uav.id}', fontsize=9)
                ax.plot(uav.goal[0], uav.goal[1], 'x', color=color, markersize=10, linewidth=2)

                if not uav.is_reached:
                    ax.plot(target[0], target[1], '*', color=color, markersize=9)

            ax.set_xlim(0, env_map.width)
            ax.set_ylim(0, env_map.height)
            ax.set_aspect('equal')
            ax.set_title(f"Safe PIBT MAPF - Step {t} | Coverage {current_coverage * 100:.1f}% | Frame {f + 1}/{render_frames_per_step}")
            ax.grid(True)
            plt.pause(0.01)

    if not sim_finished:
        print(f"\n⚠️ Maximum steps reached ({max_logical_steps}) simulation end. Part of UAVs may not complete the task")
        current_coverage = inspection_map.coverage_ratio()

    print(f"Map seed：{map_seed}")
    print(f"Coverage rate：{current_coverage * 100:.2f}%")
    print(f"Total resolutions of PIBT backtrack: {total_resolutions}")
    print("-" * 30)
    tt_length = 0
    for uav in uavs:
        print(f"UAV {uav.id}  {uav.path_length:.2f}")
        tt_length += uav.path_length

    print(f"Total path length: {tt_length:.2f}")

    plt.ioff()
    
    fig_final, ax_final = plt.subplots(figsize=(8, 8))
    for rx, ry, rw, rh in env_map.rectangles:
        ax_final.add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5))
    for cx, cy, r in env_map.circles:
        ax_final.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=0.5))
    for rx, ry, rw, rh in inspection_regions:
        ax_final.add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='green', facecolor='none', linestyle='--'))

    for uav in uavs:
        color = cmap((uav.id - 1) % 20)
        if len(uav.full_path) > 1:
            smoothed_path = smooth_trajectory(uav.full_path, num_points=300, smooth_factor=5.0)
            spx = [p[0] for p in smoothed_path]
            spy = [p[1] for p in smoothed_path]
            px = [p[0] for p in uav.full_path]
            py = [p[1] for p in uav.full_path]

            ax_final.plot(spx, spy, '-', color=color, linewidth=2, label=f'UAV {uav.id}')
            ax_final.plot(px[0], py[0], 'o', color=color, markersize=6)
            ax_final.plot(px[-1], py[-1], 'x', color=color, markersize=8)

    ax_final.set_xlim(0, env_map.width)
    ax_final.set_ylim(0, env_map.height)
    ax_final.set_aspect('equal')
    ax_final.set_title("Final Trajectories of All UAVs (Safe PIBT with Task Swapping)")
    ax_final.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax_final.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_uavs", type=int, default=5, help="Number of UAVs")
    parser.add_argument("--width", type=int, default=50, help="Map width")
    parser.add_argument("--height", type=int, default=50, help="Map height")
    parser.add_argument("--ob", "--num_obstacles", type=int, default=50, help="Number of obstacles")
    parser.add_argument("--map_seed", type=int, default=None, help="Random seed for map")
    parser.add_argument("--ir", "--inflation_radius", type=float, default=0.8, help="Safe radius for obstacles")
    parser.add_argument("--steps", "--max_steps", type=int, default=500, help="Max logical steps")
    parser.add_argument("--rf", "--render_frames", type=int, default=5, help="Render frame per second")
    parser.add_argument("--cr", "--coverage_radius", type=float, default=3.0, help="Coverage radius")
    parser.add_argument("--crs", "--center_region_size", type=float, default=30, help="Center region size")
    parser.add_argument("--ins_region", "--inspection_region", type=list[tuple[int, int, int, int]], default=None, help="Inspection region")
    parser.add_argument("--ws", "--waypoint_spacing", type=int, default=None, help="Waypoint spacing")
    parser.add_argument("--sd", "--safe_distance", type=float, default=1.5, help="Safe distance between UAVs")

    args = parser.parse_args()

    run_simulation(
        num_uavs=args.num_uavs,              
        width=args.width,
        height=args.height,
        num_obstacles=args.ob,
        map_seed=args.map_seed,              
        inflation_radius=args.ir,
        max_logical_steps=args.steps,   
        render_frames_per_step=args.rf,
        coverage_radius=args.cr,      
        center_region_size=args.crs,    
        inspection_regions=args.ins_region,  
        waypoint_spacing=args.ws,
        safe_distance=args.sd,  
    )