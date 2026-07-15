import math
import argparse
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import splprep, splev

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
        self.inflation_radius = inflation_radius  # obstacle inflation / safety margin

        self.grid_map = self._generate_grid()

    def _generate_grid(self):
        grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                # Circle obstacles
                for obs_x, obs_y, obs_r in self.circles:
                    if math.hypot(x - obs_x, y - obs_y) <= (obs_r + self.inflation_radius):
                        grid[y][x] = 1
                        break

                # Rectangle obstacles
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

    motions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

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
    static_map,
    uav_starts,
    inspection_regions,
    coverage_radius=3.0,
    waypoint_spacing=None,
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
        candidate_ids = [
            i for i, reachable in enumerate(per_uav_reachable)
            if reachable[wy, wx]
        ]

        if not candidate_ids:
            unassigned.append(wp)
            continue

        # Add load to balance weight
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
        
        self.path_length = 0.0      # Record total path length

        self.wait_steps = 0         
        self.is_yielding = False    
        self.yield_timer = 0
        self.history = []        

        self.full_path = [self.pos.copy()] # Record final path

    def get_distance(self, other_pos):
        return np.linalg.norm(self.pos - other_pos)

    def get_current_target(self):
        self.update_mission_status()
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        
        # Back to the goal point after completing the mission
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

        # Arrive near the unified end point and judge the end directly.
        if self.inspection_finished and np.linalg.norm(self.pos - self.goal) < 0.5:
            self.is_reached = True

    # ==========================================
    # Reference path generation (A*)
    # ==========================================
    def plan_reference_path(self, static_map):
        self.update_mission_status()

        if self.is_reached:
            return

        if self.is_yielding:
            return

        target = self.get_current_target()

        start_grid = (int(round(self.pos[0])), int(round(self.pos[1])))
        goal_grid = (int(round(target[0])), int(round(target[1])))

        start_grid = (max(0, min(static_map.width - 1, start_grid[0])),
                      max(0, min(static_map.height - 1, start_grid[1])))
        goal_grid = (max(0, min(static_map.width - 1, goal_grid[0])),
                     max(0, min(static_map.height - 1, goal_grid[1])))

        full_path = astar(start_grid, goal_grid, static_map.grid_map)

        self.current_path = []

        if not full_path:
            self.current_path = [self.pos.copy() for _ in range(self.horizon)]
            return

        if len(full_path) > 1 and full_path[0] == start_grid:
            full_path = full_path[1:]

        for i in range(self.horizon):
            if i < len(full_path):
                next_pos = np.array([full_path[i][0], full_path[i][1]], dtype=float)
                self.current_path.append(next_pos)
            else:
                self.current_path.append(target.copy())

    # ==========================================
    # Exchange message with neighbors
    # ==========================================
    def communicate(self, all_uavs):
        neighbors_info = []
        for other in all_uavs:
            if other.id != self.id and self.get_distance(other.pos) <= self.comm_range:
                neighbors_info.append({
                    'id': other.id,
                    'pos': other.pos,
                    'path': other.current_path,
                    'safe_radius': other.safe_radius,
                    'is_reached': other.is_reached
                })
        return neighbors_info

    # ==========================================
    # Conflict detection and path replanning
    # ==========================================
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
                # Ignore landed neighbors
                if neighbor.get('is_reached', False):
                    continue

                if step < len(neighbor['path']):
                    neighbor_next_pos = neighbor['path'][step]
                else:
                    neighbor_next_pos = neighbor['pos'] if not neighbor['path'] else neighbor['path'][-1]

                dist = np.linalg.norm(my_next_pos - neighbor_next_pos)
                safe_dist = self.safe_radius + neighbor['safe_radius'] + 0.5

                if dist < safe_dist:
                    conflict_detected = True
                    conflict_neighbor = neighbor
                    break
            if conflict_detected:
                break

        if conflict_detected:
            self.wait_steps += 1

            if self.wait_steps > 3:
                # Small ID has higher priority
                if self.id > conflict_neighbor['id']:
                    parking_spot = self._find_parking_spot(static_map, neighbors_info, conflict_neighbor)

                    if parking_spot is not None:
                        self.is_yielding = True
                        self.yield_timer = 3  
                        self.wait_steps = 0

                        self.current_path = []
                        dir_to_park = parking_spot - self.pos
                        dist = np.linalg.norm(dir_to_park)

                        if dist > 0:
                            dir_norm = dir_to_park / dist
                        else:
                            dir_norm = np.array([0.0, 0.0])

                        for i in range(self.horizon):
                            if i < self.yield_timer:
                                step_dist = min(self.max_speed * (i + 1), dist)
                                self.current_path.append(self.pos + dir_norm * step_dist)
                            else:
                                self.current_path.append(parking_spot.copy())

                        print(f"[Deadlock Resolved] UAV {self.id} found a parking spot, yielding to UAV {conflict_neighbor['id']}!")
                        return True
                    else:
                        self.current_path = [self.pos.copy() for _ in range(self.horizon)]
                        print(f"[Warning] UAV {self.id} is trapped and cannot yield!")
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
            # Ignore landed neighbors when finding parking place
            if neighbor.get('is_reached', False):
                continue
                
            forbidden_grids.add((int(round(neighbor['pos'][0])), int(round(neighbor['pos'][1]))))
            for p in neighbor['path']:
                px, py = int(round(p[0])), int(round(p[1]))
                forbidden_grids.add((px, py))
                if neighbor['id'] == conflict_neighbor['id']:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        forbidden_grids.add((px + dx, py + dy))

        history_grids = set()
        for p in self.history:
            history_grids.add((int(round(p[0])), int(round(p[1]))))

        start_grid = (int(round(self.pos[0])), int(round(self.pos[1])))
        queue = [start_grid]
        visited = {start_grid}
        motions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        candidates = []
        search_count = 0

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
                if 0 <= nx < static_map.width and 0 <= ny < static_map.height:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        if not candidates:
            return None

        best_spot = None
        max_score = -float('inf')
        conf_pos = conflict_neighbor['pos']

        for spot in candidates:
            dist_to_conflict = math.hypot(spot[0] - conf_pos[0], spot[1] - conf_pos[1])

            min_dist_to_uavs = float('inf')
            if not neighbors_info:
                min_dist_to_uavs = 0.0
            for neighbor in neighbors_info:
                if neighbor.get('is_reached', False):
                    continue
                d = math.hypot(spot[0] - neighbor['pos'][0], spot[1] - neighbor['pos'][1])
                if d < min_dist_to_uavs:
                    min_dist_to_uavs = d

            min_dist_to_obs = float('inf')
            for cx, cy, r in static_map.circles:
                d = math.hypot(spot[0] - cx, spot[1] - cy) - r
                if d < min_dist_to_obs:
                    min_dist_to_obs = d

            for rx, ry, rw, rh in static_map.rectangles:
                dx = max(rx - spot[0], 0, spot[0] - (rx + rw))
                dy = max(ry - spot[1], 0, spot[1] - (ry + rh))
                d = math.hypot(dx, dy)
                if d < min_dist_to_obs:
                    min_dist_to_obs = d

            if min_dist_to_obs == float('inf'):
                min_dist_to_obs = 0.0

            history_penalty = 100.0 if spot in history_grids else 0.0

            score = (dist_to_conflict * 1.0) + \
                    (min_dist_to_uavs * 1.5) + \
                    (min_dist_to_obs * 2.0) - \
                    history_penalty

            if score > max_score:
                max_score = score
                best_spot = spot

        return np.array([float(best_spot[0]), float(best_spot[1])])

    # ==========================================
    # Next step
    # ==========================================
    def step_forward(self):
        self.update_mission_status()

        if self.is_reached:
            return

        if self.current_path:
            self.history.append(self.pos.copy())
            if len(self.history) > 10:
                self.history.pop(0)

            next_step = self.current_path.pop(0)
            self.velocity = next_step - self.pos
            self.path_length += np.linalg.norm(self.velocity)
            self.pos = next_step
            
            self.full_path.append(self.pos.copy()) 
        else:
            self.velocity = np.array([0.0, 0.0])

        self.update_mission_status()

def smooth_trajectory(path, num_points=300, smooth_factor=3.0):
    """
    B-spline smoothing the discrete grid path

    Args:
        path (list): original coordinates list
        num_points: points generated after smoothing
        smooth_factor: smooth factor s, bigger -> smoother
    """
    if len(path) < 3:
        return path

    # Filter duplicate points
    filtered_path = [path[0]]
    for p in path[1:]:
        if np.linalg.norm(np.array(p) - np.array(filtered_path[-1])) > 0.1:
            filtered_path.append(p)

    k = 3 if len(filtered_path) >= 4 else (1 if len(filtered_path) >= 2 else 0)
    if k < 2:
        return filtered_path

    # Abstract x & y
    x = [p[0] for p in filtered_path]
    y = [p[1] for p in filtered_path]

    tck, u = splprep([x, y], s=smooth_factor, k=k)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck, der=0)

    return list(zip(x_new, y_new))

# ==========================================
# main
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
):
    test_circles, test_rectangles, density = generate_test_map(
        width=width,
        height=height,
        num_obstacles=num_obstacles,
        seed=map_seed,
    )

    print("Map generated successfully.")
    print(f"  width={width}, height={height}")
    print(f"  obstacles={num_obstacles}, density={density:.2%}, seed={map_seed}")
    print(f"  circles={len(test_circles)}, rectangles={len(test_rectangles)}")

    env_map = StaticMap(
        width,
        height,
        circles=test_circles,
        rectangles=test_rectangles,
        inflation_radius=inflation_radius,
    )

    num_uavs = max(1, min(8, num_uavs))  
    
    # Predefine 8 posible positions
    all_uav_tasks = [
        ([2, 2], [2, 2]),                                 
        ([width - 3, 2], [width - 3, 2]),                 
        ([2, 5], [2, 2]),                                 
        ([width - 3, 5], [width - 3, 2]),                 
        ([5, 2], [2, 2]),                                 
        ([width - 6, 2], [width - 3, 2]),                 
        ([5, 5], [2, 2]),                                 
        ([width - 6, 5], [width - 3, 2]),                
    ]

    uav_tasks = all_uav_tasks[:num_uavs]
    
    print(f"  Initializing {num_uavs} UAV(s)...")

    if inspection_regions is None:
        inspection_regions = make_center_inspection_region(width, height, center_region_size)

    uav_starts = [start for start, _ in uav_tasks]

    inspection_waypoints, reachable_mask, used_spacing = generate_reachable_inspection_waypoints(
        static_map=env_map,
        uav_starts=uav_starts,
        inspection_regions=inspection_regions,
        coverage_radius=coverage_radius,
        waypoint_spacing=waypoint_spacing,
    )

    waypoint_assignments, unassigned_waypoints = assign_waypoints_to_uavs(
        static_map=env_map,
        waypoints=inspection_waypoints,
        uav_tasks=uav_tasks,
    )

    inspection_map = InspectionMap(
        static_map=env_map,
        inspection_regions=inspection_regions,
        coverage_radius=coverage_radius,
    )

    print("Inspection task initialized.")
    print(f"  coverage_radius={coverage_radius}")
    print(f"  inspection_regions={inspection_regions}")
    print(f"  waypoint_spacing={used_spacing}")
    print(f"  reachable inspection waypoints={len(inspection_waypoints)}")
    print(f"  unassigned waypoints={len(unassigned_waypoints)}")
    print(f"  required inspection cells={int(np.sum(inspection_map.required))}")

    for uav_id in sorted(waypoint_assignments.keys()):
        print(f"  UAV {uav_id}: assigned {len(waypoint_assignments[uav_id])} waypoints")

    uavs = [
        UAV(
            uav_id=i,
            start=start,
            goal=goal,
            safe_radius=0.8,
            comm_range=7.0,
            horizon=5,
            waypoints=waypoint_assignments[i],
            coverage_radius=coverage_radius,
        )
        for i, (start, goal) in enumerate(uav_tasks, start=1)
    ]

    for uav in uavs:
        inspection_map.update_coverage(uav.pos)

    plt.ion()
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 8))

    total_deadlocks_resolved = 0  
    sim_finished = False

    for t in range(max_logical_steps):
        for uav in uavs:
            uav.plan_reference_path(env_map)

        for uav in uavs:
            neighbors_info = uav.communicate(uavs)
            if uav.resolve_conflicts_and_replan(neighbors_info, env_map):
                total_deadlocks_resolved += 1

        old_positions = {uav.id: uav.pos.copy() for uav in uavs}
        for uav in uavs:
            uav.step_forward()
            inspection_map.update_coverage(uav.pos)

        current_coverage = inspection_map.coverage_ratio()

        # Render
        for f in range(render_frames_per_step):
            ax.clear()

            covered_display = np.full((env_map.height, env_map.width), np.nan)
            covered_display[inspection_map.required & inspection_map.covered] = 1.0
            ax.imshow(
                covered_display,
                origin='lower',
                extent=[0, env_map.width, 0, env_map.height],
                alpha=0.25,
                vmin=0,
                vmax=1,
            )

            for rx, ry, rw, rh in inspection_regions:
                ax.add_patch(
                    patches.Rectangle(
                        (rx, ry), rw, rh,
                        linewidth=2,
                        edgecolor='green',
                        facecolor='none',
                        linestyle='--',
                    )
                )

            for rx, ry, rw, rh in env_map.rectangles:
                ax.add_patch(
                    patches.Rectangle(
                        (rx, ry), rw, rh,
                        linewidth=1,
                        edgecolor='black',
                        facecolor='gray',
                        alpha=0.5,
                    )
                )
            for cx, cy, r in env_map.circles:
                ax.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=0.5))

            if inspection_waypoints:
                wp_x = [wp[0] for wp in inspection_waypoints]
                wp_y = [wp[1] for wp in inspection_waypoints]
                ax.plot(wp_x, wp_y, '.', color='green', markersize=3, alpha=0.45)

            alpha = (f + 1) / render_frames_per_step

            for uav in uavs:
                color = cmap((uav.id - 1) % 20)
                interp_pos = old_positions[uav.id] * (1 - alpha) + uav.pos * alpha
                target = uav.get_current_target()

                ax.add_patch(
                    plt.Circle(
                        (interp_pos[0], interp_pos[1]),
                        uav.coverage_radius,
                        color=color,
                        alpha=0.08,
                    )
                )

                ax.add_patch(
                    plt.Circle(
                        (interp_pos[0], interp_pos[1]),
                        uav.safe_radius,
                        color=color,
                        alpha=0.15,
                    )
                )

                ax.plot(interp_pos[0], interp_pos[1], 'o', color=color, markersize=5)
                ax.text(interp_pos[0] + 0.5, interp_pos[1] + 0.5, f'UAV{uav.id}', fontsize=9)
                ax.plot(uav.goal[0], uav.goal[1], 'x', color=color, markersize=10, linewidth=2)

                if not uav.is_reached:
                    ax.plot(target[0], target[1], '*', color=color, markersize=9)

                if uav.current_path:
                    path_x = [interp_pos[0]] + [p[0] for p in uav.current_path]
                    path_y = [interp_pos[1]] + [p[1] for p in uav.current_path]
                    ax.plot(path_x, path_y, '--', color=color, alpha=0.5)

            ax.set_xlim(0, env_map.width)
            ax.set_ylim(0, env_map.height)
            ax.set_aspect('equal')
            ax.set_title(
                f"Inspection MAPF - Step {t} | Coverage {current_coverage * 100:.1f}% | "
                f"Frame {f + 1}/{render_frames_per_step}"
            )
            ax.grid(True)
            plt.pause(0.01)

        # All arrived?
        if all(uav.is_reached for uav in uavs):
            print(f"\n All drones have completed the inspection task and successfully returned to the starting point (Total steps: {t}).")
            sim_finished = True
            break

    # Output data
    if not sim_finished:
        print(f"\n reached max steps ({max_logical_steps}), simulation is killed. Part of uavs may not complete the task")
        current_coverage = inspection_map.coverage_ratio()

    print(f"Map seed: {map_seed}")
    print(f"Coverage rate: {current_coverage * 100:.2f}%")
    print(f"Number of deadlocks: {total_deadlocks_resolved}")
    print("-" * 30)
    tt_length = 0
    for uav in uavs:
        print(f"UAV {uav.id} path length: {uav.path_length:.2f}")
        tt_length += uav.path_length

    print(f"Total path length: {tt_length:.2f}")

    plt.ioff()
    
    fig_final, ax_final = plt.subplots(figsize=(8, 8))
    
    # Draw static obstacles
    for rx, ry, rw, rh in env_map.rectangles:
        ax_final.add_patch(
            patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5)
        )
    for cx, cy, r in env_map.circles:
        ax_final.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=0.5))
        
    # Draw boundary of inspection region
    for rx, ry, rw, rh in inspection_regions:
        ax_final.add_patch(
            patches.Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
        )

    # Draw complete paths
    for uav in uavs:
        color = cmap((uav.id - 1) % 20)
        if len(uav.full_path) > 1:
            # Smooth
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
    ax_final.set_title("Final Trajectories of All UAVs")
    
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
    )