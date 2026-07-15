import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from a_star import astar
from map_generator import generate_test_map

# ==========================================
# Static map and UAV parameters
# ==========================================
import math

class StaticMap:
    def __init__(self, width, height, circles=None, rectangles=None, inflation_radius=0.0):
        self.width = width
        self.height = height
        self.circles = circles if circles else []
        self.rectangles = rectangles if rectangles else []
        self.inflation_radius = inflation_radius # Safe radius
        
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

class UAV:
    def __init__(self, uav_id, start, goal, safe_radius, comm_range, horizon):
        self.id = uav_id
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.safe_radius = safe_radius
        self.comm_range = comm_range
        self.horizon = horizon  # plan window (H steps)
        
        self.current_path = []  # future H steps
        self.velocity = np.array([0.0, 0.0])
        self.is_reached = False
        self.max_speed = 1.0

        self.wait_steps = 0         # Record wait time 
        self.is_yielding = False    # Record motion status
        self.yield_timer = 0      
        self.history = []           # Record history trajectory

    def get_distance(self, other_pos):
        return np.linalg.norm(self.pos - other_pos)

    # ==========================================
    # Reference path generation (A*), can be 
    # changed to other algorithm (Not implemented)
    # ==========================================
    def plan_reference_path(self, static_map):

        if self.is_reached:
            return

        if self.is_yielding:
            return 
        
        # Transfer start and goal point to grid map
        start_grid = (int(round(self.pos[0])), int(round(self.pos[1])))
        goal_grid = (int(round(self.goal[0])), int(round(self.goal[1])))
        
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
                self.current_path.append(self.goal.copy())

    # ==========================================
    # Exchange message with neighbors
    # ==========================================
    def communicate(self, all_uavs):
        neighbors_info = []
        for other in all_uavs:
            if other.id != self.id and self.get_distance(other.pos) <= self.comm_range:
                # Record neighbors' future H steps and current status
                neighbors_info.append({
                    'id': other.id,
                    'pos': other.pos,
                    'path': other.current_path,
                    'safe_radius': other.safe_radius
                })
        return neighbors_info

    # ==========================================
    # Conflict detection and path replanning
    # ==========================================
    def resolve_conflicts_and_replan(self, neighbors_info, static_map):
        if self.is_reached or not self.current_path:
            return

        if self.is_yielding:
            self.yield_timer -= 1
            if self.yield_timer <= 0:
                self.is_yielding = False 
            return

        conflict_detected = False
        conflict_neighbor = None 
        
        for step, my_next_pos in enumerate(self.current_path):
            for neighbor in neighbors_info:
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
                        self.yield_timer = 3  # yielding time
                        self.wait_steps = 0
                        
                        # Generate way to the parking place
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
                    else:
                        # if no parking place is found
                        self.current_path = [self.pos.copy() for _ in range(self.horizon)]
                        print(f"[Warning] UAV {self.id} is trapped and cannot yield!")
                else:
                    self.current_path = [self.pos.copy() for _ in range(self.horizon)]
            else:
                self.current_path = [self.pos.copy() for _ in range(self.horizon)]
        else:
            self.wait_steps = 0

    def _find_parking_spot(self, static_map, neighbors_info, conflict_neighbor):
        forbidden_grids = set()
        
        for neighbor in neighbors_info:
            forbidden_grids.add((int(round(neighbor['pos'][0])), int(round(neighbor['pos'][1]))))
            for p in neighbor['path']:
                px, py = int(round(p[0])), int(round(p[1]))
                forbidden_grids.add((px, py))
                # expand trajectory
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
        
        # BFS 
        while queue and search_count < 150: 
            curr = queue.pop(0)
            search_count += 1
            if 0 <= curr[0] < static_map.width and 0 <= curr[1] < static_map.height:
                if curr not in forbidden_grids and static_map.grid_map[curr[1]][curr[0]] == 0:
                    if curr != start_grid: 
                        candidates.append(curr)
                        if len(candidates) >= 15: break
            
            for m in motions:
                nx, ny = curr[0] + m[0], curr[1] + m[1]
                if 0 <= nx < static_map.width and 0 <= ny < static_map.height:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
                        
        if not candidates: return None 
            
        # Punish
        best_spot = None
        max_score = -float('inf')
        conf_pos = conflict_neighbor['pos']
        
        for spot in candidates:
            # Target A: stay away from conflict uavs
            dist_to_conflict = math.hypot(spot[0] - conf_pos[0], spot[1] - conf_pos[1])
            
            # Target B: stay away from other uavs
            min_dist_to_uavs = float('inf')
            if not neighbors_info:
                min_dist_to_uavs = 0.0
            for neighbor in neighbors_info:
                d = math.hypot(spot[0] - neighbor['pos'][0], spot[1] - neighbor['pos'][1])
                if d < min_dist_to_uavs:
                    min_dist_to_uavs = d
                    
            # Target C：stay away from all static obstacles
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
            
            # Give higher weight to UAVs and obstacles, make it tend to stop 
            # in the middle of open area 
            score = (dist_to_conflict * 1.0) + \
                    (min_dist_to_uavs * 1.5) + \
                    (min_dist_to_obs * 2.0) - \
                    history_penalty
            
            if score > max_score:
                max_score = score
                best_spot = spot
                
        return np.array([float(best_spot[0]), float(best_spot[1])])
    
    # ==========================================
    # Next
    # ==========================================
    def step_forward(self):
        if np.linalg.norm(self.pos - self.goal) < 0.1:
            self.is_reached = True
            return

        if self.current_path:
            self.history.append(self.pos.copy())
            if len(self.history) > 10:
                self.history.pop(0)
                
            next_step = self.current_path.pop(0)
            self.velocity = next_step - self.pos
            self.pos = next_step


# ==========================================
# main
# ==========================================
def run_simulation(
    width=50,
    height=50,
    num_obstacles=50,
    map_seed=42,
    inflation_radius=0.8,
    max_logical_steps=500,
    render_frames_per_step=5,
):
    """
    Run multi-UAV path planning simulation.

    The map is generated directly by map_generator.generate_test_map(),
    so there is no need to manually copy circles/rectangles anymore.
    """
    # Generate map directly from map_generator.py
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

    # UAV setting. The four corner safe zones in map_generator.py are reserved for these start/goal points.
    uavs = [
        UAV(1, start=[2, 2], goal=[width - 2, height - 2], safe_radius=0.8, comm_range=5.0, horizon=5),
        UAV(2, start=[width - 2, 2], goal=[2, height - 2], safe_radius=0.8, comm_range=5.0, horizon=5),
        UAV(3, start=[2, height - 2], goal=[width - 2, 2], safe_radius=0.8, comm_range=5.0, horizon=5),
        UAV(4, start=[width - 2, height - 2], goal=[2, 2], safe_radius=0.8, comm_range=5.0, horizon=5),
    ]

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    for t in range(max_logical_steps):
        for uav in uavs:
            uav.plan_reference_path(env_map)

        for uav in uavs:
            neighbors_info = uav.communicate(uavs)
            uav.resolve_conflicts_and_replan(neighbors_info, env_map)

        # UAV move forward and record history trajectory to visualize
        old_positions = {uav.id: uav.pos.copy() for uav in uavs}
        for uav in uavs:
            uav.step_forward()

        # Render
        for f in range(render_frames_per_step):
            ax.clear()

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

            # Interpolation frame
            alpha = (f + 1) / render_frames_per_step

            for uav in uavs:
                color = ['red', 'blue', 'green', 'orange'][uav.id - 1]
                interp_pos = old_positions[uav.id] * (1 - alpha) + uav.pos * alpha

                # Draw UAVs
                ax.plot(interp_pos[0], interp_pos[1], 'o', color=color, markersize=5)
                ax.add_patch(plt.Circle((interp_pos[0], interp_pos[1]), uav.safe_radius, color=color, alpha=0.15))
                ax.plot(uav.goal[0], uav.goal[1], 'x', color=color, markersize=10, linewidth=2)
                ax.text(interp_pos[0] + 0.5, interp_pos[1] + 0.5, f'UAV{uav.id}', fontsize=9)

                # Future trajectory
                if uav.current_path:
                    path_x = [interp_pos[0]] + [p[0] for p in uav.current_path]
                    path_y = [interp_pos[1]] + [p[1] for p in uav.current_path]
                    ax.plot(path_x, path_y, '--', color=color, alpha=0.5)

            ax.set_xlim(0, env_map.width)
            ax.set_ylim(0, env_map.height)
            ax.set_aspect('equal')
            ax.set_title(f"Cooperative MAPF - Logic Step {t} | Render Frame {f + 1}/{render_frames_per_step}")
            ax.grid(True)

            plt.pause(0.01)

        if all(uav.is_reached for uav in uavs):
            print(f"All uavs have reached their destination at step {t}.")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_simulation(
        width=50,
        height=50,
        num_obstacles=50,
        map_seed=None,        
        inflation_radius=0.8,
        max_logical_steps=500,
        render_frames_per_step=5,
    )
