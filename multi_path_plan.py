import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from a_star import astar

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
            # 指标 A：远离主要冲突无人机
            dist_to_conflict = math.hypot(spot[0] - conf_pos[0], spot[1] - conf_pos[1])
            
            # 指标 B：远离所有其他邻居无人机
            min_dist_to_uavs = float('inf')
            if not neighbors_info:
                min_dist_to_uavs = 0.0
            for neighbor in neighbors_info:
                d = math.hypot(spot[0] - neighbor['pos'][0], spot[1] - neighbor['pos'][1])
                if d < min_dist_to_uavs:
                    min_dist_to_uavs = d
                    
            # 指标 C：远离所有静态障碍物 (分别计算圆形和方形)
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
            ''' 
            Problem found: uav will trend to keep close to map edge
            consider add edge punishment
            '''
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
def run_simulation():
    # Map generated by map_generator.py
    test_circles = [(19.290364740406527, 12.86854850195367, 1.6133577136190191), (36.161246307070044, 3.0196546481233786, 1.0447520705407185), (22.995007811412588, 33.52214335170763, 1.3034347163590334), (23.344234887633338, 25.25637044069046, 2.3488525042443626), (39.50859739919309, 33.68290445644095, 1.741427846637851), (32.92148166537825, 31.539507818932904, 2.1012584856862766), (25.117464692964795, 16.192894115717234, 1.9551690166435696), (21.364163250187882, 39.64650871716653, 1.775442597577451), (30.54156619627903, 6.8776727083999125, 1.9821858163644552), (21.053025423595926, 10.612699357991268, 2.033599510643614), (9.500270299285631, 30.699198778059333, 2.037221827876005), (27.784207831194166, 28.179595762199412, 1.900463086866583), (21.51335553675754, 20.151202873956642, 1.3625080176810163), (46.48495579309042, 20.67448788254814, 2.4664305371855924), (33.12880410453508, 21.815519983502053, 1.5703011872344794), (36.301336159219574, 45.77065936753006, 2.243883817610212), (27.611366915995507, 17.939331274868657, 2.3284717735825846)]  
    test_rectangles = [(28.669519496006192, 15.851942983594224, 2.577847480093891, 3.479348429844897), (31.58202856851029, 32.74571894804405, 2.0354164053377897, 2.0354164053377897), (2.1405641900489916, 27.743934459459542, 3.853897607813321, 2.2886191574089687), (11.270941292077337, 39.44048590934167, 4.6987510791957625, 3.101083495213622), (42.82674923540524, 27.911075968962628, 3.2274402589694873, 4.107617025022407), (8.484105711181876, 37.02876012852835, 3.2329200319503064, 2.498025489922211), (38.321854585147946, 34.077935186432335, 4.321910376370678, 2.9620042669928095), (16.625303264733503, 43.32265722881793, 2.515609328543456, 2.515609328543456), (25.9970330742708, 16.85173384782889, 4.448237405347051, 2.6282089296673092), (20.65163225218532, 44.11264964453834, 2.4267517224145982, 3.473178495383365), (37.636905292835344, 26.41097983733556, 3.203345872409167, 3.203345872409167), (27.358580758034325, 20.197725715892236, 1.793441623758177, 1.793441623758177), (30.703393344274676, 31.84608049263334, 3.438301138516178, 3.438301138516178), (10.913450490989609, 19.303786591326674, 4.026254132840667, 3.7439733850893995), (22.456445215070236, 26.54583112014356, 3.060922478140597, 3.060922478140597), (29.345482708957256, 24.999243058131178, 2.3050735868798387, 2.3050735868798387), (7.237441261234712, 30.089521871284827, 3.838757236835202, 2.7604715363456274), (22.31205740755959, 4.299577926636322, 3.7839081935522394, 3.7839081935522394), (18.47299464956643, 31.542958747805617, 2.077835296753728, 4.598644268350552), (6.612062284983101, 27.57688875616494, 3.752373658287413, 3.752373658287413), (9.908340279629746, 31.442008585644576, 2.709484470806118, 2.709484470806118), (13.959455769094404, 40.500294120653514, 1.6723857623918426, 1.6723857623918426), (15.42573471991298, 23.764507607543386, 2.9946956529450057, 2.9946956529450057), (10.21654005724997, 43.51965589546159, 4.751607327166527, 4.869743920770189), (42.95272987445535, 31.027425233739603, 4.050086792354785, 3.7515648218693474), (43.14740978015769, 35.11599862086139, 2.326049054490092, 2.326049054490092), (23.347669584141183, 8.191714251992652, 4.0396492765370535, 4.303355154892193), (1.1176584269321068, 24.046016922727105, 3.693060777440994, 2.393008892033143), (24.797808895191512, 40.98737879387069, 3.3786302149908645, 3.6787649004711263), (32.67860715203975, 17.9822278884827, 3.190809536554294, 4.633355633223358), (39.84503131502752, 35.47096586120676, 2.2606089754483683, 2.2606089754483683), (21.948858764259384, 45.79833704473109, 2.25586253165926, 2.25586253165926), (11.21302169030966, 25.833838271483057, 3.1344459835606315, 3.1344459835606315)] 

    env_map = StaticMap(50, 50, circles=test_circles, rectangles=test_rectangles, inflation_radius=0.8)    
    # UAV setting
    uavs = [
        UAV(1, start=[2, 2], goal=[48, 48], safe_radius=0.8, comm_range=5.0, horizon=5),
        UAV(2, start=[48, 2], goal=[2, 48], safe_radius=0.8, comm_range=5.0, horizon=5),
        UAV(3, start=[2, 48], goal=[48, 2], safe_radius=0.8, comm_range=5.0, horizon=5),
        UAV(4, start=[48, 48], goal=[2, 2], safe_radius=0.8, comm_range=5.0, horizon=5)
    ]

    plt.ion() 
    fig, ax = plt.subplots(figsize=(8, 8)) 
    
    max_logical_steps = 500
    render_frames_per_step = 5  
    
    for t in range(max_logical_steps):
        for uav in uavs: uav.plan_reference_path(env_map)
        for uav in uavs: uav.resolve_conflicts_and_replan(uav.communicate(uavs), env_map)
        
        # UAV move forward and record history trajectory to visualize
        old_positions = {uav.id: uav.pos.copy() for uav in uavs}    
        for uav in uavs: uav.step_forward()

        # Render
        for f in range(render_frames_per_step):
            ax.clear()
            
            for rx, ry, rw, rh in env_map.rectangles:
                ax.add_patch(patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.5))
            for cx, cy, r in env_map.circles:
                ax.add_patch(plt.Circle((cx, cy), r, color='gray', alpha=0.5))

            # Interpolation fram
            alpha = (f + 1) / render_frames_per_step

            for uav in uavs:
                color = ['red', 'blue', 'green', 'orange'][uav.id - 1]
                
                interp_pos = old_positions[uav.id] * (1 - alpha) + uav.pos * alpha

                # Draw UAVs
                ax.plot(interp_pos[0], interp_pos[1], 'o', color=color, markersize=5)
                ax.add_patch(plt.Circle((interp_pos[0], interp_pos[1]), uav.safe_radius, color=color, alpha=0.15))
                ax.plot(uav.goal[0], uav.goal[1], 'x', color=color, markersize=10, linewidth=2)
                ax.text(interp_pos[0]+0.5, interp_pos[1]+0.5, f'UAV{uav.id}', fontsize=9)
                
                # Future trajectory
                if uav.current_path:
                    path_x = [interp_pos[0]] + [p[0] for p in uav.current_path]
                    path_y = [interp_pos[1]] + [p[1] for p in uav.current_path]
                    ax.plot(path_x, path_y, '--', color=color, alpha=0.5)

            ax.set_xlim(0, env_map.width)
            ax.set_ylim(0, env_map.height)
            ax.set_title(f"Cooperative MAPF - Logic Step {t} | Render Frame {f+1}/{render_frames_per_step}")
            ax.grid(True)
            
            plt.pause(0.01) 
            
        if all(uav.is_reached for uav in uavs):
            print(f"所有无人机已在第 {t} 步到达终点！")
            break
            
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()