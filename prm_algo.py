import numpy as np
import heapq

from python_motion_planning.common import TYPES


class PRM:
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