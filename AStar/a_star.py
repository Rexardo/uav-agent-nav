import heapq
import numpy as np

class Node:
    """Node class"""
    def __init__(self, x, y, cost=0.0, parent=None):
        self.x = x
        self.y = y
        self.g = 0.0  
        self.h = 0.0  
        self.f = 0.0  
        self.parent = parent 

    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def heuristic(node1, node2):
    """
    Heuristic Function
    """
    return np.hypot(node1.x - node2.x, node1.y - node2.y)

def astar(start_pos, goal_pos, grid_map):
    """
    A* main function

    Args:
        start_pos (Node): start point
        goal_pos (Node): goal point
        grid_map (list): 2d map, 0 is road, 1 is obstacle

    Returns:
        List of Node: A list containing Nodes:
            [Node]
    """
    start_node = Node(start_pos[0], start_pos[1])
    goal_node = Node(goal_pos[0], goal_pos[1])

    open_list = []
    heapq.heappush(open_list, start_node)
    
    closed_set = set()
    
    g_score = { (start_node.x, start_node.y): 0.0 }

    motions = [
        (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
    ]

    map_width = len(grid_map[0])
    map_height = len(grid_map)

    while open_list:
        # Pop node
        current_node = heapq.heappop(open_list)

        # If reach goal point
        if current_node.x == goal_node.x and current_node.y == goal_node.y:
            path = []
            while current_node is not None:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1] 

        # Record node
        closed_set.add((current_node.x, current_node.y))

        # Visit neighbor
        for motion in motions:
            neighbor_x = current_node.x + motion[0]
            neighbor_y = current_node.y + motion[1]
            cost = motion[2]

            if not (0 <= neighbor_x < map_width and 0 <= neighbor_y < map_height):
                continue
            
            if grid_map[neighbor_y][neighbor_x] == 1:
                continue
                
            if (neighbor_x, neighbor_y) in closed_set:
                continue

            tentative_g = current_node.g + cost

            # If not visited or better g
            if (neighbor_x, neighbor_y) not in g_score or tentative_g < g_score[(neighbor_x, neighbor_y)]:
                neighbor_node = Node(neighbor_x, neighbor_y, parent=current_node)
                neighbor_node.g = tentative_g
                neighbor_node.h = heuristic(neighbor_node, goal_node)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                
                g_score[(neighbor_x, neighbor_y)] = tentative_g
                heapq.heappush(open_list, neighbor_node)

    return []

# ==========================================
# Test code and visualize
# ==========================================
if __name__ == '__main__':
    grid = [[0 for _ in range(20)] for _ in range(20)]
    
    for i in range(5, 15):
        grid[i][10] = 1 
    grid[14][11] = 1
    grid[14][12] = 1

    start = (2, 2)
    goal = (18, 12)

    path = astar(start, goal, grid)

    if path:
        print("Path is found! Length:", len(path))
        for y in range(20):
            row_str = ""
            for x in range(20):
                if (x, y) == start:
                    row_str += " S "
                elif (x, y) == goal:
                    row_str += " G "
                elif (x, y) in path:
                    row_str += " * "
                elif grid[y][x] == 1:
                    row_str += "███"
                else:
                    row_str += " . "
            print(row_str)
    else:
        print("Path not found")