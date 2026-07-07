import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from multi_path_plan import StaticMap

def generate_test_map(width=50, height=50, num_obstacles=50):
    """
    生成指定数量、较小体积障碍物的随机地图，确保有足够通路
    """
    circles = []
    rectangles = []
    
    # 依然保留安全区，防止老家被堵死
    safe_zones = [
        (0, 0, 10, 10),
        (width-10, height-10, width, height),
        (0, height-10, 10, height),
        (width-10, 0, width, 10)
    ]
    
    def is_in_safe_zone(x, y, r_or_w, h=None):
        check_w = r_or_w if h is None else r_or_w
        check_h = r_or_w if h is None else h
        for sx, sy, s_max_x, s_max_y in safe_zones:
            if not (x + check_w < sx or x > s_max_x or y + check_h < sy or y > s_max_y):
                return True
        return False

    attempts = 0
    max_attempts = num_obstacles * 5 # 防止陷入死循环
    
    while (len(circles) + len(rectangles)) < num_obstacles and attempts < max_attempts:
        attempts += 1
        shape_type = random.choice(['circle', 'square', 'rectangle'])
        
        if shape_type == 'circle':
            r = random.uniform(1.0, 2.5)
            cx = random.uniform(r, width - r)
            cy = random.uniform(r, height - r)
            if not is_in_safe_zone(cx-r, cy-r, r*2):
                circles.append((cx, cy, r))
                
        elif shape_type == 'square':
            side = random.uniform(1.5, 4.0)
            rx = random.uniform(0, width - side)
            ry = random.uniform(0, height - side)
            if not is_in_safe_zone(rx, ry, side, side):
                rectangles.append((rx, ry, side, side))
                
        else: # rectangle
            # 【修改点】：缩小矩形长宽，从 2~10 缩小为 2~5
            w = random.uniform(2.0, 5.0)
            h = random.uniform(2.0, 5.0)
            rx = random.uniform(0, width - w)
            ry = random.uniform(0, height - h)
            if not is_in_safe_zone(rx, ry, w, h):
                rectangles.append((rx, ry, w, h))

    # 计算一下实际密度，仅供标题展示参考
    temp_map = StaticMap(width, height, circles, rectangles)
    obstacles_count = sum(sum(row) for row in temp_map.grid_map)
    actual_density = obstacles_count / (width * height)

    return circles, rectangles, actual_density

# ==========================================
# 批量生成并可视化 5 张地图
# ==========================================
if __name__ == '__main__':
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for i in range(5):
        print(f"正在生成第 {i+1} 张地图...")
        # 设定生成 50 个较小的障碍物（你可以随时调大或调小这个数值来控制拥挤程度）
        circles, rectangles, density = generate_test_map(50, 50, num_obstacles=50)
        
        ax = axes[i]
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_aspect('equal')
        ax.set_title(f"Map {i+1}\nObstacles: 50 | Density: {density:.1%}")
        
        # 绘制方形障碍物
        for rx, ry, rw, rh in rectangles:
            rect = patches.Rectangle((rx, ry), rw, rh, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
            
        # 绘制圆形障碍物
        for cx, cy, r in circles:
            circ = patches.Circle((cx, cy), r, linewidth=1, edgecolor='black', facecolor='darkgray', alpha=0.7)
            ax.add_patch(circ)
            
        # 标记安全区
        safe_zones = [(0, 0, 10, 10), (40, 40, 50, 50), (0, 40, 10, 50), (40, 0, 50, 10)]
        for sx, sy, ex, ey in safe_zones:
            safe_box = patches.Rectangle((sx, sy), 10, 10, fill=False, edgecolor='green', linestyle='--')
            ax.add_patch(safe_box)
        
        print("circles:", circles, "\n")
        print("rectangles:", rectangles, "\n")
    plt.tight_layout()
    plt.show()