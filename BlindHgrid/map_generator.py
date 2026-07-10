import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_obstacle_density(width, height, circles=None, rectangles=None, inflation_radius=0.0):
    """
    Calculate obstacle density on a discrete grid.
    
    0 means free space, 1 means obstacle.
    This function is kept inside map_generator.py to avoid circular import with hgrid.py.
    """

    circles = circles if circles else []
    rectangles = rectangles if rectangles else []

    obstacle_count = 0
    for y in range(height):
        for x in range(width):
            occupied = False

            for cx, cy, r in circles:
                if math.hypot(x - cx, y - cy) <= r + inflation_radius:
                    occupied = True
                    break

            if not occupied:
                for rx, ry, rw, rh in rectangles:
                    if (rx - inflation_radius) <= x <= (rx + rw + inflation_radius) and \
                       (ry - inflation_radius) <= y <= (ry + rh + inflation_radius):
                        occupied = True
                        break

            if occupied:
                obstacle_count += 1

    return obstacle_count / (width * height)

def generate_test_map(width=50, height=50, num_obstacles=50, seed=None, safe_zone_size=10):
    """
    Generate a random 2D obstacle map.
    
    Parameters
    ----------
    width, height : int
        Map size.
    num_obstacles : int
        Total number of circular/rectangular obstacles.
    seed : int or None
        Random seed. Use a fixed seed if you want repeatable maps.
    safe_zone_size : int
        Two corner areas reserved for UAV start/goal positions.
        
    Returns
    -------
    circles : list[tuple]
        Each circle is (cx, cy, r).
    rectangles : list[tuple]
        Each rectangle is (rx, ry, rw, rh).
    actual_density : float
        Obstacle-grid density without inflation.
    """
    if seed is not None:
        random.seed(seed)

    circles = []
    rectangles = []

    safe_zones = [
        (0, 0, safe_zone_size, safe_zone_size),
        (width - safe_zone_size, 0, width, safe_zone_size),
    ]

    def is_in_safe_zone(x, y, w, h):
        for sx, sy, ex, ey in safe_zones:
            if not (x + w < sx or x > ex or y + h < sy or y > ey):
                return True
        return False
    
    attempts = 0
    max_attempts = num_obstacles * 10

    while (len(circles) + len(rectangles)) < num_obstacles and attempts < max_attempts:
        attempts += 1
        shape_type = random.choice(["circle", "square", "rectangle"])

        if shape_type == "circle":
            r = random.uniform(1.0, 2.5)
            cx = random.uniform(r, width - r)
            cy = random.uniform(r, height - r)
            if not is_in_safe_zone(cx - r, cy - r, 2 * r, 2 * r):
                circles.append((cx, cy, r))

        elif shape_type == "square":
            side = random.uniform(1.5, 4.0)
            rx = random.uniform(0, width - side)
            ry = random.uniform(0, height - side)
            if not is_in_safe_zone(rx, ry, side, side):
                rectangles.append((rx, ry, side, side))

        else:
            w = random.uniform(2.0, 5.0)
            h = random.uniform(2.0, 5.0)
            rx = random.uniform(0, width - w)
            ry = random.uniform(0, height - h)
            if not is_in_safe_zone(rx, ry, w, h):
                rectangles.append((rx, ry, w, h))

    actual_density = calculate_obstacle_density(width, height, circles, rectangles)
    return circles, rectangles, actual_density

def visualize_generated_maps(num_maps=5, width=50, height=50, num_obstacles=50, seed=None):
    """
    Optional visualization tool for checking random maps.
    This is not required by path planner the main test function.
    """
    fig, axes = plt.subplots(1, num_maps, figsize=(5 * num_maps, 5))
    if num_maps == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        map_seed = None if seed is None else seed + i
        circles, rectangles, density = generate_test_map(
            width=width,
            height=height,
            num_obstacles=num_obstacles,
            seed=map_seed,
        )

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.set_title(f"Map {i + 1}\nObstacles: {num_obstacles} | Density: {density:.1%}")

        for rx, ry, rw, rh in rectangles:
            rect = patches.Rectangle(
                (rx, ry), rw, rh,
                linewidth=1,
                edgecolor="black",
                facecolor="gray",
                alpha=0.7,
            )
            ax.add_patch(rect)

        for cx, cy, r in circles:
            circ = patches.Circle(
                (cx, cy), r,
                linewidth=1,
                edgecolor="black",
                facecolor="darkgray",
                alpha=0.7,
            )
            ax.add_patch(circ)

        safe_zone_size = 10
        safe_zones = [
            (0, 0),
            (width - safe_zone_size, 0),
        ]
        for sx, sy in safe_zones:
            safe_box = patches.Rectangle(
                (sx, sy), safe_zone_size, safe_zone_size,
                fill=False,
                edgecolor="green",
                linestyle="--",
            )
            ax.add_patch(safe_box)

        print(f"Map {i + 1} seed={map_seed}")
        print("circles:", circles, "\n")
        print("rectangles:", rectangles, "\n")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_generated_maps(num_maps=5, width=50, height=50, num_obstacles=50, seed=None)
