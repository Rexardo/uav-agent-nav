import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

# Import baseline
from basic_benchmark import (
    StaticMap, make_center_inspection_region, generate_reachable_inspection_waypoints, 
    assign_waypoints_to_uavs, InspectionMap, run_simulation as run_baseline_simulation
)
from map_generator import generate_test_map

# pibt core
from pibt_core import PIBTStepPlanner, project_goal_to_reachable

# ==========================================
# Simulator based on PIBT
# ==========================================
def run_pibt_simulation(
    num_uavs=8, width=50, height=50, num_obstacles=50, map_seed=42, inflation_radius=0.8,
    max_logical_steps=1000, coverage_radius=3.0, center_region_size=30, waypoint_spacing=None,
    verbose=False
):
    # Initialize map
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

    free_grid = (np.array(env_map.grid_map) == 0)

    uav_states = []
    for i in range(num_uavs):
        start_coord = (int(uav_starts[i][0]), int(uav_starts[i][1]))
        goal_coord = (int(uav_tasks[i][1][0]), int(uav_tasks[i][1][1]))
        uav_states.append({
            'id': i + 1,
            'pos': start_coord,
            'goal': goal_coord,
            'waypoints': [(int(w[0]), int(w[1])) for w in waypoint_assignments[i+1]],
            'wp_idx': 0,
            'path_length': 0.0,
            'is_reached': False
        })
        inspection_map.update_coverage(start_coord)

    total_resolutions = 0  # Record pibt backtrack
    total_steps = max_logical_steps

    if verbose:
        print(f"  [Map {map_seed}] 启动 PIBT 仿真: UAV数量={num_uavs}...")

    # Main loop
    for t in range(max_logical_steps):
        current_positions = []
        targets = []
        priorities = []

        # Update priority
        for uav in uav_states:
            # waypoint arived?
            while uav['wp_idx'] < len(uav['waypoints']):
                wp = uav['waypoints'][uav['wp_idx']]
                if uav['pos'] == wp:  
                    uav['wp_idx'] += 1
                else:
                    break

            if uav['wp_idx'] >= len(uav['waypoints']) and uav['pos'] == uav['goal']:
                uav['is_reached'] = True

            current_positions.append(uav['pos'])

            if uav['is_reached']:
                target = uav['pos']
                priority = 0.0
            else:
                target_wp = uav['waypoints'][uav['wp_idx']] if uav['wp_idx'] < len(uav['waypoints']) else uav['goal']
                target, _ = project_goal_to_reachable(free_grid, uav['pos'], target_wp)
                
                priority = abs(uav['pos'][0] - target[0]) + abs(uav['pos'][1] - target[1])

            targets.append(target)
            priorities.append(float(priority))

        if all(uav['is_reached'] for uav in uav_states):
            total_steps = t
            break

        pibt = PIBTStepPlanner(free_grid, targets, seed=map_seed + t)
        next_positions, stats = pibt.step(current_positions, priorities)
        
        total_resolutions += stats.backtracks 

        for i, uav in enumerate(uav_states):
            if uav['pos'] != next_positions[i]:
                uav['path_length'] += np.linalg.norm(np.array(uav['pos']) - np.array(next_positions[i]))
                uav['pos'] = next_positions[i]
            inspection_map.update_coverage(uav['pos'])

    coverage = inspection_map.coverage_ratio() * 100.0
    total_path = sum(uav['path_length'] for uav in uav_states)
    avg_path = total_path / num_uavs if num_uavs > 0 else 0

    if verbose:
        print(f"    -> PIBT完成! 覆盖率:{coverage:.1f}%, PIBT回溯化解次数:{total_resolutions}, 总步数:{total_steps}, 平均路径:{avg_path:.1f}")

    return {
        "coverage": coverage,
        "deadlocks": total_resolutions,
        "total_path": total_path,
        "avg_path": avg_path,
        "total_steps": total_steps
    }

# ==========================================
# Benchmark & Plotting (Dual comparation)
# ==========================================
def run_comparison_benchmark(num_maps=5, max_uavs=8):
    print(f"\n==============================================")
    print(f"🚀 开始核心算法对比基准测试 (Baseline vs PIBT)")
    print(f"   共计 {num_maps} 张随机地图 | 每张跑 1 到 {max_uavs} 架无人机")
    print(f"==============================================\n")

    # 数据结构初始化
    methods = ["Baseline", "PIBT"]
    metrics = {
        m: {u: {"coverage": [], "deadlocks": [], "total_path": [], "avg_path": [], "total_steps": []} 
            for u in range(1, max_uavs + 1)}
        for m in methods
    }

    for step in range(num_maps):
        current_seed = random.randint(0, 999999)
        print(f">>> 正在测试 Map {step + 1}/{num_maps} (Seed: {current_seed})")
        
        for num_uavs in range(1, max_uavs + 1):
            # 1. 跑 Baseline (A* + 停车让行)
            res_base = run_baseline_simulation(num_uavs=num_uavs, map_seed=current_seed, verbose=False)
            # 2. 跑 PIBT (严格的一步规划)
            res_pibt = run_pibt_simulation(num_uavs=num_uavs, map_seed=current_seed, verbose=False)
            
            # 记录数据
            for m_name, res in zip(methods, [res_base, res_pibt]):
                metrics[m_name][num_uavs]["coverage"].append(res["coverage"])
                metrics[m_name][num_uavs]["deadlocks"].append(res["deadlocks"])
                metrics[m_name][num_uavs]["total_path"].append(res["total_path"])
                metrics[m_name][num_uavs]["avg_path"].append(res["avg_path"])
                metrics[m_name][num_uavs]["total_steps"].append(res["total_steps"])

    print("\n✅ 所有测试运行完毕，正在生成对比图表...")

    x_axis = list(range(1, max_uavs + 1))
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"MAPF Benchmark: Baseline vs PIBT (Averaged over {num_maps} Maps)", fontsize=16)

    colors = {"Baseline": "blue", "PIBT": "red"}
    markers = {"Baseline": "o", "PIBT": "s"}

    for m_name in methods:
        c = colors[m_name]
        mk = markers[m_name]
        
        mean_cov = [np.mean(metrics[m_name][u]["coverage"]) for u in x_axis]
        mean_steps = [np.mean(metrics[m_name][u]["total_steps"]) for u in x_axis]
        mean_t_path = [np.mean(metrics[m_name][u]["total_path"]) for u in x_axis]
        mean_dead = [np.mean(metrics[m_name][u]["deadlocks"]) for u in x_axis]

        # 1. 任务完成总时间(Steps) -> 最重要的性能指标
        axs[0, 0].plot(x_axis, mean_steps, linestyle='-', marker=mk, color=c, label=m_name)
        axs[0, 0].set_title('Total Mission Time (Steps)')
        axs[0, 0].set_xlabel('Number of UAVs')
        axs[0, 0].set_ylabel('Steps (Lower is better)')
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)

        # 2. 冲突解决/死锁次数
        axs[0, 1].plot(x_axis, mean_dead, linestyle='-', marker=mk, color=c, label=m_name)
        axs[0, 1].set_title('Conflicts/Deadlocks Resolved')
        axs[0, 1].set_xlabel('Number of UAVs')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)

        # 3. 总巡航路径
        axs[1, 0].plot(x_axis, mean_t_path, linestyle='-', marker=mk, color=c, label=m_name)
        axs[1, 0].set_title('Total Path Length')
        axs[1, 0].set_xlabel('Number of UAVs')
        axs[1, 0].set_ylabel('Distance')
        axs[1, 0].grid(True, linestyle='--', alpha=0.6)

        # 4. 任务覆盖率 (验证任务是否100%完成)
        axs[1, 1].plot(x_axis, mean_cov, linestyle='-', marker=mk, color=c, label=m_name)
        axs[1, 1].set_title('Task Coverage Ratio (%)')
        axs[1, 1].set_xlabel('Number of UAVs')
        axs[1, 1].set_ylabel('Coverage (%)')
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    for ax_row in axs:
        for ax in ax_row:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("pibt_vs_baseline.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_maps", type=int, default=10, help="Number of random maps to test")
    parser.add_argument("--max_uavs", type=int, default=8, help="Max number of UAVs to test")
    args = parser.parse_args()

    run_comparison_benchmark(num_maps=args.num_maps, max_uavs=args.max_uavs)