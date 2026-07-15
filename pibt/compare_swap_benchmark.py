import argparse
import random
import time
import numpy as np
import matplotlib.pyplot as plt

# 请确保这些现有的模块在同一目录下
from basic_benchmark import (
    StaticMap, make_center_inspection_region, generate_reachable_inspection_waypoints, 
    assign_waypoints_to_uavs, InspectionMap
)
from map_generator import generate_test_map
from pibt_core import PIBTStepPlanner, project_goal_to_reachable

# ==========================================
# 统一的 PIBT Simulator (支持开关 Task Swap)
# ==========================================
def run_pibt_simulation(
    num_uavs=8, width=50, height=50, num_obstacles=50, map_seed=42, inflation_radius=0.8,
    max_logical_steps=1000, coverage_radius=3.0, center_region_size=30, waypoint_spacing=None,
    use_task_swap=False, verbose=False
):
    # 初始化地图
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

    # 生成航点与分配
    inspection_waypoints, _, _ = generate_reachable_inspection_waypoints(
        static_map=env_map, uav_starts=uav_starts, inspection_regions=inspection_regions,
        coverage_radius=coverage_radius, waypoint_spacing=waypoint_spacing
    )
    waypoint_assignments, _ = assign_waypoints_to_uavs(env_map, inspection_waypoints, uav_tasks)
    
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

    total_steps = max_logical_steps
    mode_name = "PIBT (Task Swap)" if use_task_swap else "Basic PIBT"

    if verbose:
        print(f"  [Map {map_seed}] 启动 {mode_name} 仿真: UAV数量={num_uavs}...")

    # === 开始核心计时 ===
    start_time = time.perf_counter()

    for t in range(max_logical_steps):
        # 1. 更新任务到达状态
        for uav in uav_states:
            while uav['wp_idx'] < len(uav['waypoints']):
                wp = uav['waypoints'][uav['wp_idx']]
                if uav['pos'] == wp:  
                    uav['wp_idx'] += 1
                else:
                    break
            if uav['wp_idx'] >= len(uav['waypoints']) and uav['pos'] == uav['goal']:
                uav['is_reached'] = True

        # 2. 执行任务交换 (Task Swapping) 检查
        if use_task_swap:
            for i in range(len(uav_states)):
                for j in range(i + 1, len(uav_states)):
                    u1 = uav_states[i]
                    u2 = uav_states[j]
                    
                    if u1['is_reached'] and u2['is_reached']:
                        continue
                    
                    # 判断曼哈顿距离
                    dist_between = abs(u1['pos'][0] - u2['pos'][0]) + abs(u1['pos'][1] - u2['pos'][1])
                    if dist_between <= 4:
                        t1 = u1['waypoints'][u1['wp_idx']] if u1['wp_idx'] < len(u1['waypoints']) else u1['goal']
                        t2 = u2['waypoints'][u2['wp_idx']] if u2['wp_idx'] < len(u2['waypoints']) else u2['goal']
                        
                        dist_orig = abs(u1['pos'][0] - t1[0]) + abs(u1['pos'][1] - t1[1]) + \
                                    abs(u2['pos'][0] - t2[0]) + abs(u2['pos'][1] - t2[1])
                        
                        dist_swap = abs(u1['pos'][0] - t2[0]) + abs(u1['pos'][1] - t2[1]) + \
                                    abs(u2['pos'][0] - t1[0]) + abs(u2['pos'][1] - t1[1])
                                    
                        if dist_swap < dist_orig:
                            # 交换剩余任务
                            rem_wp1 = u1['waypoints'][u1['wp_idx']:]
                            rem_wp2 = u2['waypoints'][u2['wp_idx']:]
                            u1['waypoints'] = rem_wp2
                            u2['waypoints'] = rem_wp1
                            u1['wp_idx'] = 0
                            u2['wp_idx'] = 0
                            u1['goal'], u2['goal'] = u2['goal'], u1['goal']
                            u1['is_reached'] = False
                            u2['is_reached'] = False

        # 3. 构建 PIBT 输入
        current_positions = []
        targets = []
        priorities = []

        for uav in uav_states:
            if not uav['is_reached']:
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
        next_positions, _ = pibt.step(current_positions, priorities)

        for i, uav in enumerate(uav_states):
            if uav['pos'] != next_positions[i]:
                uav['path_length'] += np.linalg.norm(np.array(uav['pos']) - np.array(next_positions[i]))
                uav['pos'] = next_positions[i]

    # === 结束核心计时 ===
    runtime_sec = time.perf_counter() - start_time
    total_time_cost = total_steps + runtime_sec

    max_path = max(uav['path_length'] for uav in uav_states) if num_uavs > 0 else 0

    if verbose:
        print(f"    -> {mode_name}完成! 步数:{total_steps}, 运行耗时:{runtime_sec:.4f}s, 总代价:{total_time_cost:.2f}")

    return {
        "max_path": max_path,     
        "total_steps": total_steps,
        "runtime": runtime_sec,
        "total_time_cost": total_time_cost
    }

# ==========================================
# 比较 Benchmark (Basic PIBT vs Task Swap PIBT)
# ==========================================
def run_comparison_benchmark(num_maps=5, max_uavs=8):
    print(f"\n==============================================")
    print(f"🚀 开始消融实验 (Basic PIBT  vs  PIBT + Task Swap)")
    print(f"   共计 {num_maps} 张随机地图 | 每张跑 1 到 {max_uavs} 架无人机")
    print(f"==============================================\n")

    methods = ["Basic PIBT", "PIBT (Task Swap)"]
    metrics = {
        m: {u: {"max_path": [], "total_steps": [], "runtime": [], "total_time_cost": []} 
            for u in range(1, max_uavs + 1)}
        for m in methods
    }

    for step in range(num_maps):
        current_seed = random.randint(0, 999999)
        print(f">>> 正在测试 Map {step + 1}/{num_maps} (Seed: {current_seed})")
        
        for num_uavs in range(1, max_uavs + 1):
            res_basic = run_pibt_simulation(num_uavs=num_uavs, map_seed=current_seed, use_task_swap=False, verbose=False)
            res_swap = run_pibt_simulation(num_uavs=num_uavs, map_seed=current_seed, use_task_swap=True, verbose=False)
            
            for m_name, res in zip(methods, [res_basic, res_swap]):
                metrics[m_name][num_uavs]["max_path"].append(res["max_path"]) 
                metrics[m_name][num_uavs]["total_steps"].append(res["total_steps"])
                metrics[m_name][num_uavs]["runtime"].append(res["runtime"])
                metrics[m_name][num_uavs]["total_time_cost"].append(res["total_time_cost"])

    print("\n✅ 所有测试运行完毕，正在生成对比图表...")

    x_axis = list(range(1, max_uavs + 1))
    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"Ablation Study: Basic vs Task Swap PIBT (Averaged over {num_maps} Maps)", fontsize=16)

    colors = {"Basic PIBT": "gray", "PIBT (Task Swap)": "red"}
    markers = {"Basic PIBT": "X", "PIBT (Task Swap)": "s"}
    line_styles = {"Basic PIBT": "--", "PIBT (Task Swap)": "-"}

    for m_name in methods:
        c = colors[m_name]
        mk = markers[m_name]
        ls = line_styles[m_name]
        
        mean_steps = [np.mean(metrics[m_name][u]["total_steps"]) for u in x_axis]
        mean_runtime = [np.mean(metrics[m_name][u]["runtime"]) for u in x_axis]
        mean_max_path = [np.mean(metrics[m_name][u]["max_path"]) for u in x_axis]
        mean_cost = [np.mean(metrics[m_name][u]["total_time_cost"]) for u in x_axis]

        # 1. 任务完成总逻辑步数(Steps) 
        axs[0, 0].plot(x_axis, mean_steps, linestyle=ls, marker=mk, color=c, label=m_name)
        axs[0, 0].set_title('Logical Mission Time (Steps)')
        axs[0, 0].set_xlabel('Number of UAVs')
        axs[0, 0].set_ylabel('Steps')
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)

        # 2. 代码运行时间 (Runtime)
        axs[0, 1].plot(x_axis, mean_runtime, linestyle=ls, marker=mk, color=c, label=m_name)
        axs[0, 1].set_title('Code Execution Time (Seconds)')
        axs[0, 1].set_xlabel('Number of UAVs')
        axs[0, 1].set_ylabel('Real Time (s)')
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)

        # 3. Makespan / 木桶的最短板
        axs[1, 0].plot(x_axis, mean_max_path, linestyle=ls, marker=mk, color=c, label=m_name)
        axs[1, 0].set_title('Makespan (Max Single UAV Path Length)')
        axs[1, 0].set_xlabel('Number of UAVs')
        axs[1, 0].set_ylabel('Max Distance')
        axs[1, 0].grid(True, linestyle='--', alpha=0.6)

        # 4. 真实仿真延迟 (Total Time Cost = Steps + Runtime)
        axs[1, 1].plot(x_axis, mean_cost, linestyle=ls, marker=mk, color=c, label=m_name)
        axs[1, 1].set_title('Total Time Cost (Steps + Runtime)')
        axs[1, 1].set_xlabel('Number of UAVs')
        axs[1, 1].set_ylabel('Cost (Lower is better)')
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    for ax_row in axs:
        for ax in ax_row:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("pibt_ablation_task_swap_runtime.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_maps", type=int, default=10, help="Number of random maps to test")
    parser.add_argument("--max_uavs", type=int, default=8, help="Max number of UAVs to test")
    args = parser.parse_args()

    run_comparison_benchmark(num_maps=args.num_maps, max_uavs=args.max_uavs)