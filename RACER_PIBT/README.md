# RACER + PIBT 运动规划

这个目录实现了以下分层框架，并且不会修改 `RACER` 目录中的接口：

1. RACER 原代码继续负责地图、有限传感器、地图合并、HGrid/CVRP 分工、覆盖路径引导和目标点选择。
2. RACER 为每架无人机选出目标后，其 kinodynamic search 与 B-spline 路径只用于目标评分，不再用于执行。
3. 当前通信范围内的无人机组成局部通信组，每个重规划周期选择临时 coordinator；非直连无人机的状态和轨迹沿组内最短两两链路转发，并统计 message hop。
4. 组内使用官方 `pypibt` 同构的优先级继承与回溯，生成无顶点冲突、无对向换位冲突的下一 waypoint。
5. 未知区域在 PIBT 中按障碍处理；若 RACER 目标暂时不可达，就投影到当前已知连通自由区，下一周期重新计算。
6. waypoint 后接 subgoal 线性规划、SFC、逐控制点 LSC，以及静止到静止的五次 Bernstein 最小 jerk 轨迹。
7. 每段轨迹检查速度、加速度、障碍物和无人机间连续距离后才执行。

默认机体半径为 `0.25` 格，与原 RACER 仿真的 `0.50` 最小中心距碰撞监视器一致；即使命令行传入更小值，仿真也会使用碰撞监视器对应的半径作为下限。

## 运行

在项目根目录执行：

```powershell
python RACER_PIBT/RACER_PIBT.py --map-id 2 --num-uavs 4
```

无动画快速测试：

```powershell
python RACER_PIBT/RACER_PIBT.py --map-id 2 --num-uavs 4 --max-steps 500 --no-show
```

五张地图、只测试四架无人机：

```powershell
python RACER_PIBT/basic_benchmark.py --map-id 1 --num-maps 5 --uav-counts 4 --no-show
```

benchmark 的 PNG 仍只画覆盖率、步数、总路径和单机平均路径；平均转向角、最小障碍物距离和碰撞次数只在 terminal 表格输出。

地图仍由 `RACER/racer_map.py` 提供：`1` 是原随机障碍物地图，`2` 是四架无人机从左侧进入的一格宽 Dense Maze。

## 与论文实现的边界

`pibt_core.py` 参考论文作者 Keisuke Okumura 的 MIT 许可官方仓库 `Kei18/pypibt`。该仓库只公开离散 PIBT，不包含 Park、Jang、Kim 论文中的完整 CPLEX 多段 QP 工程。

这里的连续规划严格保留论文的关键结构，但针对当前二维栅格仿真做了单步特化：一个 PIBT 网格边对应一段五次 Bernstein 最小 jerk 轨迹，subgoal LP 可解析求解，SFC/LSC 和动力学作为硬可行性约束。它不是原作者未公开的 CPLEX 工程复刻，也不能把原论文的全局 deadlock-free 定理直接搬到未知地图上。

理论保证仍依赖已知自由图的双连通/简单环条件。Dense Maze 的一格宽桥和死胡同不满足该条件，因此此地图上应把结果视为死锁处理实验，而不是无条件完备性证明。
