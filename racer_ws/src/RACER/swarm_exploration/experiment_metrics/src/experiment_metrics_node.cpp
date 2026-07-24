#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h>
#include <std_msgs/UInt8.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <boost/bind/bind.hpp>
#include <boost/filesystem.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = boost::filesystem;
using boost::placeholders::_1;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct TrajectorySample {
  double elapsed_s;
  int drone_id;
  Eigen::Vector3d position;
  Eigen::Vector3d velocity;
  int fsm_state;
};

struct CoverageSample {
  double elapsed_s;
  double coverage_pct;
};

struct CollisionEvent {
  double elapsed_s;
  std::string type;
  int uav_a;
  int uav_b;
  Eigen::Vector3d position;
  double center_distance_m;
  double clearance_m;
};

struct PlanningStats {
  size_t call_count = 0;
  double time_sum_s = 0.0;
  double mean_time_s = std::numeric_limits<double>::quiet_NaN();
  double max_time_s = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> uav_mean_time_s;
};

std::string trimNumber(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << value;
  std::string text = out.str();
  while (!text.empty() && text.back() == '0') text.pop_back();
  if (!text.empty() && text.back() == '.') text.pop_back();
  std::replace(text.begin(), text.end(), '.', 'p');
  return text;
}

std::string crDirectoryName(double communication_range) {
  return communication_range < 0.0 ? "CR_inf" : "CR_" + trimNumber(communication_range) + "m";
}

std::string formatLocalTime(std::time_t value, const char* format) {
  std::tm local_tm;
#ifdef _WIN32
  localtime_s(&local_tm, &value);
#else
  localtime_r(&value, &local_tm);
#endif
  char buffer[64];
  std::strftime(buffer, sizeof(buffer), format, &local_tm);
  return buffer;
}

void requireOpen(const std::ofstream& stream, const fs::path& path) {
  if (!stream.is_open()) throw std::runtime_error("cannot open " + path.string());
}

void writeJsonNumber(std::ostream& out, double value) {
  if (std::isfinite(value))
    out << std::fixed << std::setprecision(6) << value;
  else
    out << "null";
}

void writeCsvNumber(std::ostream& out, double value) {
  if (std::isfinite(value)) out << std::fixed << std::setprecision(6) << value;
}

}  // namespace

class ExperimentMetricsNode {
public:
  ExperimentMetricsNode() : nh_(), pnh_("~"), obstacle_cloud_(new pcl::PointCloud<pcl::PointXYZ>) {
    pnh_.param("drone_num", drone_num_, 4);
    pnh_.param("scene", scene_, std::string("dense_maze"));
    pnh_.param("communication_range", communication_range_, -1.0);
    pnh_.param("seed", seed_, 0);
    pnh_.param("output_root", output_root_, std::string("/home/qian/racer_pibt_ws/results"));
    pnh_.param("odom_prefix", odom_prefix_, std::string("/state_ukf/odom"));
    pnh_.param("fsm_state_prefix", fsm_state_prefix_, std::string("/experiment/fsm_state"));
    pnh_.param(
        "planning_time_prefix", planning_time_prefix_, std::string("/experiment/planning_time"));
    pnh_.param("known_grid_prefix", known_grid_prefix_, std::string("/racer/known_grid"));
    pnh_.param("trigger_topic", trigger_topic_, std::string("/move_base_simple/goal"));
    pnh_.param("global_map_topic", global_map_topic_, std::string("/map_generator/global_cloud"));
    pnh_.param("idle_state_value", idle_state_value_, 6);
    pnh_.param("idle_hold_duration", idle_hold_duration_, 3.0);
    pnh_.param("stationary_hold_duration", stationary_hold_duration_, 30.0);
    pnh_.param("stationary_speed_threshold", stationary_speed_threshold_, 0.05);
    pnh_.param("odom_freshness_timeout", odom_freshness_timeout_, 1.0);
    pnh_.param("uav_radius", uav_radius_, 0.25);
    pnh_.param("collision_hysteresis", collision_hysteresis_, 0.05);
    pnh_.param("turn_sample_distance", turn_sample_distance_, 0.20);
    pnh_.param("trajectory_sample_period", trajectory_sample_period_, 0.05);
    pnh_.param("obstacle_check_period", obstacle_check_period_, 0.02);
    pnh_.param("obstacle_min_z", obstacle_min_z_, 0.25);
    pnh_.param("coverage_reachable_only", coverage_reachable_only_, true);
    pnh_.param("coverage_obstacle_clearance", coverage_obstacle_clearance_, uav_radius_);
    pnh_.param("coverage_slice_half_height", coverage_slice_half_height_, uav_radius_);
    pnh_.param("coverage_seed_search_radius", coverage_seed_search_radius_, 1.0);

    states_.assign(drone_num_, -1);
    have_state_.assign(drone_num_, false);
    have_odom_.assign(drone_num_, false);
    latest_positions_.assign(drone_num_, Eigen::Vector3d::Zero());
    latest_velocities_.assign(drone_num_, Eigen::Vector3d::Zero());
    last_path_positions_.assign(drone_num_, Eigen::Vector3d::Zero());
    have_last_path_position_.assign(drone_num_, false);
    path_lengths_.assign(drone_num_, 0.0);
    turn_paths_.resize(drone_num_);
    last_trajectory_sample_time_.assign(drone_num_, ros::Time(0));
    last_obstacle_check_time_.assign(drone_num_, ros::Time(0));
    last_odom_times_.assign(drone_num_, ros::Time(0));
    obstacle_contact_.assign(drone_num_, false);
    idle_entry_times_.assign(drone_num_, ros::Time(0));
    known_grids_.resize(drone_num_);
    have_grid_.assign(drone_num_, false);
    pair_contact_.assign(drone_num_, std::vector<bool>(drone_num_, false));
    planning_call_counts_.assign(drone_num_, 0);
    planning_time_sums_.assign(drone_num_, 0.0);
    planning_time_maxima_.assign(drone_num_, 0.0);

    for (int index = 0; index < drone_num_; ++index) {
      const std::string suffix = "_" + std::to_string(index + 1);
      odom_subs_.push_back(nh_.subscribe<nav_msgs::Odometry>(odom_prefix_ + suffix, 50,
          boost::bind(&ExperimentMetricsNode::odomCallback, this, _1, index)));
      state_subs_.push_back(nh_.subscribe<std_msgs::UInt8>(fsm_state_prefix_ + suffix, 10,
          boost::bind(&ExperimentMetricsNode::stateCallback, this, _1, index)));
      planning_time_subs_.push_back(nh_.subscribe<std_msgs::Float64>(planning_time_prefix_ + suffix,
          100, boost::bind(&ExperimentMetricsNode::planningTimeCallback, this, _1, index)));
      grid_subs_.push_back(nh_.subscribe<nav_msgs::OccupancyGrid>(known_grid_prefix_ + suffix, 2,
          boost::bind(&ExperimentMetricsNode::gridCallback, this, _1, index)));
    }

    trigger_sub_ = nh_.subscribe(trigger_topic_, 1, &ExperimentMetricsNode::triggerCallback, this);
    map_sub_ = nh_.subscribe(global_map_topic_, 1, &ExperimentMetricsNode::mapCallback, this);
    finish_timer_ = nh_.createTimer(ros::Duration(0.1), &ExperimentMetricsNode::finishTimer, this);
    coverage_timer_ =
        nh_.createTimer(ros::Duration(1.0), &ExperimentMetricsNode::coverageTimer, this);

    ROS_INFO("Experiment recorder ready: scene=%s, CR=%s, output=%s", scene_.c_str(),
        crDirectoryName(communication_range_).c_str(), output_root_.c_str());
  }

private:
  void triggerCallback(const geometry_msgs::PoseStampedConstPtr&) {
    if (started_ || finished_) return;

    started_ = true;
    start_time_ = ros::Time::now();
    run_wall_start_ = std::time(nullptr);
    all_idle_since_ = ros::Time(0);
    all_stationary_since_ = ros::Time(0);
    finish_reason_.clear();
    std::fill(idle_entry_times_.begin(), idle_entry_times_.end(), ros::Time(0));

    for (int i = 0; i < drone_num_; ++i) {
      if (!have_odom_[i]) continue;
      last_path_positions_[i] = latest_positions_[i];
      have_last_path_position_[i] = true;
      turn_paths_[i].push_back(latest_positions_[i]);
    }

    ROS_INFO("Experiment recording started from the first exploration trigger");
    if (!map_ready_) ROS_WARN("Ground-truth point cloud is not ready yet; waiting for it");
  }

  void stateCallback(const std_msgs::UInt8ConstPtr& msg, int index) {
    if (index < 0 || index >= drone_num_) return;
    const int previous_state = states_[index];
    states_[index] = static_cast<int>(msg->data);
    have_state_[index] = true;

    if (!started_ || finished_) return;
    if (states_[index] == idle_state_value_) {
      if (previous_state != idle_state_value_ || idle_entry_times_[index].isZero())
        idle_entry_times_[index] = ros::Time::now();
    } else {
      idle_entry_times_[index] = ros::Time(0);
    }
  }

  void planningTimeCallback(const std_msgs::Float64ConstPtr& msg, int index) {
    if (index < 0 || index >= drone_num_ || finished_ || !std::isfinite(msg->data) ||
        msg->data < 0.0)
      return;
    ++planning_call_counts_[index];
    planning_time_sums_[index] += msg->data;
    planning_time_maxima_[index] = std::max(planning_time_maxima_[index], msg->data);
  }

  void gridCallback(const nav_msgs::OccupancyGridConstPtr& msg, int index) {
    if (index < 0 || index >= drone_num_) return;
    known_grids_[index] = *msg;
    have_grid_[index] = true;
  }

  void mapCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    if (map_ready_) return;
    pcl::PointCloud<pcl::PointXYZ> raw_cloud;
    pcl::fromROSMsg(*msg, raw_cloud);
    obstacle_cloud_->clear();
    obstacle_cloud_->reserve(raw_cloud.size());
    for (const auto& point : raw_cloud) {
      if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
          point.z >= obstacle_min_z_)
        obstacle_cloud_->push_back(point);
    }
    if (obstacle_cloud_->empty()) {
      ROS_WARN("Received an empty ground-truth point cloud");
      return;
    }
    obstacle_kdtree_.setInputCloud(obstacle_cloud_);
    map_ready_ = true;
    ROS_INFO("Ground-truth point cloud ready: %zu points", obstacle_cloud_->size());
  }

  void odomCallback(const nav_msgs::OdometryConstPtr& msg, int index) {
    if (index < 0 || index >= drone_num_) return;

    const ros::Time now = ros::Time::now();

    const Eigen::Vector3d position(msg->pose.pose.position.x, msg->pose.pose.position.y,
        msg->pose.pose.position.z);
    const Eigen::Vector3d velocity(msg->twist.twist.linear.x, msg->twist.twist.linear.y,
        msg->twist.twist.linear.z);
    latest_positions_[index] = position;
    latest_velocities_[index] = velocity;
    have_odom_[index] = true;
    last_odom_times_[index] = now;

    if (!started_ || finished_) return;
    const double elapsed_s = (now - start_time_).toSec();

    if (!have_last_path_position_[index]) {
      last_path_positions_[index] = position;
      have_last_path_position_[index] = true;
      turn_paths_[index].push_back(position);
    } else {
      const double increment = (position - last_path_positions_[index]).norm();
      if (increment > 1e-6) {
        path_lengths_[index] += increment;
        last_path_positions_[index] = position;
      }
    }

    if (turn_paths_[index].empty() ||
        (position.head<2>() - turn_paths_[index].back().head<2>()).norm() >=
            turn_sample_distance_) {
      turn_paths_[index].push_back(position);
    }

    if (last_trajectory_sample_time_[index].isZero() ||
        (now - last_trajectory_sample_time_[index]).toSec() >= trajectory_sample_period_) {
      trajectory_samples_.push_back(
          { elapsed_s, index + 1, position, velocity, states_[index] });
      last_trajectory_sample_time_[index] = now;
    }

    if (map_ready_ && (last_obstacle_check_time_[index].isZero() ||
                          (now - last_obstacle_check_time_[index]).toSec() >=
                              obstacle_check_period_)) {
      checkObstacleCollision(index, position, elapsed_s);
      last_obstacle_check_time_[index] = now;
    }
    checkInterUavCollisions(elapsed_s);
  }

  void checkObstacleCollision(int index, const Eigen::Vector3d& position, double elapsed_s) {
    pcl::PointXYZ query;
    query.x = static_cast<float>(position.x());
    query.y = static_cast<float>(position.y());
    query.z = static_cast<float>(position.z());
    std::vector<int> nearest_index(1);
    std::vector<float> squared_distance(1);
    if (obstacle_kdtree_.nearestKSearch(query, 1, nearest_index, squared_distance) <= 0) return;

    const double center_distance = std::sqrt(std::max(0.0f, squared_distance[0]));
    const double clearance = center_distance - uav_radius_;
    min_obstacle_center_distance_ = std::min(min_obstacle_center_distance_, center_distance);
    min_obstacle_clearance_ = std::min(min_obstacle_clearance_, clearance);

    if (!obstacle_contact_[index] && center_distance <= uav_radius_) {
      obstacle_contact_[index] = true;
      collision_events_.push_back({ elapsed_s, "obstacle", index + 1, 0, position,
          center_distance, clearance });
    } else if (obstacle_contact_[index] &&
        center_distance > uav_radius_ + collision_hysteresis_) {
      obstacle_contact_[index] = false;
    }
  }

  void checkInterUavCollisions(double elapsed_s) {
    for (int first = 0; first < drone_num_; ++first) {
      if (!have_odom_[first]) continue;
      for (int second = first + 1; second < drone_num_; ++second) {
        if (!have_odom_[second]) continue;
        const double center_distance = (latest_positions_[first] - latest_positions_[second]).norm();
        const double clearance = center_distance - 2.0 * uav_radius_;
        min_inter_uav_center_distance_ =
            std::min(min_inter_uav_center_distance_, center_distance);

        if (!pair_contact_[first][second] && center_distance <= 2.0 * uav_radius_) {
          pair_contact_[first][second] = true;
          collision_events_.push_back({ elapsed_s, "uav", first + 1, second + 1,
              0.5 * (latest_positions_[first] + latest_positions_[second]), center_distance,
              clearance });
        } else if (pair_contact_[first][second] &&
            center_distance > 2.0 * uav_radius_ + collision_hysteresis_) {
          pair_contact_[first][second] = false;
        }
      }
    }
  }

  bool gridsCompatible(
      const nav_msgs::OccupancyGrid& first, const nav_msgs::OccupancyGrid& second) const {
    return first.info.width == second.info.width && first.info.height == second.info.height &&
        std::fabs(first.info.resolution - second.info.resolution) < 1e-6 &&
        std::fabs(first.info.origin.position.x - second.info.origin.position.x) < 1e-6 &&
        std::fabs(first.info.origin.position.y - second.info.origin.position.y) < 1e-6 &&
        first.data.size() == second.data.size();
  }

  bool findFreeCoverageSeed(const nav_msgs::OccupancyGrid& grid,
      const std::vector<uint8_t>& obstacle_mask, const Eigen::Vector3d& position,
      size_t& seed_cell) const {
    const double resolution = grid.info.resolution;
    const double origin_x = grid.info.origin.position.x;
    const double origin_y = grid.info.origin.position.y;
    const int width = static_cast<int>(grid.info.width);
    const int height = static_cast<int>(grid.info.height);
    const int center_x = static_cast<int>(std::floor((position.x() - origin_x) / resolution));
    const int center_y = static_cast<int>(std::floor((position.y() - origin_y) / resolution));
    const int search_cells =
        std::max(0, static_cast<int>(std::ceil(coverage_seed_search_radius_ / resolution)));

    double best_distance_sq = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int dy = -search_cells; dy <= search_cells; ++dy) {
      const int y = center_y + dy;
      if (y < 0 || y >= height) continue;
      for (int dx = -search_cells; dx <= search_cells; ++dx) {
        const int x = center_x + dx;
        if (x < 0 || x >= width) continue;
        const size_t cell = static_cast<size_t>(y) * grid.info.width + x;
        if (obstacle_mask[cell]) continue;
        const double cell_x = origin_x + (x + 0.5) * resolution;
        const double cell_y = origin_y + (y + 0.5) * resolution;
        const double distance_sq =
            std::pow(cell_x - position.x(), 2) + std::pow(cell_y - position.y(), 2);
        if (distance_sq > coverage_seed_search_radius_ * coverage_seed_search_radius_ + 1e-9 ||
            distance_sq >= best_distance_sq)
          continue;
        best_distance_sq = distance_sq;
        seed_cell = cell;
        found = true;
      }
    }
    return found;
  }

  bool buildReachableCoverageMask(const nav_msgs::OccupancyGrid& grid) {
    if (!coverage_reachable_only_) {
      reachable_coverage_mask_.assign(grid.data.size(), 1);
      coverage_denominator_cells_ = reachable_coverage_mask_.size();
      coverage_denominator_area_m2_ = coverage_denominator_cells_ *
          static_cast<double>(grid.info.resolution) * grid.info.resolution;
      coverage_mask_grid_ = grid;
      coverage_mask_ready_ = true;
      return true;
    }
    if (!map_ready_) {
      ROS_WARN_THROTTLE(5.0, "Coverage mask is waiting for the ground-truth point cloud");
      return false;
    }
    for (int drone = 0; drone < drone_num_; ++drone) {
      if (!have_odom_[drone]) {
        ROS_WARN_THROTTLE(5.0, "Coverage mask is waiting for all UAV odometry seeds");
        return false;
      }
    }

    const double resolution = grid.info.resolution;
    if (resolution <= 0.0 || grid.data.empty()) return false;
    const int width = static_cast<int>(grid.info.width);
    const int height = static_cast<int>(grid.info.height);
    const double origin_x = grid.info.origin.position.x;
    const double origin_y = grid.info.origin.position.y;
    const double coverage_height = grid.info.origin.position.z;
    std::vector<uint8_t> obstacle_mask(grid.data.size(), 0);
    const int inflation_cells =
        std::max(0, static_cast<int>(std::ceil(coverage_obstacle_clearance_ / resolution)));

    for (const auto& point : obstacle_cloud_->points) {
      if (std::fabs(point.z - coverage_height) > coverage_slice_half_height_ + 0.5 * resolution)
        continue;
      const int center_x = static_cast<int>(std::floor((point.x - origin_x) / resolution));
      const int center_y = static_cast<int>(std::floor((point.y - origin_y) / resolution));
      for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
        const int y = center_y + dy;
        if (y < 0 || y >= height) continue;
        for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
          const int x = center_x + dx;
          if (x < 0 || x >= width) continue;
          const double cell_x = origin_x + (x + 0.5) * resolution;
          const double cell_y = origin_y + (y + 0.5) * resolution;
          const double distance_sq =
              std::pow(cell_x - point.x, 2) + std::pow(cell_y - point.y, 2);
          if (distance_sq <=
              coverage_obstacle_clearance_ * coverage_obstacle_clearance_ + 1e-9)
            obstacle_mask[static_cast<size_t>(y) * grid.info.width + x] = 1;
        }
      }
    }

    reachable_coverage_mask_.assign(grid.data.size(), 0);
    std::deque<size_t> frontier;
    for (int drone = 0; drone < drone_num_; ++drone) {
      size_t seed_cell = 0;
      if (!findFreeCoverageSeed(grid, obstacle_mask, latest_positions_[drone], seed_cell)) {
        ROS_ERROR("Cannot find a free coverage seed within %.2fm of UAV %d at (%.2f, %.2f)",
            coverage_seed_search_radius_, drone + 1, latest_positions_[drone].x(),
            latest_positions_[drone].y());
        continue;
      }
      if (!reachable_coverage_mask_[seed_cell]) {
        reachable_coverage_mask_[seed_cell] = 1;
        frontier.push_back(seed_cell);
      }
    }
    if (frontier.empty()) return false;

    while (!frontier.empty()) {
      const size_t cell = frontier.front();
      frontier.pop_front();
      const int x = static_cast<int>(cell % grid.info.width);
      const int y = static_cast<int>(cell / grid.info.width);
      const int neighbor_x[4] = { x + 1, x - 1, x, x };
      const int neighbor_y[4] = { y, y, y + 1, y - 1 };
      for (int neighbor = 0; neighbor < 4; ++neighbor) {
        const int nx = neighbor_x[neighbor];
        const int ny = neighbor_y[neighbor];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const size_t next = static_cast<size_t>(ny) * grid.info.width + nx;
        if (obstacle_mask[next] || reachable_coverage_mask_[next]) continue;
        reachable_coverage_mask_[next] = 1;
        frontier.push_back(next);
      }
    }

    coverage_denominator_cells_ = static_cast<size_t>(
        std::count(reachable_coverage_mask_.begin(), reachable_coverage_mask_.end(), uint8_t(1)));
    if (coverage_denominator_cells_ == 0) return false;
    coverage_denominator_area_m2_ = coverage_denominator_cells_ * resolution * resolution;
    coverage_mask_grid_ = grid;
    coverage_mask_ready_ = true;
    ROS_INFO("Reachable coverage mask ready: %zu cells, %.3f m^2 at z=%.2f m",
        coverage_denominator_cells_, coverage_denominator_area_m2_, coverage_height);
    return true;
  }

  double computeCoverage() {
    int reference = -1;
    for (int i = 0; i < drone_num_; ++i) {
      if (have_grid_[i] && !known_grids_[i].data.empty()) {
        reference = i;
        break;
      }
    }
    if (reference < 0) return std::numeric_limits<double>::quiet_NaN();

    const auto& reference_grid = known_grids_[reference];
    const size_t total_cells = reference_grid.data.size();
    if (total_cells == 0) return std::numeric_limits<double>::quiet_NaN();
    if (!coverage_mask_ready_ || !gridsCompatible(reference_grid, coverage_mask_grid_)) {
      coverage_mask_ready_ = false;
      if (!buildReachableCoverageMask(reference_grid))
        return std::numeric_limits<double>::quiet_NaN();
    }

    size_t known_cells = 0;
    for (size_t cell = 0; cell < total_cells; ++cell) {
      if (!reachable_coverage_mask_[cell]) continue;
      bool known = false;
      for (int drone = 0; drone < drone_num_; ++drone) {
        if (!have_grid_[drone]) continue;
        const auto& grid = known_grids_[drone];
        if (gridsCompatible(grid, reference_grid) && grid.data[cell] >= 0) {
          known = true;
          break;
        }
      }
      if (known) ++known_cells;
    }
    return 100.0 * static_cast<double>(known_cells) /
        static_cast<double>(coverage_denominator_cells_);
  }

  void coverageTimer(const ros::TimerEvent&) {
    if (!started_ || finished_) return;
    const double coverage = computeCoverage();
    if (!std::isfinite(coverage)) return;
    coverage_history_.push_back({ (ros::Time::now() - start_time_).toSec(), coverage });
  }

  void finishTimer(const ros::TimerEvent&) {
    if (!started_ || finished_) return;

    const ros::Time now = ros::Time::now();

    bool all_idle = true;
    for (int i = 0; i < drone_num_; ++i) {
      if (!have_state_[i] || states_[i] != idle_state_value_) {
        all_idle = false;
        break;
      }
    }

    if (all_idle) {
      if (all_idle_since_.isZero()) {
        all_idle_since_ = now;
        ROS_INFO("All %d UAVs are IDLE; validating for %.1f seconds", drone_num_,
            idle_hold_duration_);
      }
      if ((now - all_idle_since_).toSec() >= idle_hold_duration_) {
        finished_ = true;
        end_time_ = all_idle_since_;
        finish_reason_ = "all_idle";
        finalizeRun();
        return;
      }
    } else {
      all_idle_since_ = ros::Time(0);
    }

    bool all_stationary = true;
    for (int i = 0; i < drone_num_; ++i) {
      const bool odom_fresh = have_odom_[i] && !last_odom_times_[i].isZero() &&
          (now - last_odom_times_[i]).toSec() <= odom_freshness_timeout_;
      if (!odom_fresh || latest_velocities_[i].norm() > stationary_speed_threshold_) {
        all_stationary = false;
        break;
      }
    }

    if (!all_stationary) {
      all_stationary_since_ = ros::Time(0);
      return;
    }

    if (all_stationary_since_.isZero()) {
      all_stationary_since_ = now;
      ROS_INFO("All %d UAVs are stationary; validating for %.1f seconds", drone_num_,
          stationary_hold_duration_);
      return;
    }
    if ((now - all_stationary_since_).toSec() < stationary_hold_duration_) return;

    finished_ = true;
    end_time_ = all_stationary_since_;
    finish_reason_ = "all_stationary";
    finalizeRun();
  }

  double computeMeanTurnAngleDeg() {
    double sum_degrees = 0.0;
    size_t angle_count = 0;
    for (int drone = 0; drone < drone_num_; ++drone) {
      if (have_odom_[drone] && !turn_paths_[drone].empty() &&
          (latest_positions_[drone].head<2>() - turn_paths_[drone].back().head<2>()).norm() >
              1e-3) {
        turn_paths_[drone].push_back(latest_positions_[drone]);
      }
      const auto& points = turn_paths_[drone];
      for (size_t i = 1; i + 1 < points.size(); ++i) {
        const Eigen::Vector2d first = points[i].head<2>() - points[i - 1].head<2>();
        const Eigen::Vector2d second = points[i + 1].head<2>() - points[i].head<2>();
        const double first_norm = first.norm();
        const double second_norm = second.norm();
        if (first_norm < 1e-6 || second_norm < 1e-6) continue;
        const double cosine = std::max(-1.0,
            std::min(1.0, first.dot(second) / (first_norm * second_norm)));
        sum_degrees += std::acos(cosine) * 180.0 / kPi;
        ++angle_count;
      }
    }
    return angle_count == 0 ? std::numeric_limits<double>::quiet_NaN()
                            : sum_degrees / static_cast<double>(angle_count);
  }

  int nextRunNumber(const fs::path& runs_directory) const {
    int maximum = 0;
    if (!fs::exists(runs_directory)) return 1;
    for (fs::directory_iterator it(runs_directory), end; it != end; ++it) {
      if (!fs::is_directory(it->path())) continue;
      const std::string name = it->path().filename().string();
      if (name.size() < 10 || name.compare(0, 4, "run_") != 0) continue;
      try {
        maximum = std::max(maximum, std::stoi(name.substr(4, 6)));
      } catch (const std::exception&) {
      }
    }
    return maximum + 1;
  }

  PlanningStats computePlanningStats() const {
    PlanningStats stats;
    stats.uav_mean_time_s.assign(
        drone_num_, std::numeric_limits<double>::quiet_NaN());
    double maximum = 0.0;
    for (int i = 0; i < drone_num_; ++i) {
      stats.call_count += planning_call_counts_[i];
      stats.time_sum_s += planning_time_sums_[i];
      maximum = std::max(maximum, planning_time_maxima_[i]);
      if (planning_call_counts_[i] > 0)
        stats.uav_mean_time_s[i] =
            planning_time_sums_[i] / static_cast<double>(planning_call_counts_[i]);
    }
    if (stats.call_count > 0) {
      stats.mean_time_s = stats.time_sum_s / static_cast<double>(stats.call_count);
      stats.max_time_s = maximum;
    }
    return stats;
  }

  void finalizeRun() {
    const double mission_time = std::max(0.0, (end_time_ - start_time_).toSec());
    const double coverage = computeCoverage();
    if (std::isfinite(coverage) &&
        (coverage_history_.empty() || coverage_history_.back().elapsed_s < mission_time - 1e-6))
      coverage_history_.push_back({ mission_time, coverage });

    std::vector<double> drone_times(drone_num_, mission_time);
    double sum_flight_time = 0.0;
    for (int i = 0; i < drone_num_; ++i) {
      if (!idle_entry_times_[i].isZero())
        drone_times[i] = std::max(0.0, (idle_entry_times_[i] - start_time_).toSec());
      sum_flight_time += drone_times[i];
    }

    const double mean_turn_angle = computeMeanTurnAngleDeg();
    double total_path_length = 0.0;
    for (double length : path_lengths_) total_path_length += length;
    const PlanningStats planning_stats = computePlanningStats();

    int obstacle_collisions = 0;
    int uav_collisions = 0;
    for (const auto& event : collision_events_) {
      if (event.elapsed_s > mission_time + 1e-6) continue;
      if (event.type == "obstacle")
        ++obstacle_collisions;
      else if (event.type == "uav")
        ++uav_collisions;
    }

    try {
      const fs::path cr_directory = fs::path(output_root_) / scene_ /
          crDirectoryName(communication_range_);
      const fs::path runs_directory = cr_directory / "runs";
      fs::create_directories(runs_directory);
      const int run_number = nextRunNumber(runs_directory);
      std::ostringstream run_name;
      run_name << "run_" << std::setw(6) << std::setfill('0') << run_number << "_seed"
               << std::setw(3) << std::setfill('0') << seed_ << "_"
               << formatLocalTime(run_wall_start_, "%Y%m%d_%H%M%S");
      const std::string run_id = run_name.str();
      const fs::path run_directory = runs_directory / run_id;
      fs::create_directories(run_directory);

      writeExperimentYaml(cr_directory / "experiment.yaml");
      writeParamsYaml(run_directory / "params.yaml", run_id);
      writeTrajectories(run_directory / "trajectories.csv", mission_time);
      writeCoverageHistory(run_directory / "coverage_history.csv", mission_time);
      writeCollisionEvents(run_directory / "collision_events.csv", mission_time);
      writeMetricsJson(run_directory / "metrics.json", run_id, mission_time, drone_times,
          sum_flight_time, coverage, mean_turn_angle, total_path_length, obstacle_collisions,
          uav_collisions, planning_stats);
      appendSummary(cr_directory / "summary.csv", run_id, mission_time, drone_times,
          sum_flight_time, coverage, mean_turn_angle, total_path_length, obstacle_collisions,
          uav_collisions, planning_stats);

      ROS_INFO("Metrics saved: %s", run_directory.string().c_str());
      ROS_INFO(
          "Result: time=%.3fs coverage=%.3f%% mean_turn=%.3fdeg min_clearance=%.3fm "
          "collisions=%d mean_plan=%.3fms calls=%zu",
          mission_time, coverage, mean_turn_angle, min_obstacle_clearance_,
          obstacle_collisions + uav_collisions, planning_stats.mean_time_s * 1000.0,
          planning_stats.call_count);
    } catch (const std::exception& error) {
      ROS_ERROR("Failed to save experiment metrics: %s", error.what());
    }
  }

  void writeExperimentYaml(const fs::path& path) const {
    if (fs::exists(path)) return;
    std::ofstream out(path.string());
    requireOpen(out, path);
    out << "scene: " << scene_ << "\n";
    out << "communication_range: ";
    if (communication_range_ < 0.0)
      out << "inf\n";
    else
      out << communication_range_ << "\n";
    out << "drone_num: " << drone_num_ << "\n";
    out << "coverage_definition: union of known cells divided by ground-truth reachable free "
           "cells at the known-grid height\n";
    out << "turn_angle_definition: pooled XY turning angles after spatial resampling\n";
    out << "planning_time_definition: steady-clock elapsed time of each complete planning call, "
           "pooled across UAVs and attempts\n";
    out << "collision_definition: contact-entry events with hysteresis\n";
    out << "obstacle_distance_definition: body clearance to the ground-truth wall cloud\n";
  }

  void writeParamsYaml(const fs::path& path, const std::string& run_id) const {
    std::ofstream out(path.string());
    requireOpen(out, path);
    out << "run_id: " << run_id << "\n";
    out << "status: complete\n";
    out << "scene: " << scene_ << "\n";
    out << "seed: " << seed_ << "\n";
    out << "communication_range: ";
    if (communication_range_ < 0.0)
      out << "inf\n";
    else
      out << communication_range_ << "\n";
    out << "drone_num: " << drone_num_ << "\n";
    out << "idle_state_value: " << idle_state_value_ << "\n";
    out << "idle_hold_duration_s: " << idle_hold_duration_ << "\n";
    out << "stationary_hold_duration_s: " << stationary_hold_duration_ << "\n";
    out << "stationary_speed_threshold_mps: " << stationary_speed_threshold_ << "\n";
    out << "odom_freshness_timeout_s: " << odom_freshness_timeout_ << "\n";
    out << "finish_reason: " << finish_reason_ << "\n";
    out << "uav_radius_m: " << uav_radius_ << "\n";
    out << "collision_hysteresis_m: " << collision_hysteresis_ << "\n";
    out << "turn_sample_distance_m: " << turn_sample_distance_ << "\n";
    out << "trajectory_sample_period_s: " << trajectory_sample_period_ << "\n";
    out << "obstacle_check_period_s: " << obstacle_check_period_ << "\n";
    out << "obstacle_min_z_m: " << obstacle_min_z_ << "\n";
    out << "coverage_reachable_only: " << (coverage_reachable_only_ ? "true" : "false") << "\n";
    out << "coverage_obstacle_clearance_m: " << coverage_obstacle_clearance_ << "\n";
    out << "coverage_slice_half_height_m: " << coverage_slice_half_height_ << "\n";
    out << "coverage_seed_search_radius_m: " << coverage_seed_search_radius_ << "\n";
    out << "coverage_denominator_cells: " << coverage_denominator_cells_ << "\n";
    out << "coverage_denominator_area_m2: " << coverage_denominator_area_m2_ << "\n";
    out << "topics:\n";
    out << "  odom_prefix: " << odom_prefix_ << "\n";
    out << "  fsm_state_prefix: " << fsm_state_prefix_ << "\n";
    out << "  planning_time_prefix: " << planning_time_prefix_ << "\n";
    out << "  known_grid_prefix: " << known_grid_prefix_ << "\n";
    out << "  global_map: " << global_map_topic_ << "\n";
  }

  void writeTrajectories(const fs::path& path, double mission_time) const {
    std::ofstream out(path.string());
    requireOpen(out, path);
    out << "elapsed_s,drone_id,x,y,z,vx,vy,vz,fsm_state\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& sample : trajectory_samples_) {
      if (sample.elapsed_s > mission_time + 1e-6) continue;
      out << sample.elapsed_s << ',' << sample.drone_id << ',' << sample.position.x() << ','
          << sample.position.y() << ',' << sample.position.z() << ',' << sample.velocity.x() << ','
          << sample.velocity.y() << ',' << sample.velocity.z() << ',' << sample.fsm_state << '\n';
    }
  }

  void writeCoverageHistory(const fs::path& path, double mission_time) const {
    std::ofstream out(path.string());
    requireOpen(out, path);
    out << "elapsed_s,coverage_pct\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& sample : coverage_history_) {
      if (sample.elapsed_s <= mission_time + 1e-6)
        out << sample.elapsed_s << ',' << sample.coverage_pct << '\n';
    }
  }

  void writeCollisionEvents(const fs::path& path, double mission_time) const {
    std::ofstream out(path.string());
    requireOpen(out, path);
    out << "elapsed_s,type,uav_a,uav_b,x,y,z,center_distance_m,clearance_m\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& event : collision_events_) {
      if (event.elapsed_s > mission_time + 1e-6) continue;
      out << event.elapsed_s << ',' << event.type << ',' << event.uav_a << ',' << event.uav_b << ','
          << event.position.x() << ',' << event.position.y() << ',' << event.position.z() << ','
          << event.center_distance_m << ',' << event.clearance_m << '\n';
    }
  }

  void writeMetricsJson(const fs::path& path, const std::string& run_id, double mission_time,
      const std::vector<double>& drone_times, double sum_flight_time, double coverage,
      double mean_turn_angle, double total_path_length, int obstacle_collisions,
      int uav_collisions, const PlanningStats& planning_stats) const {
    std::ofstream out(path.string());
    requireOpen(out, path);
    out << "{\n";
    out << "  \"run_id\": \"" << run_id << "\",\n";
    out << "  \"status\": \"complete\",\n";
    out << "  \"finish_reason\": \"" << finish_reason_ << "\",\n";
    out << "  \"scene\": \"" << scene_ << "\",\n";
    out << "  \"cr_directory\": \"" << crDirectoryName(communication_range_) << "\",\n";
    out << "  \"seed\": " << seed_ << ",\n";
    out << "  \"start_ros_time_s\": ";
    writeJsonNumber(out, start_time_.toSec());
    out << ",\n  \"end_ros_time_s\": ";
    writeJsonNumber(out, end_time_.toSec());
    out << ",\n  \"mission_time_s\": ";
    writeJsonNumber(out, mission_time);
    out << ",\n  \"drone_flight_time_s\": [";
    for (size_t i = 0; i < drone_times.size(); ++i) {
      if (i) out << ", ";
      writeJsonNumber(out, drone_times[i]);
    }
    out << "],\n  \"sum_flight_time_s\": ";
    writeJsonNumber(out, sum_flight_time);
    out << ",\n  \"coverage_pct\": ";
    writeJsonNumber(out, coverage);
    out << ",\n  \"coverage_definition\": \"ground_truth_reachable_free_cells\"";
    out << ",\n  \"coverage_denominator_cells\": " << coverage_denominator_cells_;
    out << ",\n  \"coverage_denominator_area_m2\": ";
    writeJsonNumber(out, coverage_denominator_area_m2_);
    out << ",\n  \"mean_turn_angle_deg\": ";
    writeJsonNumber(out, mean_turn_angle);
    out << ",\n  \"min_obstacle_distance_m\": ";
    writeJsonNumber(out, min_obstacle_clearance_);
    out << ",\n  \"min_obstacle_center_distance_m\": ";
    writeJsonNumber(out, min_obstacle_center_distance_);
    out << ",\n  \"min_inter_uav_center_distance_m\": ";
    writeJsonNumber(out, min_inter_uav_center_distance_);
    out << ",\n  \"path_length_m\": [";
    for (size_t i = 0; i < path_lengths_.size(); ++i) {
      if (i) out << ", ";
      writeJsonNumber(out, path_lengths_[i]);
    }
    out << "],\n  \"total_path_length_m\": ";
    writeJsonNumber(out, total_path_length);
    out << ",\n  \"planning_call_count\": " << planning_stats.call_count;
    out << ",\n  \"planning_time_sum_s\": ";
    writeJsonNumber(out, planning_stats.time_sum_s);
    out << ",\n  \"mean_planning_time_s\": ";
    writeJsonNumber(out, planning_stats.mean_time_s);
    out << ",\n  \"max_planning_time_s\": ";
    writeJsonNumber(out, planning_stats.max_time_s);
    out << ",\n  \"uav_planning_call_count\": [";
    for (size_t i = 0; i < planning_call_counts_.size(); ++i) {
      if (i) out << ", ";
      out << planning_call_counts_[i];
    }
    out << "],\n  \"uav_mean_planning_time_s\": [";
    for (size_t i = 0; i < planning_stats.uav_mean_time_s.size(); ++i) {
      if (i) out << ", ";
      writeJsonNumber(out, planning_stats.uav_mean_time_s[i]);
    }
    out << "]";
    out << ",\n  \"obstacle_collision_count\": " << obstacle_collisions;
    out << ",\n  \"uav_collision_count\": " << uav_collisions;
    out << ",\n  \"total_collision_count\": " << obstacle_collisions + uav_collisions << "\n";
    out << "}\n";
  }

  void upgradeSummarySchema(const fs::path& path) const {
    if (!fs::exists(path) || fs::file_size(path) == 0) return;

    std::ifstream in(path.string());
    if (!in.is_open()) throw std::runtime_error("cannot open " + path.string());
    std::string header;
    if (!std::getline(in, header) || header.find("mean_planning_time_s") != std::string::npos)
      return;

    const fs::path temporary_path(path.string() + ".schema_upgrade.tmp");
    std::ofstream out(temporary_path.string(), std::ios::out | std::ios::trunc);
    requireOpen(out, temporary_path);
    out << header
        << ",planning_call_count,planning_time_sum_s,mean_planning_time_s,max_planning_time_s\n";
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty()) out << line << ",,,,\n";
    }
    in.close();
    out.flush();
    if (!out.good())
      throw std::runtime_error("failed while upgrading " + path.string());
    out.close();
    fs::rename(temporary_path, path);
  }

  void appendSummary(const fs::path& path, const std::string& run_id, double mission_time,
      const std::vector<double>& drone_times, double sum_flight_time, double coverage,
      double mean_turn_angle, double total_path_length, int obstacle_collisions,
      int uav_collisions, const PlanningStats& planning_stats) const {
    upgradeSummarySchema(path);
    const bool write_header = !fs::exists(path) || fs::file_size(path) == 0;
    std::ofstream out(path.string(), std::ios::out | std::ios::app);
    requireOpen(out, path);
    if (write_header) {
      out << "run_id,seed,start_time,mission_time_s";
      for (int i = 0; i < drone_num_; ++i) out << ",uav" << i + 1 << "_time_s";
      out << ",sum_flight_time_s,coverage_pct,mean_turn_angle_deg,min_obstacle_distance_m,"
             "min_obstacle_center_distance_m,min_inter_uav_center_distance_m,"
             "total_path_length_m,obstacle_collision_count,uav_collision_count,"
             "total_collision_count,planning_call_count,planning_time_sum_s,"
             "mean_planning_time_s,max_planning_time_s\n";
    }

    out << run_id << ',' << seed_ << ',' << formatLocalTime(run_wall_start_, "%Y-%m-%dT%H:%M:%S")
        << ',';
    writeCsvNumber(out, mission_time);
    for (double time : drone_times) {
      out << ',';
      writeCsvNumber(out, time);
    }
    out << ',';
    writeCsvNumber(out, sum_flight_time);
    out << ',';
    writeCsvNumber(out, coverage);
    out << ',';
    writeCsvNumber(out, mean_turn_angle);
    out << ',';
    writeCsvNumber(out, min_obstacle_clearance_);
    out << ',';
    writeCsvNumber(out, min_obstacle_center_distance_);
    out << ',';
    writeCsvNumber(out, min_inter_uav_center_distance_);
    out << ',';
    writeCsvNumber(out, total_path_length);
    out << ',' << obstacle_collisions << ',' << uav_collisions << ','
        << obstacle_collisions + uav_collisions << ',' << planning_stats.call_count << ',';
    writeCsvNumber(out, planning_stats.time_sum_s);
    out << ',';
    writeCsvNumber(out, planning_stats.mean_time_s);
    out << ',';
    writeCsvNumber(out, planning_stats.max_time_s);
    out << '\n';
    out.flush();
    if (!out.good()) throw std::runtime_error("failed while appending " + path.string());
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  std::vector<ros::Subscriber> odom_subs_, state_subs_, planning_time_subs_, grid_subs_;
  ros::Subscriber trigger_sub_, map_sub_;
  ros::Timer finish_timer_, coverage_timer_;

  int drone_num_ = 4;
  int seed_ = 0;
  int idle_state_value_ = 6;
  double communication_range_ = -1.0;
  double idle_hold_duration_ = 3.0;
  double stationary_hold_duration_ = 30.0;
  double stationary_speed_threshold_ = 0.05;
  double odom_freshness_timeout_ = 1.0;
  double uav_radius_ = 0.25;
  double collision_hysteresis_ = 0.05;
  double turn_sample_distance_ = 0.20;
  double trajectory_sample_period_ = 0.05;
  double obstacle_check_period_ = 0.02;
  double obstacle_min_z_ = 0.25;
  bool coverage_reachable_only_ = true;
  double coverage_obstacle_clearance_ = 0.25;
  double coverage_slice_half_height_ = 0.25;
  double coverage_seed_search_radius_ = 1.0;
  std::string scene_, output_root_, odom_prefix_, fsm_state_prefix_, planning_time_prefix_,
      known_grid_prefix_;
  std::string trigger_topic_, global_map_topic_;

  bool started_ = false;
  bool finished_ = false;
  bool map_ready_ = false;
  bool coverage_mask_ready_ = false;
  ros::Time start_time_, end_time_, all_idle_since_, all_stationary_since_;
  std::time_t run_wall_start_ = 0;
  std::string finish_reason_;

  std::vector<int> states_;
  std::vector<bool> have_state_, have_odom_, have_last_path_position_, obstacle_contact_, have_grid_;
  std::vector<Eigen::Vector3d> latest_positions_, latest_velocities_, last_path_positions_;
  std::vector<double> path_lengths_;
  std::vector<size_t> planning_call_counts_;
  std::vector<double> planning_time_sums_, planning_time_maxima_;
  std::vector<std::vector<Eigen::Vector3d>> turn_paths_;
  std::vector<ros::Time> last_trajectory_sample_time_, last_obstacle_check_time_, last_odom_times_,
      idle_entry_times_;
  std::vector<std::vector<bool>> pair_contact_;
  std::vector<nav_msgs::OccupancyGrid> known_grids_;
  nav_msgs::OccupancyGrid coverage_mask_grid_;
  std::vector<uint8_t> reachable_coverage_mask_;
  size_t coverage_denominator_cells_ = 0;
  double coverage_denominator_area_m2_ = 0.0;

  std::vector<TrajectorySample> trajectory_samples_;
  std::vector<CoverageSample> coverage_history_;
  std::vector<CollisionEvent> collision_events_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud_;
  pcl::KdTreeFLANN<pcl::PointXYZ> obstacle_kdtree_;
  double min_obstacle_center_distance_ = std::numeric_limits<double>::infinity();
  double min_obstacle_clearance_ = std::numeric_limits<double>::infinity();
  double min_inter_uav_center_distance_ = std::numeric_limits<double>::infinity();
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "experiment_metrics");
  ExperimentMetricsNode node;
  ros::spin();
  return 0;
}
