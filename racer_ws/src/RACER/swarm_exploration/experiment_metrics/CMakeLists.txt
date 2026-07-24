cmake_minimum_required(VERSION 2.8.3)
project(experiment_metrics)

add_compile_options(-std=c++14 -O2 -Wall)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  pcl_conversions
  roscpp
  sensor_msgs
  std_msgs
)
find_package(PCL 1.7 REQUIRED COMPONENTS common kdtree)
find_package(Boost REQUIRED COMPONENTS filesystem system)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})

add_executable(experiment_metrics_node src/experiment_metrics_node.cpp)
target_link_libraries(experiment_metrics_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)
add_dependencies(experiment_metrics_node ${catkin_EXPORTED_TARGETS})

install(TARGETS experiment_metrics_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
