cmake_minimum_required(VERSION 2.8.3)
project(robotsp)

find_package(catkin REQUIRED COMPONENTS
  criutils
  glkh_solver
  lkh_solver
  raveutils
)

catkin_python_setup()

catkin_package()

# Tests
if (CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(tests)
endif()
