# search for casadi in the environment
find_package(casadi REQUIRED)

# Find the savitzky golay filter package and all its dependencies
find_package(gram_savitzky_golay REQUIRED)

include_directories(
  ${CASADI_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

# compile casadi_lib
add_custom_target(compile_casadi_lib
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/casadi_lib/compile_casadi.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/casadi_lib
)

# libraries
add_library(kf_lib
  A1KF.hpp
  A1KFCombineLO.cpp
  A1KFFootIMU.cpp
)
target_link_libraries(kf_lib
  ${catkin_LIBRARIES}
  casadi
  gram_savitzky_golay::gram_savitzky_golay
)
add_dependencies(kf_lib
  compile_casadi_lib
)

# TEST executables
add_executable(test_run_bag
  test/run_bag.cpp
)
target_link_libraries(test_run_bag
  kf_lib
)

add_executable(test_casadi_lib
  test/test_casadi_lib.cpp
)
target_link_libraries(test_casadi_lib
  casadi
)