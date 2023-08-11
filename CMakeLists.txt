# search for casadi in the environment
find_package(casadi REQUIRED) 

# Find the savitzky golay filter package and all its dependencies 
find_package(gram_savitzky_golay REQUIRED)

include_directories(
  ${CASADI_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ../../../include
)

# libraries
add_library(kf_lib 
        A1KF.hpp
        A1KFCombineLO.cpp
        A1KFSeparateLO.cpp
        A1KFCombineLOWithFoot.cpp
        A1KFCombineLOWithFootTerrain.cpp
        ../../utils/A1Kinematics.cpp
)
target_link_libraries(kf_lib
        ${catkin_LIBRARIES}
        Eigen3::Eigen
)

# compile casadi_lib
execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/casadi_lib/compile_casadi.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/casadi_lib
)
