# search for casadi in the environment
find_package(casadi REQUIRED) 

# Find the savitzky golay filter package and all its dependencies 
find_package(gram_savitzky_golay REQUIRED)

include_directories(
  ${CASADI_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

# libraries
add_library(kf_lib 
        A1KF.hpp
        A1KFCombineLO.cpp
        A1KFSeparateLO.cpp
        A1KFCombineLOWithFoot.cpp
        ../legKinematics/A1Kinematics.cpp
)

# compile casadi_lib
execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/casadi_lib/compile_casadi.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)



# executables
add_executable(test_run_bag test/run_bag.cpp)
target_link_libraries(test_run_bag
        kf_lib
        ${catkin_LIBRARIES}
        casadi
        gram_savitzky_golay::gram_savitzky_golay
        )


add_executable(test_casadi_lib test/test_casadi_lib.cpp)
target_link_libraries(test_casadi_lib
        casadi
)

add_executable(test_rotation_utils test/test_rotations.cpp

        ../utils/Utils.cpp
        ../utils/Utils.h
)        