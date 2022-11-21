# search for casadi in the environment
set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_BUILD_TYPE "Debug")
find_package(casadi REQUIRED) 

# Find the savitzky golay filter package and all its dependencies 
find_package(gram_savitzky_golay REQUIRED)


# add pinocchio 
find_package(PkgConfig REQUIRED)
pkg_check_modules(pinocchio REQUIRED pinocchio)
find_package(Boost REQUIRED)

include_directories(
  ${CASADI_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ../../include
  mhe_utils
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

add_library(kf_qp_lib
        A1KF.hpp
        A1KFQP.cpp
)

# Find OSQP
include_directories(
        /usr/local/include/osqp
)
find_package(OsqpEigen REQUIRED)


# compile casadi_lib
execute_process(
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/casadi_lib/compile_casadi.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/casadi_lib
)



# executables
add_executable(test_run_bag test/run_bag.cpp)
target_link_libraries(test_run_bag
        kf_lib
        kf_qp_lib
        ${catkin_LIBRARIES}
        casadi
        gram_savitzky_golay::gram_savitzky_golay
        OsqpEigen::OsqpEigen 
        )


add_executable(test_casadi_lib test/test_casadi_lib.cpp)
target_link_libraries(test_casadi_lib
        # ${catkin_LIBRARIES}
        casadi
        ${pinocchio_LIBRARIES}
)

add_executable(test_pinocchio_unitree test/test_pinocchio_unitree.cpp)
target_link_libraries(test_pinocchio_unitree
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        casadi
)


add_executable(test_pinocchio_dynamics test/test_pinocchio_dynamics.cpp)
target_link_libraries(test_pinocchio_dynamics
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        casadi
        ${Boost_LIBRARIES}
)

# ocs2 is found by outer cmake list file
# target_compile_options(test_casadi_lib PUBLIC ${OCS2_CXX_FLAGS})   
target_compile_options(test_pinocchio_unitree PUBLIC ${OCS2_CXX_FLAGS})   

# should be moved to main code base
#add_executable(test_rotation_utils test/test_rotations.cpp
#
#        ../utils/Utils.cpp
#        ../utils/Utils.h
#)        



add_executable(test_mhe test/test_mhe.cpp)
target_link_libraries(test_mhe
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        casadi
)