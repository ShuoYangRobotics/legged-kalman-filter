Carefully study legged robot state estimation


We use casadi to generate EKF functions



Development work flow:

1. in Matlab code repo leggedrobot-velocity-fusion, generate casadi code 
(read leggedrobot-velocity-fusion/baseline_atti_lo_filters/CODE_GEN.m)

2. copy generated c code to src/a1_cpp/src/estimation/casadi_lib/matlab_code_gen
(leggedrobot-velocity-fusion/baseline_atti_lo_filters/copy_lib.sh does this if all repo paths are correctly configured)

3. The CMakeLists.txt in this folder should compile c code to .so libs during catkin build. The script also copies genereated lib to /tmp foloder

4. in A1KFCombineLO, load casadi function



Code structure explained: (read test/run_bag.cpp for a complete flow)
When /hardare_a1/imu, /hardware_a1/joint_foot, /mocap_node/Robot_1/pose are presented in ROS environment (now we publish rosbag to get them), then two sensor callbacks are called to fill sensor data to A1SensorData. This class also contains signal filtering mechanisms to process sensor data. Sensor callbacks also init KF and update KF.



Alternatively, we can instantiate A1KFCombineLO directly in robot controller.

TODO: 
1. replace A1BasicEKF with A1KFCombineLO
2. implement A1KFCombineLOWithFoot (baseline 3 in leggedrobot-velocity-fusion)