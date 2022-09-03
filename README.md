# Foot IMU support

## Get started

1. 
```
git clone git@github.com:RoboticExplorationLab/RosDockerWorkspace.git leggged_ws
cd leggged_ws
git checkout ws/legged-KF
cp .gitmodules.example .gitmodules  # [optional - hack for nested git repo in Vscode]

cd src
git clone git@github.com:ShuoYangRobotics/legged-kalman-filter.git
cd legged-kalman-filter
git checkout feature/footIMU
```

2. Reopen workspace in container

3. `catkin build`

## To-do

See inline TODO comments.


