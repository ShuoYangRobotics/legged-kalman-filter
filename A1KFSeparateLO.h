#pragma once

#include <mutex>

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include "A1KF.hpp"


/* 
   Baseline 2
   This filter fuses the imu and joint foot data to estimate the state of the system.
   from the imu data, the filter estimates the acceleration, angular velocity and position.
   from the joint foot data, the filter estimates the foot velocity and infer the body velocity
   We combine the two estimates in EKF
   The function of EKF are generated by casadi 

   state = [x, y, z, vx, vy, vz, euler_roll, euler_pitch, euler_yaw, tk]
           [0, 1, 2,  3,  4,  5,  6,        7,        8,        9,        10]

   ctrl = [acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z, dt]
            [0,    1,    2,     3,        4,        5,        6]

   observation = [foot 1 lo velocity, foot 2 lo velocity, foot 3 lo velocity, foot 4 lo velocity]
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

                
 */
#define EKF_STATE_SIZE 10
#define CONTROL_SIZE 7
#define OBSERVATION_SIZE 12   // observe leg odometry velocity
#define OPTI_OBSERVATION_SIZE 6 // observe position and velocity using optitrack
#define UPDATE_DT 0.002


class A1KFSeparateLO : public A1KF {
    public:
        A1KFSeparateLO ();
        void init_filter(A1SensorData data);
        void update_filter(A1SensorData data);
        void update_filter_with_opti(A1SensorData data);

        Eigen::Matrix<double, EKF_STATE_SIZE,1> get_state() {return curr_state;}

    private:

        void load_casadi_functions();
        void process(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state, 
                                                     Eigen::Matrix<double, CONTROL_SIZE, 1> prev_ctrl, 
                                                     Eigen::Matrix<double, CONTROL_SIZE, 1> ctrl, double dt);

        void measure(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state, 
                                                     Eigen::Matrix<double, 3, 1> w, 
                                                     Eigen::Matrix<double, 12, 1> joint_ang, 
                                                     Eigen::Matrix<double, 12, 1> joint_vel);

        Eigen::Matrix<double, EKF_STATE_SIZE, 1>   curr_state;                  
        Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>   curr_covariance;

        Eigen::Matrix<double, EKF_STATE_SIZE, 1>   x01;           // intermediate state
        Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>   process_jacobian;  // intermediate covariance
        Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>   P01;  // intermediate covariance


        Eigen::Matrix<double, OBSERVATION_SIZE, 1>             measurement;  
        Eigen::Matrix<double, OBSERVATION_SIZE, EKF_STATE_SIZE>    measurement_jacobian;  

        Eigen::Matrix<double, CONTROL_SIZE, 1> prev_ctrl;
        Eigen::Matrix<double, CONTROL_SIZE, 1> curr_ctrl;

        Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>   process_noise;
        Eigen::Matrix<double, OBSERVATION_SIZE, OBSERVATION_SIZE>   measure_noise;


        // optitrack related 
        Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, EKF_STATE_SIZE> opti_jacobian;
        Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, OPTI_OBSERVATION_SIZE> opti_noise;

        std::mutex update_mutex;

        casadi::Function process_func;
        casadi::Function process_jac_func;
        casadi::Function measure_func;
        casadi::Function measure_jac_func;
        // helper matrices
        Eigen::Matrix<double, 3, 3> eye3; // 3x3 identity
};