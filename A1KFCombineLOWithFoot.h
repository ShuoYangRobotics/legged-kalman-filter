#pragma once

#include <mutex>

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>

#include "A1KF.hpp"
#include "../legKinematics/A1Kinematics.h"

/*
   Baseline 3
   This filter follows http://www.roboticsproceedings.org/rss08/p03.pdf
   We do not consider bias here

   state = [x, y, z, vx, vy, vz, euler_roll, euler_pitch, euler_yaw, foot1, foot2, foot3, foot4, tk]
           [0, 1, 2,  3,  4,  5,  6,        7,        8,    9:11, 12:14, 15:17. 18:20    21]

   ctrl = [acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z, dt]
            [0,    1,    2,     3,        4,        5,        6]

   observation = [foot1 pos, foot2 pos, foot3 pos, foot4 pos, foot1 vel, foot2 vel, foot3 vel, foot4 vel]
                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

 */
#define EKF_STATE_SIZE 22
#define CONTROL_SIZE 7
#define OBSERVATION_SIZE 24

#define OPTI_OBSERVATION_SIZE 6 // observe position and velocity using optitrack
#define UPDATE_DT 0.002

class A1KFCombineLOWithFoot : public A1KF
{
public:
    A1KFCombineLOWithFoot();
    void init_filter(A1SensorData data, Eigen::Vector3d _init_pos = Eigen::Vector3d(0, 0, 0.15));
    void update_filter(A1SensorData data);
    void update_filter_with_opti(A1SensorData data);

    Eigen::Matrix<double, EKF_STATE_SIZE, 1> get_state() { return curr_state; }
    Eigen::Matrix<double, NUM_LEG, 1> get_contacts() { return estimated_contact; }

private:
    void load_casadi_functions();
    void process(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state,
                 Eigen::Matrix<double, CONTROL_SIZE, 1> prev_ctrl,
                 Eigen::Matrix<double, CONTROL_SIZE, 1> ctrl, double dt);

    void measure(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state,
                 Eigen::Matrix<double, 3, 1> w,
                 Eigen::Matrix<double, 12, 1> joint_ang,
                 Eigen::Matrix<double, 12, 1> joint_vel);
    Eigen::Matrix<double, EKF_STATE_SIZE, 1> curr_state;
    Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE> curr_covariance;

    Eigen::Matrix<double, EKF_STATE_SIZE, 1> x01;                           // intermediate state
    Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE> process_jacobian; // intermediate covariance
    Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE> P01;              // intermediate covariance

    Eigen::Matrix<double, OBSERVATION_SIZE, 1> measurement;
    Eigen::Matrix<double, OBSERVATION_SIZE, EKF_STATE_SIZE> measurement_jacobian;

    Eigen::Matrix<double, CONTROL_SIZE, 1> prev_ctrl;
    Eigen::Matrix<double, CONTROL_SIZE, 1> curr_ctrl;

    Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE> process_noise;
    Eigen::Matrix<double, OBSERVATION_SIZE, OBSERVATION_SIZE> measure_noise;

    // optitrack related
    Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, EKF_STATE_SIZE> opti_jacobian;
    Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, OPTI_OBSERVATION_SIZE> opti_noise;

    std::mutex update_mutex;

    casadi::Function process_func;
    casadi::Function process_jac_func;
    casadi::Function measure_func;
    casadi::Function measure_jac_func;

    A1Kinematics kinematics;
    double leg_offset_x[4] = {};
    double leg_offset_y[4] = {};
    // for each leg, there is an offset between the body frame and the hip motor (fx, fy)
    double motor_offset[4] = {};
    double upper_leg_length[4] = {};
    double lower_leg_length[4] = {};
    std::vector<Eigen::VectorXd> rho_fix_list;
    std::vector<Eigen::VectorXd> rho_opt_list;

    // estimated contact (see which foot velocity agrees with state_v)
    Eigen::Matrix<double, NUM_LEG, 1> estimated_contact;

    // helper matrices
    Eigen::Matrix<double, 3, 3> eye3; // 3x3 identity
    Eigen::Matrix<double, OBSERVATION_SIZE, OBSERVATION_SIZE> S;
};