#pragma once

#include <mutex>

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>

#include "A1KF.hpp"
#include "utils/A1Kinematics.h"
#include "OsqpEigen/OsqpEigen.h"

/* 
   Use QP for KF update step 
   
   A Constrained Kalman Filter for Rigid Body Systems with Frictional Contact
   Algorithm 1 with slight tweak
                
 */
#define EKF_STATE_SIZE 22
#define CONTROL_SIZE 7
#define OBSERVATION_SIZE 28
#define OBS_PER_LEG 7

#define OPTI_OBSERVATION_SIZE 7 // observe position, velocity, and yaw using optitrack
#define UPDATE_DT 0.002


class A1KFQP : public A1KF {
    public:
        A1KFQP ();
        void init_filter(A1SensorData& data, Eigen::Vector3d _init_pos = Eigen::Vector3d(0,0,0.15));
        void update_filter(A1SensorData& data);

        Eigen::Matrix<double, EKF_STATE_SIZE,1> get_state() {return curr_state;}
        Eigen::Matrix<double, NUM_LEG, 1> get_contacts() {return estimated_contact;}

        // new function, set noise constants
        void set_noise_params(double _inital_cov = 0.01,
                              double _noise_process_pos_xy = 0.001,
                              double _noise_process_pos_z = 0.001,
                              double _noise_process_vel_xy = 0.001,
                              double _noise_process_vel_z = 0.01,
                              double _noise_process_rot = 1e-6,
                              double _noise_process_foot = 0.001,
                              double _noise_measure_fk = 0.01,
                              double _noise_measure_vel = 0.01,
                              double _noise_measure_height = 0.0001,
                              double _noise_opti_pos = 0.001,
                              double _noise_opti_vel = 999.0,
                              double _noise_opti_yaw = 0.01);

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

        // noise constants
        double inital_cov = 0.001;
        double noise_process_pos_xy = 0.001;
        double noise_process_pos_z = 0.001;
        double noise_process_vel_xy = 0.001;
        double noise_process_vel_z = 0.01;
        double noise_process_rot = 1e-6;
        double noise_process_foot = 0.001;

        double noise_measure_fk = 0.01;
        double noise_measure_vel = 0.01;
        double noise_measure_height = 0.0001;

        double noise_opti_pos = 0.001;
        double noise_opti_vel = 999.0;
        double noise_opti_yaw = 0.01;

        std::mutex update_mutex;

        casadi::Function process_func;
        casadi::Function process_jac_func;
        casadi::Function measure_func;
        casadi::Function measure_jac_func;

        legged::A1Kinematics kinematics;       
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

        // OSQP
        // OsqpEigen::Solver solver;
        Eigen::VectorXd lowerBound_; 
        Eigen::VectorXd upperBound_; 
        Eigen::VectorXd gradient_; 
        Eigen::MatrixXd hessian_; 
        Eigen::SparseMatrix<double> sparse_hessian_;

};