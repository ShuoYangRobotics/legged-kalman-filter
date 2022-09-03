#pragma once

#include <mutex>

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include "A1KF.hpp"

#include <legged-kalman-filter/AccessHelper.h>

/*
%   - position        (0:2)
%   - velocity        (3:5)
%   - euler angle     (6:8)
%   - foot1 pos       (9:11)
%   - foot1 vel       (12:14)
%   - foot2 pos       (15:17)
%   - foot2 vel       (18:20)
%   - foot3 pos       (21:23)
%   - foot3 vel       (24:26)
%   - foot4 pos       (27:29)
%   - foot4 vel       (30:32)

% control u
%   - w      (0:2)     body IMU angular veolocity
%   - a      (3:5)     body IMU acceleration
%   - a1     (6:8)     foot 1 IMU acceleration (already in body frame)
%   - a2     (9:11)    foot 2 IMU acceleration (already in body frame)
%   - a3     (12:14)   foot 3 IMU acceleration (already in body frame)
%   - a4     (15:17)   foot 4 IMU acceleration (already in body frame)
*/

constexpr int EKF_STATE_SIZE = 33;
constexpr int CONTROL_SIZE = 18;
constexpr int OBSERVATION_SIZE = 18;
constexpr int OPTI_OBSERVATION_SIZE = 6;  // observe position
constexpr double UPDATE_DT = 0.002;

class A1KFFootIMU : public A1KF {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using vState = Eigen::Matrix<double, EKF_STATE_SIZE, 1>;
  using vControl = Eigen::Matrix<double, CONTROL_SIZE, 1>;
  using VMeasure = Eigen::Matrix<double, OBSERVATION_SIZE, 1>;

  A1KFFootIMU();
  void init_filter(A1FootIMUSensorData& data, const Eigen::Vector3d& _init_pos = Eigen::Vector3d(0, 0, 0.15));
  void update_filter(A1FootIMUSensorData& data);
  void update_filter_with_opti(A1FootIMUSensorData& data);

  vState get_state() const { return curr_state; }

 private:
  void load_casadi_functions();
  void reset();
  void process(const vState& state, const vControl& prev_ctrl, const vControl& ctrl, double dt);

  // Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE> process_jac(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state,
  //                                              Eigen::Matrix<double, CONTROL_SIZE, 1> prev_ctrl,
  //                                              Eigen::Matrix<double, CONTROL_SIZE, 1> ctrl, double dt);

  void measure(const vState& state, const Eigen::Matrix<double, 3, 1>& w, const Eigen::Matrix<double, 12, 1>& joint_ang,
               const Eigen::Matrix<double, 12, 1>& joint_vel, const Eigen::Matrix<double, 4, 1>& contact);

  // Eigen::Matrix<double, OBSERVATION_SIZE, EKF_STATE_SIZE> meas_jac(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state,
  //                                              Eigen::Matrix<double, 3, 1> w,
  //                                              Eigen::Matrix<double, 12, 1> joint_ang,
  //                                              Eigen::Matrix<double, 12, 1> joint_vel,
  //                                              Eigen::Matrix<double, 4, 1> contact);

  Eigen::Matrix<double, EKF_STATE_SIZE, 1> curr_state;
  Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE> curr_covariance;

  Eigen::Matrix<double, EKF_STATE_SIZE, 1> x01;                            // intermediate state
  Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE> process_jacobian;  // intermediate covariance
  Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE> P01;               // intermediate covariance

  Eigen::Matrix<double, OBSERVATION_SIZE, 1> measurement;
  Eigen::Matrix<double, OBSERVATION_SIZE, EKF_STATE_SIZE> measurement_jacobian;

  Eigen::Matrix<double, CONTROL_SIZE, 1> prev_ctrl;
  Eigen::Matrix<double, CONTROL_SIZE, 1> curr_ctrl;

  Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE> process_noise;
  Eigen::Matrix<double, OBSERVATION_SIZE, OBSERVATION_SIZE> measure_noise;

  // optitrack related
  Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, EKF_STATE_SIZE> opti_jacobian;
  Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, OPTI_OBSERVATION_SIZE> opti_noise;

  mutable std::mutex update_mutex;

  casadi::Function process_func;
  casadi::Function process_jac_func;
  casadi::Function measure_func;
  casadi::Function measure_jac_func;
};