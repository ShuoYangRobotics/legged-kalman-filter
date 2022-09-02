#pragma once

#include <deque>
#include <mutex>

#include <gram_savitzky_golay/gram_savitzky_golay.h>
#include <Eigen/Dense>

#include "filter.hpp"

constexpr int NUM_LEG = 4;
constexpr int NUM_DOF = 12;

// This class is a common data structure of all filters,
// The state estimation requires IMU data (acceleration, angular velocity) and joint data (joint angles)
// optionally, it can take in joint velocity data and optitrack data
// for IMU data we apply mean filter to remove noise
// for joint angle data, we apply mean filter to remove noise and then use SavitzkyGolayFilter to get joint velocity
// For optitrack data we also use SavitzkyGolayFilter to get velocity
class A1SensorData {
 public:
  A1SensorData() {
    for (size_t i = 0; i < 3; ++i) {
      acc_filter[i] = MovingWindowFilter(30);
      ang_vel_filter[i] = MovingWindowFilter(15);
      opti_euler_filter[i] = MovingWindowFilter(15);

      opti_pos_filter[i] = MovingWindowFilter(15);
      opti_vel_filter_sgolay[i] = gram_sg::SavitzkyGolayFilter(sgolay_order, sgolay_order, sgolay_order, 1);
    }
    for (size_t i = 0; i < NUM_DOF; ++i) {
      joint_pos_filter[i] = MovingWindowFilter(15);
      joint_vel_filter[i] = MovingWindowFilter(15);
      joint_vel_filter_sgolay[i] = gram_sg::SavitzkyGolayFilter(sgolay_order, sgolay_order, sgolay_order, 1);
    }
    dt = 0.001;       // because hardware_imu is at 1000Hz
    opti_dt = 0.005;  // because optitrack is at 200Hz
    data_lock = new std::mutex();
  }

  /* IMU and joint data */
  void input_imu(Eigen::Matrix<double, 3, 1> acc, Eigen::Matrix<double, 3, 1> ang_vel) {
    for (size_t i = 0; i < 3; ++i) {
      this->acc[i] = acc_filter[i].CalculateAverage(acc[i]);
      this->ang_vel[i] = ang_vel_filter[i].CalculateAverage(ang_vel[i]);
    }
  }

  void input_leg(Eigen::Matrix<double, NUM_DOF, 1> joint_pos, Eigen::Matrix<double, NUM_DOF, 1> joint_vel,
                 Eigen::Matrix<double, NUM_LEG, 1> contact) {
    for (size_t i = 0; i < NUM_DOF; ++i) {
      this->joint_pos[i] = joint_pos_filter[i].CalculateAverage(joint_pos[i]);
      // this->joint_vel[i] = joint_vel_filter[i].CalculateAverage(joint_vel[i]);

      if (sgolay_values[i].size() < sgolay_frame) {
        this->joint_vel[i] = joint_vel_filter[i].CalculateAverage(joint_vel[i]);
        sgolay_values[i].push_back(joint_pos[i]);
        joint_sglolay_initialized = false;
      } else {
        sgolay_values[i].pop_front();
        sgolay_values[i].push_back(joint_pos[i]);
        this->joint_vel[i] = joint_vel_filter_sgolay[i].filter(sgolay_values[i]) / average_dt;
        joint_sglolay_initialized = true;
      }
    }
    this->plan_contacts = contact;
  }

  void input_dt(double dt) {
    this->dt = dt;
    dt_values.push_back(dt);
    if (dt_values.size() > sgolay_frame) {
      dt_values.pop_front();
    }
    average_dt = 0.0;
    for (size_t i = 0; i < dt_values.size(); ++i) {
      average_dt += dt_values[i];
    }
    average_dt /= dt_values.size();
  }

  /* opti track data */
  void input_opti_pos(Eigen::Matrix<double, 3, 1> _opti_pos) {
    data_lock->lock();
    for (size_t i = 0; i < 3; i++) {
      this->opti_pos[i] = opti_pos_filter[i].CalculateAverage(_opti_pos[i]);

      if (opti_sgolay_values[i].size() < sgolay_frame) {
        this->opti_vel[i] = 0.0;
        opti_sgolay_values[i].push_back(_opti_pos[i]);
        opti_sglolay_initialized = false;
      } else {
        opti_sglolay_initialized = true;
        opti_sgolay_values[i].pop_front();
        opti_sgolay_values[i].push_back(_opti_pos[i]);
        this->opti_vel[i] = opti_vel_filter_sgolay[i].filter(opti_sgolay_values[i]) / opti_average_dt;
      }
    }
    data_lock->unlock();
  }

  void input_opti_euler(Eigen::Vector3d euler_angs) {
    for (size_t i = 0; i < 3; i++) {
      this->opti_euler[i] = opti_euler_filter[i].CalculateAverage(euler_angs[i]);
    }
  }

  bool opti_vel_ready() { return opti_sglolay_initialized; }
  bool joint_vel_ready() { return joint_sglolay_initialized; }

  void input_opti_dt(double opti_dt) {
    data_lock->lock();
    this->opti_dt = opti_dt;
    opti_dt_values.push_back(opti_dt);
    if (opti_dt_values.size() > sgolay_frame) {
      opti_dt_values.pop_front();
    }
    opti_average_dt = 0.0;
    for (size_t i = 0; i < opti_dt_values.size(); ++i) {
      opti_average_dt += opti_dt_values[i];
    }
    opti_average_dt /= opti_dt_values.size();
    data_lock->unlock();
  }
  // data in IMU and jointState
  Eigen::Vector3d acc;
  Eigen::Vector3d ang_vel;
  Eigen::Vector3d opti_euler;
  Eigen::Matrix<double, NUM_DOF, 1> joint_pos;
  Eigen::Matrix<double, NUM_DOF, 1> joint_vel;
  Eigen::Matrix<double, NUM_LEG, 1> plan_contacts;
  double dt;
  double average_dt;

  // data in optitrack position
  Eigen::Vector3d opti_pos;
  Eigen::Vector3d opti_vel;  // use SavitzkyGolayFilter to get smoothed velocity
  bool joint_sglolay_initialized = false;
  bool opti_sglolay_initialized = false;
  double opti_dt;
  double opti_average_dt;

 private:
  /* filters for IMU/joint data */
  MovingWindowFilter acc_filter[3];
  MovingWindowFilter ang_vel_filter[3];
  MovingWindowFilter joint_pos_filter[NUM_DOF];
  MovingWindowFilter joint_vel_filter[NUM_DOF];
  gram_sg::SavitzkyGolayFilter joint_vel_filter_sgolay[NUM_DOF];
  std::deque<double> sgolay_values[NUM_DOF];  // for each dimension, store the values of the sgolay filter
  std::deque<double> dt_values;

  /* filters for opti track data */
  MovingWindowFilter opti_pos_filter[3];
  MovingWindowFilter opti_euler_filter[3];
  gram_sg::SavitzkyGolayFilter opti_vel_filter_sgolay[3];
  std::deque<double> opti_sgolay_values[3];
  std::deque<double> opti_dt_values;  // optitrack dt is different from hardware_imu/joint_foot dt

  // common SavitzkyGolay filter parameters
  const size_t sgolay_order = 7;
  const size_t sgolay_frame = 15;  // must be sgolay_order*2+1

  std::mutex* data_lock;
};

class A1KF {
 public:
  A1KF(){};
  bool is_initialized() { return KF_initialized; }

  bool KF_initialized = false;
};