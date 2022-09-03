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
class A1SensorDataInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using vAcc = Eigen::Matrix<double, 3, 1>;
  using vAngVel = Eigen::Matrix<double, 3, 1>;
  using vOptiPos = Eigen::Matrix<double, 3, 1>;
  using vJointPos = Eigen::Matrix<double, NUM_DOF, 1>;
  using vJointVel = Eigen::Matrix<double, NUM_DOF, 1>;

  enum ImuIndex {
    BODY_IMU = 0,
    FOOT_FL_IMU,  // Front left
    FOOT_FR_IMU,  // Front right
    FOOT_RL_IMU,  // Rear left
    FOOT_RR_IMU,  // Rear right
    IMU_MAX_SIZE  // Helper enum
  };

  A1SensorDataInterface() {
    for (size_t i = 0; i < 3; ++i) {
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
  }

  /* IMU and joint data */
  virtual void input_imu(const vAcc& acc, const vAngVel& ang_vel, ImuIndex imu_index) = 0;

  void input_leg(const vJointPos& joint_pos, const vJointVel& joint_vel, const Eigen::Matrix<double, NUM_LEG, 1>& contact) {
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
  void input_opti_pos(const vOptiPos& _opti_pos) {
    const std::lock_guard<std::mutex> lock(data_lock);

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
  }

  void input_opti_euler(const Eigen::Vector3d& euler_angs) {
    for (size_t i = 0; i < 3; i++) {
      this->opti_euler[i] = opti_euler_filter[i].CalculateAverage(euler_angs[i]);
    }
  }

  bool opti_vel_ready() { return opti_sglolay_initialized; }
  bool joint_vel_ready() { return joint_sglolay_initialized; }

  void input_opti_dt(double opti_dt) {
    const std::lock_guard<std::mutex> lock(data_lock);

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
  }

  // Data in IMU and jointState
  Eigen::Vector3d opti_euler;

  vJointPos joint_pos;
  vJointVel joint_vel;
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

 protected:
  mutable std::mutex data_lock;
};

class A1SensorData : public A1SensorDataInterface {
 public:
  A1SensorData() {
    for (size_t i = 0; i < 3; ++i) {
      acc_filter[i] = MovingWindowFilter(30);
      ang_vel_filter[i] = MovingWindowFilter(15);
    }
  }

  /* IMU and joint data */
  void input_imu(const vAcc& acc, const vAngVel& ang_vel, ImuIndex imu_index = BODY_IMU) override {
    for (size_t i = 0; i < 3; ++i) {
      this->acc[i] = acc_filter[i].CalculateAverage(acc[i]);
      this->ang_vel[i] = ang_vel_filter[i].CalculateAverage(ang_vel[i]);
    }
  }

  // data in IMU and jointState
  Eigen::Vector3d acc;
  Eigen::Vector3d ang_vel;

 private:
  /* filters for IMU/joint data */
  MovingWindowFilter acc_filter[3];
  MovingWindowFilter ang_vel_filter[3];
};

class A1FootIMUSensorData : public A1SensorDataInterface {
 public:
  A1FootIMUSensorData() {
    for (size_t i = 0; i < IMU_MAX_SIZE; ++i) {
      for (size_t j = 0; i < 3; ++i) {
        acc_filter[i][j] = MovingWindowFilter(30);
        ang_vel_filter[i][j] = MovingWindowFilter(15);
      }
    }
  }

  /* IMU and joint data */
  void input_imu(const vAcc& acc, const vAngVel& ang_vel, ImuIndex imu_index) override {
    assert(imu_index >= 0 && imu_index < IMU_MAX_SIZE);

    for (size_t i = 0; i < 3; ++i) {
      this->acc[imu_index][i] = acc_filter[imu_index][i].CalculateAverage(acc[i]);
      this->ang_vel[imu_index][i] = ang_vel_filter[imu_index][i].CalculateAverage(ang_vel[i]);
    }
  }

  const Eigen::Vector3d& getAcc(ImuIndex imu_index) const { return acc[imu_index]; }
  const Eigen::Vector3d& getAngVel(ImuIndex imu_index) const { return ang_vel[imu_index]; }

  // Data in IMU and jointState
  Eigen::Vector3d acc[IMU_MAX_SIZE];
  Eigen::Vector3d ang_vel[IMU_MAX_SIZE];

 private:
  MovingWindowFilter acc_filter[IMU_MAX_SIZE][3];
  MovingWindowFilter ang_vel_filter[IMU_MAX_SIZE][3];
};

class A1KF {
 public:
  A1KF(){};
  bool is_initialized() { return KF_initialized; }

  bool KF_initialized = false;
};