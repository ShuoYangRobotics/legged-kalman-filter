#include "A1KFFootIMU.h"

namespace {
void update_process_noise(Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>& process_noise, double dt) {
  //
  // [0.0001*dt/20.0*ones(2,1); % pos x y
  //   0.0001*dt/20.0*ones(1,1);   % pos z
  //   0.001 * dt * 9.8 / 20.0*ones(1,1); % vel x
  //   0.001 * dt * 9.8 / 20.0*ones(1,1); % vel y
  //   0.0001 * dt * 9.8 / 20.0*ones(1,1);  % vel z
  //   1e-6*ones(3,1); // euler
  //  repmat(...
  //  [0.1 * dt * 9.8 / 2.0*ones(2,1);  % foot1 pos x y
  //   0.1 * dt * 9.8 / 2.0*ones(1,1);  % foot1 pos z
  //   0.4 * dt * 9.8 / 2.0*ones(2,1);  % foot1 vel  x y
  //   0.4 * dt * 9.8

  auto diagNoise = process_noise.diagonal();
  getBodyPosition(diagNoise).head(2).setConstant(0.0001 * dt / 20.0);  // pos x y
  getBodyPosition(diagNoise).tail(1).setConstant(0.0001 * dt / 20.0);  // pos z

  getBodyVelocity(diagNoise).head(2).setConstant(0.001 * dt * 9.8 / 20.0);   // vel x y
  getBodyVelocity(diagNoise).tail(1).setConstant(0.0001 * dt * 9.8 / 20.0);  // vel z

  getBodyEulerAngle(diagNoise).setConstant(1e-6);

  for (int ind = A1FootIMUSensorData::FOOT_FL_IMU; ind <= A1FootIMUSensorData::FOOT_RR_IMU; ++ind) {
    getFootPosition(diagNoise, ind).head(2).setConstant(0.1 * dt * 9.8 / 2.0);  // foot pos x y
    getFootPosition(diagNoise, ind).tail(1).setConstant(0.1 * dt * 9.8 / 2.0);  // foot pos z

    getFootVelocity(diagNoise, ind).head(2).setConstant(0.4 * dt * 9.8 / 2.0);  // foot vel x y
    getFootVelocity(diagNoise, ind).tail(1).setConstant(0.4 * dt * 9.8 / 2.0);  // foot vel z
  }
}

void update_measure_noise(Eigen::Matrix<double, OBSERVATION_SIZE, OBSERVATION_SIZE>& measure_noise, double dt) {
  // TODO: Implement this. Maybe, change function signature to include contact flags
}
}  // namespace

// default constructor
A1KFFootIMU::A1KFFootIMU() : A1KF() {
  load_casadi_functions();

  reset();

  KF_initialized = false;
}

void A1KFFootIMU::reset() {
  curr_state.setZero();
  prev_ctrl.setZero();
  curr_ctrl.setZero();

  curr_covariance.setIdentity();
  curr_covariance *= 0.001;

  process_noise.setIdentity();
  measure_noise.setIdentity();
}

void A1KFFootIMU::init_filter(A1FootIMUSensorData& data, const Eigen::Vector3d& _init_pos) {
  reset();

  // TODO: Initial state ?? How to initial foot pos & vel ?? No fk lib
  getBodyPosition(curr_state) = _init_pos;

  KF_initialized = true;

  // TODO: Is it correct?
  getBodyAngularVelocity(prev_ctrl) = data.ang_vel[A1FootIMUSensorData::BODY_IMU];
  getBodyAcceleration(prev_ctrl) = data.acc[A1FootIMUSensorData::BODY_IMU];
  for (int ind = A1FootIMUSensorData::FOOT_FL_IMU; ind <= A1FootIMUSensorData::FOOT_RR_IMU; ++ind) {
    getFootAcceleration(prev_ctrl, ind) = data.acc[ind];
  }

  // TODO: Move below to reset
  // opti track related
  opti_jacobian.setZero();
  opti_jacobian.block<6, 6>(0, 0) = Eigen::Matrix<double, 6, 6>::Identity();
  opti_noise.setZero();
  opti_noise.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * 0.02;  // opti_pos
  opti_noise.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity() * 0.02;  // opti_vel
}

void A1KFFootIMU::update_filter(A1FootIMUSensorData& data) {
  std::lock_guard<std::mutex> lock(update_mutex);

  getBodyAngularVelocity(curr_ctrl) = data.ang_vel[A1FootIMUSensorData::BODY_IMU];
  getBodyAcceleration(curr_ctrl) = data.acc[A1FootIMUSensorData::BODY_IMU];
  for (int ind = A1FootIMUSensorData::FOOT_FL_IMU; ind <= A1FootIMUSensorData::FOOT_RR_IMU; ++ind) {
    getFootAcceleration(curr_ctrl, ind) = data.acc[ind];
  }

  // process updates x01 and calculates process_jacobian
  process(curr_state, prev_ctrl, curr_ctrl, data.dt);
  update_process_noise(process_noise, data.dt);

  P01 = process_jacobian * curr_covariance * process_jacobian.transpose() + process_noise;

  double sum_contact = data.plan_contacts.sum();
  if (sum_contact < 2) {
    // no contact, update the covariance directly without using measurements
    curr_state = x01;
    curr_covariance = P01;
  } else {
    // contact, update the state and covariance using measurements
    // measure calculates measurement residual and measurement_jacobian
    measure(x01, data.ang_vel[A1FootIMUSensorData::BODY_IMU], data.joint_pos, data.joint_vel, data.plan_contacts);
    update_measure_noise(measure_noise, data.dt);

    Eigen::Matrix<double, OBSERVATION_SIZE, OBSERVATION_SIZE> S =
        measurement_jacobian * P01 * measurement_jacobian.transpose() + measure_noise;

    VMeasure invSy = S.fullPivHouseholderQr().solve(measurement);

    // outlier rejection
    double mahalanobis_distance = measurement.transpose() * invSy;
    if (mahalanobis_distance < 0.1) {
      // update the state and covariance
      vState Ky = P01 * measurement_jacobian.transpose() * invSy;
      curr_state = x01 - Ky;
      Eigen::Matrix<double, OBSERVATION_SIZE, EKF_STATE_SIZE> invSH = S.fullPivHouseholderQr().solve(measurement_jacobian);

      curr_covariance =
          (Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity() - P01 * measurement_jacobian.transpose() * invSH) * P01;

      curr_covariance = (curr_covariance + curr_covariance.transpose()) / 2;
    } else {
      // update the state and covariance directly without using measurements
      curr_state = x01;
      curr_covariance = P01;
    }
  }

  // finally save previous control
  prev_ctrl = curr_ctrl;
  return;
}

// update state using opti track data
void A1KFFootIMU::update_filter_with_opti(A1FootIMUSensorData& data) {
  const std::lock_guard<std::mutex> lock(update_mutex);

  // actual measurement
  Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, 1> opti_meas;
  opti_meas << data.opti_pos, data.opti_vel;
  Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, 1> opti_residual = opti_meas - opti_jacobian * curr_state;

  //
  Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, OPTI_OBSERVATION_SIZE> S =
      opti_jacobian * curr_covariance * opti_jacobian.transpose() + opti_noise;

  Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, 1> invSy = S.fullPivHouseholderQr().solve(opti_residual);

  // // outlier rejection
  // double mahalanobis_distance = opti_residual.transpose()*invSy;
  // if (mahalanobis_distance < 1) {
  Eigen::Matrix<double, EKF_STATE_SIZE, 1> Ky = curr_covariance * opti_jacobian.transpose() * invSy;
  curr_state += Ky;
  Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, EKF_STATE_SIZE> invSH = S.fullPivHouseholderQr().solve(opti_jacobian);

  curr_covariance =
      (Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity() - curr_covariance * opti_jacobian.transpose() * invSH) *
      curr_covariance;
  // }
}

// private

void A1KFFootIMU::load_casadi_functions() {
  // load casadi functions
  process_func = casadi::external("process", "/tmp/casadi_kf_baseline1_process.so");
  process_jac_func = casadi::external("process_jac", "/tmp/casadi_kf_baseline1_process_jac.so");
  measure_func = casadi::external("meas", "/tmp/casadi_kf_baseline1_meas.so");
  measure_jac_func = casadi::external("meas_jac", "/tmp/casadi_kf_baseline1_meas_jac.so");
}

void A1KFFootIMU::process(const vState& state, const vControl& prev_ctrl, const vControl& ctrl, double dt) {
  std::vector<double> xk_vec;
  xk_vec.resize(state.size());
  Eigen::Matrix<double, EKF_STATE_SIZE, 1>::Map(&xk_vec[0], state.size()) = state;

  std::vector<double> uk_vec;
  uk_vec.resize(prev_ctrl.size());
  Eigen::Matrix<double, CONTROL_SIZE, 1>::Map(&uk_vec[0], prev_ctrl.size()) = prev_ctrl;

  std::vector<double> uk1_vec;
  uk1_vec.resize(ctrl.size());
  Eigen::Matrix<double, CONTROL_SIZE, 1>::Map(&uk1_vec[0], ctrl.size()) = ctrl;

  // assemble input into arg vector
  std::vector<casadi::DM> arg = {casadi::DM(xk_vec), casadi::DM(uk_vec), casadi::DM(uk1_vec)};

  // process update
  std::vector<casadi::DM> res = process_func(arg);
  std::vector<double> res_vec = std::vector<double>(res.at(0));
  x01 = Eigen::Matrix<double, EKF_STATE_SIZE, 1>::Map(res_vec.data(), res_vec.size());

  res = process_jac_func(arg);
  res_vec = std::vector<double>(res.at(0));
  process_jacobian = Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>(res_vec.data());

  return;
}

void A1KFFootIMU::measure(const vState& state, const Eigen::Matrix<double, 3, 1>& w, const Eigen::Matrix<double, 12, 1>& joint_ang,
                          const Eigen::Matrix<double, 12, 1>& joint_vel, const Eigen::Matrix<double, 4, 1>& contact) {
  std::vector<double> xk_vec;
  xk_vec.resize(state.size());
  Eigen::Matrix<double, EKF_STATE_SIZE, 1>::Map(&xk_vec[0], state.size()) = state;
  std::vector<double> w_vec;
  w_vec.resize(w.size());
  Eigen::Matrix<double, 3, 1>::Map(&w_vec[0], w.size()) = w;

  std::vector<double> joint_ang_vec;
  joint_ang_vec.resize(joint_ang.size());
  Eigen::Matrix<double, 12, 1>::Map(&joint_ang_vec[0], joint_ang.size()) = joint_ang;

  std::vector<double> joint_vel_vec;
  joint_vel_vec.resize(joint_vel.size());
  Eigen::Matrix<double, 12, 1>::Map(&joint_vel_vec[0], joint_vel.size()) = joint_vel;

  std::vector<double> contact_vec;
  contact_vec.resize(contact.size());
  Eigen::Matrix<double, 4, 1>::Map(&contact_vec[0], contact.size()) = contact;

  // assemble input into arg vector
  std::vector<casadi::DM> arg = {casadi::DM(xk_vec), casadi::DM(w_vec), casadi::DM(joint_ang_vec), casadi::DM(joint_vel_vec),
                                 casadi::DM(contact_vec)};

  // measurement update
  std::vector<casadi::DM> res = measure_func(arg);
  std::vector<double> res_vec = std::vector<double>(res.at(0));
  measurement = Eigen::Matrix<double, OBSERVATION_SIZE, 1>(res_vec.data());

  res = measure_jac_func(arg);
  res_vec = std::vector<double>(res.at(0));
  measurement_jacobian = Eigen::Matrix<double, OBSERVATION_SIZE, EKF_STATE_SIZE>(res_vec.data());

  return;
}