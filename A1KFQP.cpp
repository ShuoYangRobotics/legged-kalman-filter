#include "A1KFQP.h"

// default constructor
A1KFQP::A1KFQP(): A1KF() {
    load_casadi_functions();
    curr_state = Eigen::Matrix<double, EKF_STATE_SIZE, 1>::Zero();
    curr_covariance = Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity()*0.01;
    KF_initialized = false;
    eye3.setIdentity();

    // kinematics related, copy from GazeboA1ROS.cpp    
    // leg order: 0-FL  1-FR  2-RL  3-RR
    leg_offset_x[0] = 0.1805;
    leg_offset_x[1] = 0.1805;
    leg_offset_x[2] = -0.1805;
    leg_offset_x[3] = -0.1805;
    leg_offset_y[0] = 0.047;
    leg_offset_y[1] = -0.047;
    leg_offset_y[2] = 0.047;
    leg_offset_y[3] = -0.047;
    motor_offset[0] = 0.0838;
    motor_offset[1] = -0.0838;
    motor_offset[2] = 0.0838;
    motor_offset[3] = -0.0838;
    upper_leg_length[0] = upper_leg_length[1] = upper_leg_length[2] = upper_leg_length[3] = 0.20;
    lower_leg_length[0] = lower_leg_length[1] = lower_leg_length[2] = lower_leg_length[3] = 0.20;

    for (int i = 0; i < NUM_LEG; i++) {
        Eigen::VectorXd rho_fix(5);
        rho_fix << leg_offset_x[i], leg_offset_y[i], motor_offset[i], upper_leg_length[i], lower_leg_length[i];
        Eigen::VectorXd rho_opt(3);
        rho_opt << 0.0, 0.0, 0.0;
        rho_fix_list.push_back(rho_fix);
        rho_opt_list.push_back(rho_opt);
    }

    // setup QP solver       
    hessian_.resize(EKF_STATE_SIZE, EKF_STATE_SIZE); hessian_.setZero(); 
    gradient_.resize(EKF_STATE_SIZE); gradient_.setZero(); 

    lowerBound_.resize(EKF_STATE_SIZE); 
    upperBound_.resize(EKF_STATE_SIZE); 

    for (int i = 0; i < EKF_STATE_SIZE; i++) {
        lowerBound_[i] = -OsqpEigen::INFTY; 
        upperBound_[i] =  OsqpEigen::INFTY; 
    }


}


void A1KFQP::init_filter(A1SensorData& data, Eigen::Vector3d _init_pos) {
    Eigen::Vector3d init_pos = _init_pos;
    curr_state = Eigen::Matrix<double, EKF_STATE_SIZE, 1>::Zero();
    curr_state.segment<3>(0) = init_pos;

    for (int i = 0; i < NUM_LEG; i++) {
        Eigen::Vector3d init_foot_pos = kinematics.fk(data.joint_pos.segment<3>(i*3), rho_opt_list[i], rho_fix_list[i]);

        //identity body rotation
        curr_state.segment<3>(9+i*3) = init_foot_pos+init_pos;
    }

    curr_covariance = Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity()*inital_cov;
    // large initial position uncertainty
    // curr_covariance.block<3,3>(0,0) = Eigen::Matrix3d::Identity()* 5.0;

    KF_initialized = true;

    prev_ctrl << data.ang_vel, data.acc, data.dt;

    // initialize noise matrices
    process_noise = Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity()*0.01;
    process_noise.diagonal().segment<3>(0) = 0.1*Eigen::Vector3d::Ones();
    process_noise.diagonal().segment<3>(3) = 0.05*Eigen::Vector3d::Ones();
    process_noise.diagonal().segment<3>(6) = 1e-6*Eigen::Vector3d::Ones();
    process_noise.diagonal().segment<12>(9) = 1e-3*Eigen::Matrix<double,12,1>::Ones();
    process_noise.diagonal()[EKF_STATE_SIZE-1] = 0; // the time is exact

    // initialize measurement noise
    measure_noise = Eigen::Matrix<double, OBSERVATION_SIZE, OBSERVATION_SIZE>::Identity()*0.01;


    // opti track related 
    opti_jacobian.setZero();
    opti_jacobian.block<6,6>(0,0) = Eigen::Matrix<double,6,6>::Identity();
    opti_jacobian(6,8) = 1.0; 
    opti_noise.setZero();
    opti_noise.block<3,3>(0,0) = Eigen::Matrix<double,3,3>::Identity()*noise_opti_pos; //opti_pos
    opti_noise.block<3,3>(3,3) = Eigen::Matrix<double,3,3>::Identity()*noise_opti_vel; // opti_vel
    opti_noise.block<1,1>(6,6) = Eigen::Matrix<double,1,1>::Identity()*noise_opti_yaw; // opti yaw 
}


void A1KFQP::update_filter(A1SensorData& data) {
    // const std::lock_guard<std::mutex> lock(update_mutex);

    update_mutex.lock();
    // filter initialized, now curr_ctrl and prev_ctrl are ready
    // update the state
    curr_ctrl << data.ang_vel, data.acc, data.dt;
    // process updates x01 and calculates process_jacobian
    process(curr_state, prev_ctrl, curr_ctrl, data.dt);


    process_noise.diagonal().segment<2>(0) = noise_process_pos_xy * data.dt/20.0*Eigen::Vector2d::Ones();           // pos x y
    process_noise.diagonal()(2) = noise_process_pos_z * data.dt / 20.0;                                             // pos z
    process_noise.diagonal().segment<2>(3) = noise_process_vel_xy * data.dt * 9.8 / 20.0*Eigen::Vector2d::Ones();  // vel x y
    process_noise.diagonal()(5) = noise_process_vel_z * data.dt * 9.8 / 20.0;                                       // vel z
    process_noise.diagonal().segment<3>(6) = noise_process_rot*Eigen::Vector3d::Ones();

    // adjust noise according to contact 
    for (int i = 0; i < NUM_LEG; ++i) {
        process_noise.block<3, 3>(9 + i * 3, 9 + i * 3)
                =
                (1 + (1 - data.plan_contacts[i]) * 1e5) * noise_process_foot * data.dt * eye3;  // foot position transition

        measure_noise.block<3, 3>(i * OBS_PER_LEG, i * OBS_PER_LEG)
                =  noise_measure_fk * eye3;     // fk estimation

        measure_noise(i * OBS_PER_LEG + 3, i * OBS_PER_LEG + 3)
                = (1 + (1 - data.plan_contacts[i]) * 1e5) * noise_measure_vel;      // vel estimation
        measure_noise(i * OBS_PER_LEG + 4, i * OBS_PER_LEG + 4)
                = (1 + (1 - data.plan_contacts[i]) * 1e5) * noise_measure_vel;      // vel estimation
        measure_noise(i * OBS_PER_LEG + 5, i * OBS_PER_LEG + 5)
                = (1 + (1 - data.plan_contacts[i]) * 1e5) * noise_measure_vel;      // vel estimation

        measure_noise(i * OBS_PER_LEG + 6, i * OBS_PER_LEG + 6)
                = (1 + (1 - data.plan_contacts[i]) * 1e5) * noise_measure_height;      // height
    }

    P01 = process_jacobian*curr_covariance*process_jacobian.transpose() + process_noise;

    // construct QP for measurement updates
    measure(x01, data.ang_vel, data.joint_pos, data.joint_vel);

    hessian_ = P01.inverse() + measurement_jacobian.transpose()*measure_noise.inverse()*measurement_jacobian;
    gradient_ = 2*measurement.transpose()*measure_noise.inverse()*measurement_jacobian;

    sparse_hessian_ = hessian_.sparseView();    
    
    OsqpEigen::Solver solver;
    solver.settings()->setVerbosity(false);
    solver.settings()->setAbsoluteTolerance(1e-6);
    solver.settings()->setRelativeTolerance(1e-6);
    solver.settings()->setWarmStart(true);

    solver.data()->setNumberOfVariables(EKF_STATE_SIZE);
    solver.data()->setNumberOfConstraints(0);
    sparse_hessian_ = hessian_.sparseView();
    solver.data()->setHessianMatrix(sparse_hessian_);
    solver.data()->setGradient(gradient_);
    
    solver.data()->setLowerBound(lowerBound_);
    solver.data()->setUpperBound(upperBound_);
    solver.initSolver();

    solver.solveProblem();

    Eigen::VectorXd solution = solver.getSolution(); //12x1
    curr_state = x01 + solution;

    S = measurement_jacobian*P01*measurement_jacobian.transpose() + measure_noise;
    // Eigen::VectorXd invSy = S.fullPivHouseholderQr().solve(measurement);
    Eigen::MatrixXd  invSH = S.fullPivHouseholderQr().solve(measurement_jacobian);

    // Eigen::Matrix<double, EKF_STATE_SIZE,1> update =  P01*measurement_jacobian.transpose()*invSy;
    // curr_state = x01 - update;

    curr_covariance = (Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity() - P01*measurement_jacobian.transpose()*invSH)*P01;
    curr_covariance = (curr_covariance + curr_covariance.transpose()) / 2;

    // // measurement updates
    // measure(x01, data.ang_vel, data.joint_pos, data.joint_vel);

    // S = measurement_jacobian*P01*measurement_jacobian.transpose() + measure_noise;

    // Eigen::VectorXd invSy = S.fullPivHouseholderQr().solve(measurement);
    // double mahalanobis_distance = measurement.transpose()*invSy;
    // if ( mahalanobis_distance< 100) {
    //     Eigen::Matrix<double, EKF_STATE_SIZE,1> update =  P01*measurement_jacobian.transpose()*invSy;

    //     curr_state = x01 - update;

    //     Eigen::MatrixXd  invSH = S.fullPivHouseholderQr().solve(measurement_jacobian);

    //     curr_covariance = (Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity() - P01*measurement_jacobian.transpose()*invSH)*P01;

    //     curr_covariance = (curr_covariance + curr_covariance.transpose()) / 2;
    // } else {
    //     curr_state = x01;
    //     curr_covariance = P01;
    // }




    // finally save previous control 
    prev_ctrl = curr_ctrl;
    update_mutex.unlock();
    return;
}


void A1KFQP::set_noise_params(double _inital_cov,
                        double _noise_process_pos_xy,
                        double _noise_process_pos_z,
                        double _noise_process_vel_xy,
                        double _noise_process_vel_z,
                        double _noise_process_rot,
                        double _noise_process_foot,
                        double _noise_measure_fk,
                        double _noise_measure_vel,
                        double _noise_measure_height,
                        double _noise_opti_pos,
                        double _noise_opti_vel,
                        double _noise_opti_yaw) {

    inital_cov = _inital_cov;
    noise_process_pos_xy = _noise_process_pos_xy;
    noise_process_pos_z = _noise_process_pos_z;
    noise_process_vel_xy = _noise_process_vel_xy;
    noise_process_vel_z = _noise_process_vel_z;
    noise_process_rot = _noise_process_rot;
    noise_process_foot = _noise_process_foot;

    noise_measure_fk = _noise_measure_fk;
    noise_measure_vel = _noise_measure_vel;
    noise_measure_height = _noise_measure_height;

    noise_opti_pos = _noise_opti_pos;
    noise_opti_vel = _noise_opti_vel;
    noise_opti_yaw = _noise_opti_yaw;          

    return;              
}

// private 
void A1KFQP::load_casadi_functions() {
    // load casadi functions
    process_func = casadi::external("process", "/tmp/casadi_kf_baseline4_process.so");
    process_jac_func = casadi::external("process_jac", "/tmp/casadi_kf_baseline4_process_jac.so");
    measure_func = casadi::external("meas", "/tmp/casadi_kf_baseline4_meas.so");
    measure_jac_func = casadi::external("meas_jac", "/tmp/casadi_kf_baseline4_meas_jac.so");
}

void A1KFQP::process(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state, 
                                                     Eigen::Matrix<double, CONTROL_SIZE, 1> prev_ctrl, 
                                                     Eigen::Matrix<double, CONTROL_SIZE, 1> ctrl, double dt) {
    // const std::lock_guard<std::mutex> lock(update_mutex);
    std::vector<double> xk_vec;
    xk_vec.resize(state.size());
    Eigen::Matrix<double, EKF_STATE_SIZE, 1>::Map(&xk_vec[0], state.size()) = state;

    std::vector<double> uk_vec;
    uk_vec.resize(prev_ctrl.size());
    Eigen::Matrix<double, CONTROL_SIZE, 1>::Map(&uk_vec[0], prev_ctrl.size()) = prev_ctrl;

    std::vector<double> uk1_vec;
    uk1_vec.resize(ctrl.size());
    Eigen::Matrix<double, CONTROL_SIZE, 1>::Map(&uk1_vec[0], ctrl.size()) = ctrl;

    std::vector<double> dt_vec;
    dt_vec.resize(1);
    dt_vec[0] = dt;

    // assemble input into arg vector
    std::vector<casadi::DM> arg = {casadi::DM(xk_vec), 
         casadi::DM(uk_vec), 
         casadi::DM(uk1_vec), 
         casadi::DM(dt_vec)};

    // process update
    std::vector<casadi::DM> res = process_func(arg);
    std::vector<double> res_vec = std::vector<double>(res.at(0));
    x01 = Eigen::Matrix<double, EKF_STATE_SIZE, 1>::Map(res_vec.data(), res_vec.size());

    res = process_jac_func(arg);
    res_vec = std::vector<double>(res.at(0));
    process_jacobian = Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>(res_vec.data());

    return;
}


void A1KFQP::measure(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state, 
                                                     Eigen::Matrix<double, 3, 1> w, 
                                                     Eigen::Matrix<double, 12, 1> joint_ang, 
                                                     Eigen::Matrix<double, 12, 1> joint_vel) {
    // const std::lock_guard<std::mutex> lock(update_mutex);

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


    // assemble input into arg vector
    std::vector<casadi::DM> arg = {casadi::DM(xk_vec),
            casadi::DM(w_vec),
            casadi::DM(joint_ang_vec),
            casadi::DM(joint_vel_vec)};
    
    // measurement update
    std::vector<casadi::DM> res = measure_func(arg);
    std::vector<double> res_vec = std::vector<double>(res.at(0));
    measurement = Eigen::Matrix<double, OBSERVATION_SIZE, 1>(res_vec.data());

    res = measure_jac_func(arg);
    res_vec = std::vector<double>(res.at(0));
    measurement_jacobian = Eigen::Matrix<double, OBSERVATION_SIZE, EKF_STATE_SIZE>(res_vec.data());

    return;
}