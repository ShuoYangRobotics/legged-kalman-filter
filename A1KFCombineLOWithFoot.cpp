#include "A1KFCombineLOWithFoot.h"

// default constructor
A1KFCombineLOWithFoot::A1KFCombineLOWithFoot(): A1KF() {
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
}


void A1KFCombineLOWithFoot::init_filter(A1SensorData& data, Eigen::Vector3d _init_pos) {
    Eigen::Vector3d init_pos = _init_pos;
    curr_state = Eigen::Matrix<double, EKF_STATE_SIZE, 1>::Zero();
    curr_state.segment<3>(0) = init_pos;

    for (int i = 0; i < NUM_LEG; i++) {
        Eigen::Vector3d init_foot_pos = kinematics.fk(data.joint_pos.segment<3>(i*3), rho_opt_list[i], rho_fix_list[i]);

        //identity body rotation
        curr_state.segment<3>(9+i*3) = init_foot_pos+init_pos;
    }

    curr_covariance = Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity()*0.1;
    // large initial position uncertainty
    // curr_covariance.block<3,3>(0,0) = Eigen::Matrix3d::Identity()* 5.0;

    KF_initialized = true;

    prev_ctrl << data.ang_vel, data.acc, data.dt;

    // initialize noise matrices
    process_noise = Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity()*0.01;
    process_noise.diagonal().segment<3>(0) = 0.1*Eigen::Vector3d::Ones();
    process_noise.diagonal().segment<3>(3) = 0.05*Eigen::Vector3d::Ones();
    process_noise.diagonal().segment<3>(6) = 1e-6*Eigen::Vector3d::Ones();
    process_noise.diagonal().segment<12>(9) = 1e-2*Eigen::Matrix<double,12,1>::Ones();
    process_noise.diagonal()[EKF_STATE_SIZE-1] = 0; // the time is exact

    // initialize measurement noise
    measure_noise = Eigen::Matrix<double, OBSERVATION_SIZE, OBSERVATION_SIZE>::Identity()*0.1;


    // opti track related 
    opti_jacobian.setZero();
    opti_jacobian.block<6,6>(0,0) = Eigen::Matrix<double,6,6>::Identity();
    opti_jacobian(6,8) = 1.0; 
    opti_noise.setZero();
    opti_noise.block<3,3>(0,0) = Eigen::Matrix<double,3,3>::Identity()*0.001; //opti_pos
    opti_noise.block<3,3>(3,3) = Eigen::Matrix<double,3,3>::Identity()*0.05; // opti_vel
    opti_noise.block<1,1>(6,6) = Eigen::Matrix<double,1,1>::Identity()*0.01; // opti yaw 
}


void A1KFCombineLOWithFoot::update_filter(A1SensorData& data) {
    // const std::lock_guard<std::mutex> lock(update_mutex);

    update_mutex.lock();
    // filter initialized, now curr_ctrl and prev_ctrl are ready
    // update the state
    curr_ctrl << data.ang_vel, data.acc, data.dt;
    // process updates x01 and calculates process_jacobian
    process(curr_state, prev_ctrl, curr_ctrl, data.dt);


    process_noise.diagonal().segment<2>(0) = 0.001*data.dt/20.0*Eigen::Vector2d::Ones();           // pos x y
    process_noise.diagonal()(2) = 0.001* data.dt / 20.0;                                             // pos z
    process_noise.diagonal().segment<2>(3) = 0.001 * data.dt * 9.8 / 20.0*Eigen::Vector2d::Ones();  // vel x y
    process_noise.diagonal()(5) = 0.001 * data.dt * 9.8 / 20.0;                                       // vel z
    process_noise.diagonal().segment<3>(6) = 1e-6*Eigen::Vector3d::Ones();

    // adjust noise according to contact 
    for (int i = 0; i < NUM_LEG; ++i) {
        process_noise.block<3, 3>(9 + i * 3, 9 + i * 3)
                =
                (1 + (1 - data.plan_contacts[i]) * 1e5) * 0.0001 * data.dt * eye3;  // foot position transition

        measure_noise.block<3, 3>(i * 3, i * 3)
                =  0.01 * eye3;     // fk estimation

        measure_noise(i * 6 + 3, i * 6 + 3)
                = (1 + (1 - data.plan_contacts[i]) * 1e5) * 2 ;      // vel estimation
        measure_noise(i * 6 + 4, i * 6 + 4)
                = (1 + (1 - data.plan_contacts[i]) * 1e5) * 2 ;      // vel estimation
        measure_noise(i * 6 + 5, i * 6 + 5)
                = (1 + (1 - data.plan_contacts[i]) * 1e5) * 0.1 ;      // vel estimation
    }

    P01 = process_jacobian*curr_covariance*process_jacobian.transpose() + process_noise;

    // measurement updates
    measure(x01, data.ang_vel, data.joint_pos, data.joint_vel);

    S = measurement_jacobian*P01*measurement_jacobian.transpose() + measure_noise;

    // outliner rejection
    bool pos_mask[NUM_LEG]; 
    bool vel_mask[NUM_LEG]; 
    int total_pos = 0, total_vel  = 0;
    double mahalanobis_distance = 0;
    for (int i = 0; i < NUM_LEG; ++i) {
        // position
        Eigen::Matrix3d subS = S.block<3,3>(i*6,i*6);
        Eigen::Vector3d suby = measurement.segment<3>(i*6);
        Eigen::Vector3d invSy = subS.fullPivHouseholderQr().solve(suby);
        mahalanobis_distance = suby.transpose()*invSy;
        if (mahalanobis_distance < 0.5) {
            pos_mask[i] = true;
            total_pos++;
        } else {
            pos_mask[i] = false;
        }

        // velocity

        // modify measurement noise according to contact
        // Eigen::Vector3d vel_meas = measurement.segment<3>(i*6+3);
        // measurement.segment<3>(i*6+3) = (1 - data.plan_contacts[i]) * Eigen::Vector3d::Zero() + data.plan_contacts[i] * vel_meas;

        subS = S.block<3,3>(i*6+3,i*6+3);
        suby = measurement.segment<3>(i*6+3);
        invSy = subS.fullPivHouseholderQr().solve(suby);
        mahalanobis_distance = suby.transpose()*invSy;
        if (mahalanobis_distance < 0.1) {
            vel_mask[i] = true;
            total_vel++;
            estimated_contact[i] = 1.0;
        } else {
            vel_mask[i] = false;
            estimated_contact[i] = 0.0;
        }
    }

    if (total_pos+total_vel >= 1) {
        // have to use MatrixXd and VectorXd to store the masked measurement
        Eigen::VectorXd masked_measurement(total_pos*3+total_vel*3);
        masked_measurement.setZero();
        // Eigen::MatrixXd masked_S(total_pos*3+total_vel*3, total_pos*3+total_vel*3);
        // masked_S.setZero();
        Eigen::MatrixXd masked_jacobian(total_pos*3+total_vel*3, EKF_STATE_SIZE);
        Eigen::MatrixXd masked_measure_noise(total_pos*3+total_vel*3, total_pos*3+total_vel*3);
        masked_measure_noise.setZero();
        masked_jacobian.setZero();
        int idx_pos = 0, idx_vel  = 0;
        for (int i = 0; i < NUM_LEG; ++i) {
            if (pos_mask[i] == true) {
                masked_measurement.segment<3>(idx_pos*3) = measurement.segment<3>(i*6);
                // masked_S.block<3,3>(idx_pos*3, idx_pos*3) = S.block<3,3>(i*6, i*6);
                masked_measure_noise.block<3,3>(idx_pos*3, idx_pos*3) = measure_noise.block<3,3>(i*6, i*6);
                masked_jacobian.block<3,EKF_STATE_SIZE>(idx_pos*3, 0) = measurement_jacobian.block<3,EKF_STATE_SIZE>(i*6, 0);
                idx_pos++;
            }

            if (vel_mask[i] == true) {
                masked_measurement.segment<3>(total_pos*3+idx_vel*3) = measurement.segment<3>(i*6+3);
                // masked_S.block<3,3>(total_pos*3+idx_vel*3, total_pos*3+idx_vel*3) = S.block<3,3>(i*6+3, i*6+3);
                masked_measure_noise.block<3,3>(total_pos*3+idx_vel*3, total_pos*3+idx_vel*3) = measure_noise.block<3,3>(i*6+3, i*6+3);
                masked_jacobian.block<3,EKF_STATE_SIZE>(total_pos*3+idx_vel*3, 0) = measurement_jacobian.block<3,EKF_STATE_SIZE>(i*6+3, 0);
                idx_vel++;
            }
        }
        Eigen::MatrixXd masked_S = masked_jacobian*P01*masked_jacobian.transpose() + masked_measure_noise;
        Eigen::VectorXd masked_invSy = masked_S.fullPivHouseholderQr().solve(masked_measurement);

        Eigen::Matrix<double, EKF_STATE_SIZE,1> update =  P01*masked_jacobian.transpose()*masked_invSy;
        curr_state = x01 - update;

        Eigen::MatrixXd  invSH = masked_S.fullPivHouseholderQr().solve(masked_jacobian);

        curr_covariance = (Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity() - P01*masked_jacobian.transpose()*invSH)*P01;

        curr_covariance = (curr_covariance + curr_covariance.transpose()) / 2;
    } else {
        //update the state and covariance directly without using measurements
        curr_state = x01;
        curr_covariance = P01;
    }

    
    // Eigen::VectorXd invSy = S.fullPivHouseholderQr().solve(measurement);

    // Eigen::Matrix<double, EKF_STATE_SIZE,1> update =  P01*measurement_jacobian.transpose()*invSy;

    // curr_state = x01 - update;

    // Eigen::MatrixXd  invSH = S.fullPivHouseholderQr().solve(measurement_jacobian);

    // curr_covariance = (Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity() - P01*measurement_jacobian.transpose()*invSH)*P01;

    // curr_covariance = (curr_covariance + curr_covariance.transpose()) / 2;
        // curr_state = x01;
        // curr_covariance = P01;
    // finally save previous control 
    prev_ctrl = curr_ctrl;
    update_mutex.unlock();
    return;
}



// update state using opti track data
void A1KFCombineLOWithFoot::update_filter_with_opti(A1SensorData& data) {
    // const std::lock_guard<std::mutex> lock(update_mutex);
    update_mutex.lock();

    // actual measurement
    Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, 1> opti_meas;
    opti_meas << data.opti_pos, data.opti_vel, data.opti_euler(2);
    Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, 1> opti_residual = opti_meas - opti_jacobian*curr_state;
    // 
    Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, OPTI_OBSERVATION_SIZE> S = opti_jacobian*curr_covariance*opti_jacobian.transpose() + opti_noise;

    Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, 1> invSy = S.fullPivHouseholderQr().solve(opti_residual);
    
    // outlier rejection
    double mahalanobis_distance = opti_residual.transpose()*invSy;
    if (mahalanobis_distance < 10.0) {
        Eigen::Matrix<double, EKF_STATE_SIZE, 1> Ky = curr_covariance*opti_jacobian.transpose()*invSy;
        curr_state += Ky;      
        Eigen::Matrix<double, OPTI_OBSERVATION_SIZE, EKF_STATE_SIZE>  invSH = S.fullPivHouseholderQr().solve(opti_jacobian);

        curr_covariance = (Eigen::Matrix<double, EKF_STATE_SIZE, EKF_STATE_SIZE>::Identity() - curr_covariance*opti_jacobian.transpose()*invSH)*curr_covariance;  
    }
    update_mutex.unlock();
}

// private 
void A1KFCombineLOWithFoot::load_casadi_functions() {
    // load casadi functions
    process_func = casadi::external("process", "/tmp/casadi_kf_baseline3_process.so");
    process_jac_func = casadi::external("process_jac", "/tmp/casadi_kf_baseline3_process_jac.so");
    measure_func = casadi::external("meas", "/tmp/casadi_kf_baseline3_meas.so");
    measure_jac_func = casadi::external("meas_jac", "/tmp/casadi_kf_baseline3_meas_jac.so");
}

void A1KFCombineLOWithFoot::process(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state, 
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


void A1KFCombineLOWithFoot::measure(Eigen::Matrix<double, EKF_STATE_SIZE, 1> state, 
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