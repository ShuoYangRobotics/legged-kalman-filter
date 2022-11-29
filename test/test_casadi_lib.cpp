#include <Eigen/Dense>
#include <iostream>

// // for model settings
// #include <ocs2_legged_robot/common/ModelSettings.h>
// // PinocchioInterface
// #include <ocs2_pinocchio_interface/PinocchioInterface.h>
// // CentroidalModelInfo
// #include <ocs2_centroidal_model/CentroidalModelInfo.h>
// // createPinocchioInterface
// #include <ocs2_centroidal_model/FactoryFunctions.h>
// // CentroidalModelPinocchioMapping
// #include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>

#include <pinocchio/math/rotation.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/parsers/urdf.hpp> //load urdf 
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>  // getFrameJacobian
#include <pinocchio/algorithm/kinematics.hpp>                     //pinocchio::Data
#include <pinocchio/autodiff/casadi.hpp> // casadi diff related
#include <pinocchio/algorithm/crba.hpp>       //   computeCRBA
#include <pinocchio/algorithm/rnea.hpp>       //  nonLinearEffects
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <casadi/casadi.hpp>

bool almost_equal(double a, double b, double epsilon) {
    return std::abs(a - b) < epsilon;
}
// load casadi library from casadi_lib folder
int main(int argc, char **argv) {

//     // load lib, notice this .so is copied to /tmp by CMakeLists.txt (line 46)
//     casadi::Function baseline1_process_func = casadi::external("process",
//     "/tmp/casadi_kf_baseline1_f.so");

//     // prepare input
//     Eigen::VectorXd xk(10); xk << -0.0000,
//     0.0000,
//     0.0000,
//    -0.0005,
//     0.0003,
//     0.0003,
//    -0.0000,
//    -0.0000,
//    -0.0000,
//     0.0352;

//     Eigen::VectorXd uk(7); uk << -0.0003,
//    -0.0017,
//    -0.0001,
//    -0.2953,
//     0.3299,
//     9.7715,
//     0.0020;
//     Eigen::VectorXd uk1(7); uk1 <<    -0.0001,
//    -0.0023,
//    -0.0004,
//    -0.3002,
//     0.3309,
//     9.7789,
//     0.0020;

//     Eigen::VectorXd dt(1); dt<< 0.0020;

//     // convert input to casadi::DM
//     std::vector<double> xk_vec;
//     xk_vec.resize(xk.size());
//     Eigen::VectorXd::Map(&xk_vec[0], xk.size()) = xk;

//     std::vector<double> uk_vec;
//     uk_vec.resize(uk.size());
//     Eigen::VectorXd::Map(&uk_vec[0], uk.size()) = uk;

//     std::vector<double> uk1_vec;
//     uk1_vec.resize(uk1.size());
//     Eigen::VectorXd::Map(&uk1_vec[0], uk1.size()) = uk1;

//     std::vector<double> dt_vec;
//     dt_vec.resize(dt.size());
//     Eigen::VectorXd::Map(&dt_vec[0], dt.size()) = dt;

//     // assemble input into arg vector
//     std::vector<casadi::DM> arg = {casadi::DM(xk_vec), 
//          casadi::DM(uk_vec), 
//          casadi::DM(uk1_vec), 
//          casadi::DM(dt_vec)};

//     std::vector<casadi::DM> res = baseline1_process_func(arg);

//     std::vector<double> res_vec = std::vector<double>(res.at(0));
//     Eigen::VectorXd xk_new = Eigen::VectorXd::Map(res_vec.data(), res_vec.size());
//     Eigen::VectorXd xk_des(10); xk_des <<    -0.0000,
//     0.0000,
//     0.0000,
//    -0.0011,
//     0.0010,
//     0.0002,
//    -0.0000,
//    -0.0000,
//    -0.0000,
//     0.0372;
    
//     std::cout << xk_new.transpose() - xk_des.transpose()  << std::endl;

    // play with some casad basics 
    // Variables
    casadi::SX x = casadi::SX::sym("x");
    casadi::SX y = casadi::SX::sym("y");

    // Simple function
    casadi::SX z = x*x + y*y;
    casadi::SX state = vertcat(x, y);
    casadi::SX dz = jacobian(z, state);
    std::cout << dz;

    // test some rotation stuff
    // casadi::SX qw = casadi::SX::sym("qw");
    // casadi::SX qx = casadi::SX::sym("qx");
    // casadi::SX qy = casadi::SX::sym("qy");
    // casadi::SX qz = casadi::SX::sym("qz");
    // Eigen::Quaternion<casadi::SX> c_q(qw, qx, qy, qz);
    // casadi::SX w0 = casadi::SX::sym("w0");
    // casadi::SX w1 = casadi::SX::sym("w1");
    // casadi::SX w2 = casadi::SX::sym("w2");
    // Eigen::Matrix<casadi::SX,3,1> w(w0, w1, w2);

    // // shitty cayley map 
    // Eigen::Matrix<casadi::SX,4,1> dq_coeff; 
    // dq_coeff[0] = 1;
    // dq_coeff.segment(1,3) = w;
    // dq_coeff = dq_coeff / (sqrt(1+w.squaredNorm()));
    // Eigen::Quaternion<casadi::SX> dq(dq_coeff);
    // Eigen::Quaternion<casadi::SX> next_q = c_q * dq;

    // std::cout << next_q.w() << std::endl;
    // std::cout << w[0] << std::endl;

    // load pinocchio urdf 
    // https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b-examples_a-model.html
    const std::string urdf_filename = "/home/REXOperator/legged_ctrl_ws/src/legged_ctrl/urdf/a1_description/urdf/a1.urdf";

    // Load the urdf model
    // add float base joint 
    pinocchio::JointModelComposite jointComposite(2);
    jointComposite.addJoint(pinocchio::JointModelTranslation());
    jointComposite.addJoint(pinocchio::JointModelSphericalZYX());
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_filename, jointComposite, model);
    std::cout << "model name: " << model.name << std::endl;
    
    // Create data required by the algorithms
    pinocchio::Data data(model);

    // bool verbose = true;
    // const std::string taskFile = "/home/REXOperator/legged_ctrl_ws/src/legged_ctrl/config/task.info";
    // const std::string urdfFile = "/home/REXOperator/legged_ctrl_ws/src/legged_ctrl/urdf/a1_description/urdf/a1.urdf";
    // const std::string referenceFile = "/home/REXOperator/legged_ctrl_ws/src/legged_ctrl/config/reference.info";
    // ocs2::legged_robot::ModelSettings modelSettings_ = ocs2::legged_robot::loadModelSettings(taskFile, "model_settings", verbose);



    // std::unique_ptr<ocs2::PinocchioInterface> pinocchioInterfacePtr_;
    // ocs2::CentroidalModelInfo centroidalModelInfo_;
    // // PinocchioInterface
    // pinocchioInterfacePtr_.reset(
    //     new ocs2::PinocchioInterface(ocs2::centroidal_model::createPinocchioInterface(urdfFile, modelSettings_.jointNames)));

    // pinocchio::Model model = pinocchioInterfacePtr_->getModel();
    // pinocchio::Data data = pinocchioInterfacePtr_->getData();

    // Eigen::VectorXd q = randomConfiguration(model);
    // q.segment(0,6).setZero();
    // std::cout << q << std::endl;
    // pinocchio::Model::TangentVectorType v(pinocchio::Model::TangentVectorType::Random(model.nv));
    // pinocchio::forwardKinematics(model, data, q);

    // // https://github.com/stack-of-tasks/pinocchio/blob/master/unittest/casadi-algo-derivatives.cpp

    // // notice namespace casadi and namespace pinocchio::casadi conflict with each other 
    // typedef casadi::SX ADScalar;
    // typedef pinocchio::ModelTpl<ADScalar> ADModel;
    // typedef ADModel::Data ADData;
    // ADModel ad_model = model.cast<ADScalar>();
    // ADData ad_data(ad_model);

    // casadi::SX cs_q = casadi::SX::sym("cq", model.nq);
    // casadi::SX cs_v = casadi::SX::sym("cv", model.nv);

    // typedef ADModel::ConfigVectorType ConfigVectorAD;
    // ConfigVectorAD q_ad(model.nq), v_ad(model.nv);
    // q_ad = Eigen::Map<ConfigVectorAD>(static_cast< std::vector<ADScalar> >(cs_q).data(),model.nq,1);
    // v_ad = Eigen::Map<ConfigVectorAD>(static_cast< std::vector<ADScalar> >(cs_v).data(),model.nv,1);
    // pinocchio::forwardKinematics(ad_model,ad_data,q_ad,v_ad);
    // pinocchio::updateFramePlacements(ad_model, ad_data);
    // pinocchio::computeJointJacobians(ad_model, ad_data);

    // // matrix_t ad_j_, dad_j_;
    // pinocchio::crba(ad_model, ad_data, q_ad);
    // ad_data.M.triangularView<Eigen::StrictlyLower>() = ad_data.M.transpose().triangularView<Eigen::StrictlyLower>();
    // pinocchio::nonLinearEffects(ad_model, ad_data, q_ad, v_ad);
    // std::vector<casadi::SX> ad_j_; 
    // static const std::vector<std::string> foot_names = {"LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"};
    // // ad_j_ = matrix_t(3 * centroidalModelInfo_.numThreeDofContacts, centroidalModelInfo_.generalizedCoordinatesNum);
    // for (size_t i = 0; i < 4; ++i)
    // {
    //     Eigen::Matrix<casadi::SX, 6, 18> jac;
    //     jac.setZero(6, 18);
    //     pinocchio::getFrameJacobian(ad_model, ad_data, ad_model.getBodyId(foot_names[i]), pinocchio::LOCAL_WORLD_ALIGNED, jac);
    //     Eigen::Matrix<casadi::SX, 3, 18> tmp = jac.template topRows<3>();
    //     casadi::SX ad_j_block(3, 18);

    //     for (int j = 0; j < 3; j++) {
    //         for (int k = 0; k < 18; k ++) {
    //             ad_j_block(j,k) = tmp(j,k);
    //         }
    //     }

    //     ad_j_.push_back(ad_j_block);
    // }
    // // std::cout << ad_j_[0] << std::endl;
    // // std::cout << "model.nq " << model.nq << std::endl;

    // int foot_id = 1;
    // // save ad_j_ as casadi function
    // casadi::Function eval_jac_fl("eval_jac_fl",
    //                             casadi::SXVector {cs_q, cs_v},
    //                             casadi::SXVector {ad_j_[foot_id]});   

    // // compare eval_jac_fl result with numerical result 
    // std::vector<double> q_vec((size_t)model.nq);
    // Eigen::Map<pinocchio::Model::ConfigVectorType>(q_vec.data(),model.nq,1) = q;
    // std::vector<double> v_vec((size_t)model.nv);
    // Eigen::Map<pinocchio::Model::TangentVectorType>(v_vec.data(),model.nv,1) = v;
    // std::vector<casadi::DM> res = eval_jac_fl(casadi::DMVector {q_vec,v_vec});
    // std::vector<double> j_vec = std::vector<double>(res.at(0));
    // Eigen::Matrix<double, 3, 18> nj = Eigen::Matrix<double, 3, 18>(j_vec.data());
    // std::cout << "casadi jac result" << std::endl;
    // std::cout << nj << std::endl;

    // Eigen::Matrix<double, 6, 18> jac;
    // jac.setZero(6, 18);
    // std::cout << model.getBodyId(foot_names[foot_id]) << std::endl;
    // std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    // pinocchio::computeJointJacobians(model, data,q);
    // pinocchio::framesForwardKinematics(model, data,q);
    // pinocchio::getFrameJacobian(model, data, model.getBodyId(foot_names[foot_id]), pinocchio::LOCAL_WORLD_ALIGNED, jac);

    // std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    // auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "compute time: " << microseconds.count() << "us" << std::endl;

    // std::cout << "numerical jac result" << std::endl;
    // std::cout << jac.template topRows<3>() << std::endl;

    // start = std::chrono::system_clock::now();
    // pinocchio::crba(model, data, q);
    // data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
    // end = std::chrono::system_clock::now();
    // microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "crba compute time: " << microseconds.count() << "us" << std::endl;

    // start = std::chrono::system_clock::now();
    // // compute references
    // Eigen::MatrixXd dtau_dq_ref(model.nv,model.nv), dtau_dv_ref(model.nv,model.nv), dtau_da_ref(model.nv,model.nv);
    // dtau_dq_ref.setZero(); dtau_dv_ref.setZero(); dtau_da_ref.setZero();
    
    // pinocchio::computeRNEADerivatives(model,data,q,q,q,dtau_dq_ref,dtau_dv_ref,dtau_da_ref);
    // dtau_da_ref.triangularView<Eigen::StrictlyLower>() = dtau_da_ref.transpose().triangularView<Eigen::StrictlyLower>();
    // end = std::chrono::system_clock::now();
    // microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "computeRNEADerivatives compute time: " << microseconds.count() << "us" << std::endl;

    // pinocchio::computeJointJacobiansTimeVariation(model, data, measured_q_, measured_v_);
    // dad_j_ = matrix_t(3 * centroidalModelInfo_.numThreeDofContacts, centroidalModelInfo_.generalizedCoordinatesNum);
    // for (size_t i = 0; i < centroidalModelInfo_.numThreeDofContacts; ++i)
    // {
    //     Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> jac;
    //     jac.setZero(6, centroidalModelInfo_.generalizedCoordinatesNum);
    //     pinocchio::getFrameJacobianTimeVariation(model, data, centroidalModelInfo_.endEffectorFrameIndices[i],
    //                                             pinocchio::LOCAL_WORLD_ALIGNED, jac);
    //     dad_j_.block(3 * i, 0, 3, centroidalModelInfo_.generalizedCoordinatesNum) = jac.template topRows<3>();
    // }

    // test joint order is we use default a1_urdf

    Eigen::VectorXd measured_q_ = Eigen::VectorXd(model.nq);
    Eigen::VectorXd measured_v_ = Eigen::VectorXd(model.nv); measured_v_.setZero();
    measured_q_ << 0.0, 0.0, 0.0,                    // position x y z 
                   0.0, 0.0, 0.0,                      // yaw, pitch, roll
                   0.05, 0.72, -1.44,                   // front-left 
                   0.05, 0.72, -1.44,                   // front-right 
                   -0.05, 0.72, -1.44,                  // rear-left 
                   -0.05, 0.72, -1.44;                  // rear-right 
    pinocchio::forwardKinematics(model,data,measured_q_,measured_v_);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::computeJointJacobians(model, data);

    std::vector<Eigen::Vector3d> pos_measured;

    static const std::vector<std::string> foot_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    for (size_t i = 0; i < 4; ++i)
    {
        pos_measured.push_back(data.oMf[model.getBodyId(foot_names[i])].translation());
    }
    std::cout << "front-left  pos_measured"<< std::endl << pos_measured[0] << std::endl; // front-left 
    std::cout << "front-right  pos_measured"<< std::endl << pos_measured[1] << std::endl; // front-right 
    std::cout << "rear-left  pos_measured"<< std::endl << pos_measured[2] << std::endl; // rear-left 
    std::cout << "rear-right  pos_measured"<< std::endl << pos_measured[3] << std::endl; // rear-right 

    return 0;
}