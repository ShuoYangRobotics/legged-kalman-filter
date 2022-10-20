#include <iostream>

// for vector_t
#include <ocs2_core/Types.h>
// for vector3_t
#include <ocs2_legged_robot/common/Types.h>
// for model settings
#include <ocs2_legged_robot/common/ModelSettings.h>
// PinocchioInterface
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
// CentroidalModelInfo
#include <ocs2_centroidal_model/CentroidalModelInfo.h>
// createPinocchioInterface
#include <ocs2_centroidal_model/FactoryFunctions.h>
// this makes inocchioInterfacePtr_->getModel() work
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
// CentroidalModelPinocchioMapping
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
// PinocchioEndEffectorKinematics
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
using namespace ocs2;
using namespace legged_robot;

int main(int argc, char ** argv)
{
    bool verbose = true;
  
    const std::string taskFile = "/home/REXOperator/legged_ctrl_ws/src/legged_ctrl/config/task.info";
    const std::string urdfFile = "/home/REXOperator/legged_ctrl_ws/src/legged_ctrl/urdf/a1_description/urdf/a1.urdf";
    const std::string referenceFile = "/home/REXOperator/legged_ctrl_ws/src/legged_ctrl/config/reference.info";
    ModelSettings modelSettings_ = loadModelSettings(taskFile, "model_settings", verbose);



    std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;
    CentroidalModelInfo centroidalModelInfo_;
    // PinocchioInterface
    pinocchioInterfacePtr_.reset(
        new PinocchioInterface(centroidal_model::createPinocchioInterface(urdfFile, modelSettings_.jointNames)));

    // CentroidalModelInfo
    centroidalModelInfo_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterfacePtr_, centroidal_model::loadCentroidalType(taskFile),
        centroidal_model::loadDefaultJointState(pinocchioInterfacePtr_->getModel().nq - 6, referenceFile),
        modelSettings_.contactNames3DoF, modelSettings_.contactNames6DoF);  

    CentroidalModelPinocchioMapping pinocchio_mapping(centroidalModelInfo_);

    PinocchioEndEffectorKinematics ee_kinematics(*pinocchioInterfacePtr_, pinocchio_mapping,
                                                modelSettings_.contactNames3DoF);
    // must set this manually                                                    
    ee_kinematics.setPinocchioInterface(*pinocchioInterfacePtr_);


    // test pinocchio kinematics 
    vector_t measured_q_ = vector_t(pinocchioInterfacePtr_->getModel().nq);
    vector_t measured_v_ = vector_t(pinocchioInterfacePtr_->getModel().nv); measured_v_.setZero();
    measured_q_ << 0.0, 0.0, 0.0,                    // position x y z 
                   0.0, 0.0, 0.0,                      // yaw, pitch, roll
                   0.05, 0.72, -1.44,                   // front-left 
                   -0.05, 0.72, -1.44,                  // rear-left 
                   0.05, 0.72, -1.44,                   // front-right 
                   -0.05, 0.72, -1.44;                  // rear-right 

    const auto& model = pinocchioInterfacePtr_->getModel();
    auto& data = pinocchioInterfacePtr_->getData();
    pinocchio::forwardKinematics(model, data, measured_q_, measured_v_);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::computeJointJacobians(model, data);

    // the angle
    std::vector<vector3_t> pos_measured = ee_kinematics.getPosition(vector_t());
    std::cout << "front-left  pos_measured"<< std::endl << pos_measured[0] << std::endl; // front-left 
    std::cout << "front-right  pos_measured"<< std::endl << pos_measured[1] << std::endl; // front-right 
    std::cout << "rear-left  pos_measured"<< std::endl << pos_measured[2] << std::endl; // rear-left 
    std::cout << "rear-right  pos_measured"<< std::endl << pos_measured[3] << std::endl; // rear-right 

    // foot force and dynamics 
    vector_t foot_force = vector_t(pinocchioInterfacePtr_->getModel().nq - 6);
    foot_force <<    0.0, 0.0, centroidalModelInfo_.robotMass*pinocchioInterfacePtr_->getModel().gravity/4.0,                   // front-left 
                     0.0, 0.0, centroidalModelInfo_.robotMass*pinocchioInterfacePtr_->getModel().gravity/4.0,                  // rear-left 
                     0.0, 0.0, centroidalModelInfo_.robotMass*pinocchioInterfacePtr_->getModel().gravity/4.0,                   // front-right 
                     0.0, 0.0, centroidalModelInfo_.robotMass*pinocchioInterfacePtr_->getModel().gravity/4.0;                  // rear-right 

    vector_t joint_tau = vector_t(pinocchioInterfacePtr_->getModel().nq); joint_tau.setZero();
    
    // how to get $M(q)\ddot{q} + h(q,\dot{q}) = S\tau + J_c^T(q)f$

    return 0;
}