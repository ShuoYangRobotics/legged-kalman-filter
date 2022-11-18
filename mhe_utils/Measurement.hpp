#pragma once

#include <memory>
#include <Eigen/Dense>

namespace MHE {
enum MeasureType { 
    BODY_IMU, // might further decompose this into BODY_ACC and BODY_GYRO
    LEG,
    FOOT_IMU,
    FOOT_FORCE,
    DIRECT_POSE
};

// define abstract measurement type 
// https://stackoverflow.com/questions/19678011/c-multiple-type-array
// time is the only common variable
class Measurement {
    public:
        Measurement () {}
        virtual MeasureType getType() = 0;
        virtual double getTime() = 0;
};

// each measurement data struct has 
//    t   - the timestamp of the sensor package
//   type - MeasureType
//   actual data that depepnds on measurement type

// measurement from the robot hardaware IMU or gazebo sim
class BodyIMUMeasurement: public Measurement {
    public:
        BodyIMUMeasurement () {}

        BodyIMUMeasurement (
            double _t,
            Eigen::Vector3d _imu_acc,
            Eigen::Vector3d _imu_gyro
        ) {
            t = _t;
            imu_acc = _imu_acc;
            imu_gyro = _imu_gyro;
        }
        MeasureType getType() {return type;}
        double getTime() {return t;}

        Eigen::Vector3d imu_acc;
        Eigen::Vector3d imu_gyro;

        MeasureType type = BODY_IMU;
        double t; 

};

// measurement from the robot hardaware joint encoders or gazebo sim
// notice we do not assume each leg is individual 
class LegMeasurement: public Measurement {
    public:
        LegMeasurement () {}

        LegMeasurement (
            double _t,
            Eigen::Matrix<double, 12, 1> _joint_pos,
            Eigen::Matrix<double, 12, 1> _joint_vel
        ) {
            t = _t;
            joint_pos = _joint_pos;
            joint_vel = _joint_vel;
        }

        LegMeasurement (
            double _t,
            Eigen::Matrix<double, 12, 1> _joint_pos,
            Eigen::Matrix<double, 12, 1> _joint_vel,
            Eigen::Matrix<double, 12, 1> _joint_tau
        ) {
            t = _t;
            joint_pos = _joint_pos;
            joint_vel = _joint_vel;
            joint_tau = _joint_tau;
        }
        MeasureType getType() {return type;}
        double getTime() {return t;}

        Eigen::Matrix<double, 12, 1> joint_pos;
        Eigen::Matrix<double, 12, 1> joint_vel;
        Eigen::Matrix<double, 12, 1> joint_tau;

        MeasureType type = LEG;
        double t; 

};

// measurement from the foot IMUs
// notice we DO assume each leg is an individual IMU package
class FootIMUMeasurement: public Measurement {
    public:
        FootIMUMeasurement () {}

        FootIMUMeasurement (
            double _t,
            Eigen::Vector3d _imu_acc,
            Eigen::Vector3d _imu_gyro,
            int _id
        ) {
            t = _t;
            imu_acc = _imu_acc;
            imu_gyro = _imu_gyro;
            id = _id;
        }
        MeasureType getType() {return type;}
        double getTime() {return t;}

        int id;
        Eigen::Vector3d imu_acc;
        Eigen::Vector3d imu_gyro;

        MeasureType type = FOOT_IMU;
        double t; 
};
// measurement from the foot forces
// notice we assume they  are individual
class FootForceMeasurement: public Measurement {
    public:
        FootForceMeasurement () {}

        FootForceMeasurement (
            double _t,
            Eigen::Vector3d _foot_force_xyz,
            int _id
        ) {
            t = _t;
            foot_force_xyz = _foot_force_xyz;
            id = _id;
        }
        MeasureType getType() {return type;}
        double getTime() {return t;}

        int id;
        Eigen::Vector3d foot_force_xyz;
        MeasureType type = FOOT_FORCE;
        double t; 

};
// direct pose measurement
// notice we assume they 
class PoseMeasurement: public Measurement {
    public:
        PoseMeasurement () {}

        PoseMeasurement (
            double _t,
            Eigen::Vector3d _pos,
            Eigen::Quaterniond _quat
        ) {
            t = _t;
            pos = _pos;
            quat = _quat;
        }
        MeasureType getType() {return type;}
        double getTime() {return t;}

        Eigen::Vector3d pos;
        Eigen::Quaterniond quat;
        MeasureType type = DIRECT_POSE;
        double t; 

};

// https://stackoverflow.com/questions/16111337/declaring-a-priority-queue-in-c-with-a-custom-comparator
// The expression comp(a,b), where comp is an object of this type and a and b are elements in the container, shall return true if a is considered to go before b in the strict weak ordering the function defines.
class MeasurementCompare
{
    public:
        bool operator() (Measurement* a, Measurement* b)
        {
            if (a->getTime() > b->getTime()) {
                return true;
            } else {
                return false;
            }
        }
};


}