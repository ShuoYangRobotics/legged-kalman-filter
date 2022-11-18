#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <queue>
#include <Eigen/Dense>
#include "Measurement.hpp"

namespace MHE {
typedef std::priority_queue<Measurement*, std::vector<Measurement*>, MeasurementCompare> pq_meas_type;
class MeasureQueue {
    public:
        MeasureQueue() {}
        // destructor, clean data structure
        ~MeasureQueue() {
            // vec_meas only keeps pointer, do not need to clean it

            // all elements in pq_meas must be properly deleted
            while (!pq_meas.empty()) {
                Measurement* tmp = pq_meas.top();
                delete tmp;
                pq_meas.pop();
            }
        }

        /* 
         * a list of push operations, they all align with measurement constructor functions
         */
        // BodyIMUMeasurement
        bool push(double _t,
            Eigen::Vector3d _imu_acc,
            Eigen::Vector3d _imu_gyro) {
            Measurement* tmp = new BodyIMUMeasurement(_t, _imu_acc, _imu_gyro);
            pq_meas.push(tmp);
        }

        // LegMeasurement
        bool push(double _t,
            Eigen::Matrix<double, 12, 1> _joint_pos,
            Eigen::Matrix<double, 12, 1> _joint_vel) {
            Measurement* tmp = new LegMeasurement(_t, _joint_pos, _joint_vel);
            pq_meas.push(tmp);
        }

        // BodyIMUMeasurement & LegMeasurement
        bool push(double _t,
            Eigen::Vector3d _imu_acc,
            Eigen::Vector3d _imu_gyro,
            Eigen::Matrix<double, 12, 1> _joint_pos,
            Eigen::Matrix<double, 12, 1> _joint_vel) {
            Measurement* tmp1 = new BodyIMUMeasurement(_t, _imu_acc, _imu_gyro);
            Measurement* tmp2 = new LegMeasurement(_t, _joint_pos, _joint_vel);
            pq_meas.push(tmp1);
            pq_meas.push(tmp2);
        }

        // LegMeasurement with tau
        bool push(double _t,
            Eigen::Matrix<double, 12, 1> _joint_pos,
            Eigen::Matrix<double, 12, 1> _joint_vel,
            Eigen::Matrix<double, 12, 1> _joint_tau) {
            Measurement* tmp = new LegMeasurement(_t, _joint_pos, _joint_vel, _joint_tau);
            pq_meas.push(tmp);
        }

        // BodyIMUMeasurement & LegMeasurement with tau
        bool push(double _t,
            Eigen::Vector3d _imu_acc,
            Eigen::Vector3d _imu_gyro,
            Eigen::Matrix<double, 12, 1> _joint_pos,
            Eigen::Matrix<double, 12, 1> _joint_vel,
            Eigen::Matrix<double, 12, 1> _joint_tau) {
            Measurement* tmp1 = new BodyIMUMeasurement(_t, _imu_acc, _imu_gyro);
            Measurement* tmp2 = new LegMeasurement(_t, _joint_pos, _joint_vel, _joint_tau);
            pq_meas.push(tmp1);
            pq_meas.push(tmp2);
        }

        // FootIMUMeasurement
        bool push(double _t,
            Eigen::Vector3d _imu_acc,
            Eigen::Vector3d _imu_gyro,
            int _id) {
            Measurement* tmp = new FootIMUMeasurement(_t, _imu_acc, _imu_gyro, _id);
            pq_meas.push(tmp);
        }

        // FootForceMeasurement
        bool push(double _t,
            Eigen::Vector3d _foot_force_xyz,
            int _id) {
            Measurement* tmp = new FootForceMeasurement(_t, _foot_force_xyz, _id);
            pq_meas.push(tmp);
        }

        // PoseMeasurement
        bool push(double _t,
            Eigen::Vector3d _pos,
            Eigen::Quaterniond _quat) {
            Measurement* tmp = new PoseMeasurement(_t, _pos, _quat);
            pq_meas.push(tmp);
        }

        bool pop() {
            pq_meas.pop();
        }

        // get measurement within the horizon
        std::vector<Measurement*> getHorizon(int horizon_length) {
            if (horizon_length > pq_meas.size()) {
                // something is wrong, throw error 
                std::cout << "horizon cannot be longer than the pq_meas.size()" << std::endl;
                throw 203;
            }
            int vec_size = 0;
            std::vector<Measurement*> tmp_vec_meas;
            tmp_vec_meas.clear();
            pq_meas_type tmp_pq_meas = pq_meas; // copy a pq_meas_type
            while (!tmp_pq_meas.empty()&& vec_size < horizon_length) {
                Measurement* tmp = tmp_pq_meas.top();
                tmp_vec_meas.push_back(tmp);
                tmp_pq_meas.pop();
                vec_size++;
            }
            return tmp_vec_meas;
        }

        // dump elements to vec_meas with 
        void dump_vec() {
            vec_meas.clear();
            pq_meas_type tmp_pq_meas = pq_meas; // copy a pq_meas_type
            while (!tmp_pq_meas.empty()) {
                Measurement* tmp = tmp_pq_meas.top();
                vec_meas.push_back(tmp);
                tmp_pq_meas.pop();
            }
        }

        int print_queue() {
            dump_vec();
            for (Measurement* element : vec_meas) {
                std::cout << "(" << element->getTime() << "\t," << element->getType() << ") ";
            }
            std::cout << std::endl;
            return vec_meas.size();
        }

    private:
        pq_meas_type pq_meas;
        // only help print queue and construct MHE
        std::vector<Measurement*> vec_meas;

        // keep track of current number and maximum number of elements 
        // maybe let out users handle this?
        int max_queue_size;
};




}