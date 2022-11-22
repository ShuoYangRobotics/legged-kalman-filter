/*
 * In this script we read ROS bags to do moving horizon estimation 
 */
// c++
#include <fstream>
#include <signal.h>

// ROS
#include <ros/ros.h>

// load yaml parameters 

// measurement array 

#include "Measurement.hpp"
#include "MeasurementQueue.hpp"
// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) {
   exit(signum);
}

int main(int argc, char **argv) {
    // Register signal and signal handler
    signal(SIGINT, signal_callback_handler);
    ros::init(argc, argv, "mhe_test");
    ros::NodeHandle nh;


    // /* subscribers */
    // ros::Subscriber opti_sub = nh.subscribe("/mocap_node/Robot_1/pose", 30, opti_callback);

    // message_filters::Subscriber<sensor_msgs::Imu> imu_sub;
    // message_filters::Subscriber<sensor_msgs::JointState> joint_state_sub;

    // // listen to hardware imu and joint foot
    // // notice we know they have the same time stamp
    // imu_sub.subscribe(nh,"/hardware_a1/imu", 30);
    // joint_state_sub.subscribe(nh,"/hardware_a1/joint_foot", 30);
    // // https://answers.ros.org/question/172772/writing-a-c-class-with-message_filters-member/
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Imu, sensor_msgs::JointState> MySyncPolicy;
    // // ExactTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(30), imu_sub, joint_state_sub);

    // sync.registerCallback(boost::bind(&a1_sensor_callback, _1, _2));
    // // foot IMUs
    // ros::Subscriber fl_foot_IMU_sub = nh.subscribe("/WT901_49_Data", 30, fl_foot_IMU_callback);
    // ros::Subscriber fr_foot_IMU_sub = nh.subscribe("/WT901_48_Data", 30, fr_foot_IMU_callback);
    // ros::Subscriber rl_foot_IMU_sub = nh.subscribe("/WT901_50_Data", 30, rl_foot_IMU_callback);
    // ros::Subscriber rr_foot_IMU_sub = nh.subscribe("/WT901_47_Data", 30, rr_foot_IMU_callback);


    // MHE::BodyIMUMeasurement m1(0);
    // MHE::LegMeasurement m2(0.1);
    // MHE::FootIMUMeasurement m3(-0.3);
    // MHE::FootForceMeasurement m4(0.2);

    MHE::MeasureQueue mq;
    Eigen::Vector3d zero_vec; zero_vec.setZero();
    mq.push(0, zero_vec, zero_vec);
    mq.push(0.1,  zero_vec, zero_vec);
    mq.push(-0.30,  zero_vec, zero_vec);
    mq.push(0.5,  zero_vec, zero_vec);
    mq.push(0.04,  zero_vec, zero_vec);
    mq.push(0.01,  zero_vec, zero_vec,0);
    mq.push(0.012,  zero_vec, zero_vec,2);
    mq.push(0.042,  zero_vec, zero_vec,3);

    mq.print_queue();

    std::vector<MHE::Measurement*> horiz_meas = mq.getHorizon(2);
    std::cout << horiz_meas.size() << std::endl;

    // help to visualize what's inside the horizon
    mq.print_queue(horiz_meas);

    try
    {
        horiz_meas = mq.getHorizon(99);
        std::cout << horiz_meas.size() << std::endl;
    }
    catch (int e)
    {
        std::cout << "An exception occurred. Exception Nr. " << e << std::endl;
    }
}