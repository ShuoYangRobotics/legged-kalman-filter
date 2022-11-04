// c++
#include <fstream>
#include <signal.h>

// ROS
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/console.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>



#include "../A1KFCombineLOWithFootTerrain.h"
#include "../A1KFQP.h"

A1KFCombineLOWithFootTerrain kf;  // Kalman filter Baseline 4 with foot, with terrain factor 
A1KFQP kf_qp;

A1SensorData data;
double curr_t;

// debug print filtered data
ros::Publisher filterd_imu_pub;
ros::Publisher filterd_joint_pub;
ros::Publisher filterd_pos_pub;
ros::Publisher filterd_pos_qp_pub;

bool first_sensor_received = false;
void sensor_callback(const sensor_msgs::Imu::ConstPtr& imu_msg, const sensor_msgs::JointState::ConstPtr& joint_msg) {

    // std::cout<<"sensor_callback"<<std::endl;
    double t = imu_msg->header.stamp.toSec();

    // assemble sensor data
    Eigen::Vector3d acc = Eigen::Vector3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    Eigen::Vector3d ang_vel = Eigen::Vector3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);

    Eigen::Matrix<double, NUM_DOF,1> joint_pos;
    Eigen::Matrix<double, NUM_DOF,1> joint_vel;
    Eigen::Matrix<double, NUM_LEG,1> plan_contacts;
    for (int i = 0; i < NUM_DOF; ++i) {
        joint_pos[i] = joint_msg->position[i];
        joint_vel[i] = joint_msg->velocity[i];
    }
    for (int i = 0; i < NUM_LEG; ++i) {
        // plan_contacts[i] = joint_msg->velocity[NUM_DOF + i];
        plan_contacts[i] = joint_msg->effort[NUM_DOF + i]>100?1.0:0.0 ;
        // joint_foot_msg.effort[NUM_DOF + i]  --> foot_force[i];
    }

    double dt;
    data.input_imu(acc, ang_vel);
    data.input_leg(joint_pos, joint_vel, plan_contacts);

    //TODO: init filter if there is no opti data after 0.01s?

    if ( !kf_qp.is_inited()) {
        // the callback is called the first time, filter may not be inited
        dt = 0;
        curr_t = t;
        data.input_dt(dt);
        // init the filter using optitrack data, not here
        kf_qp.init_filter(data);
        kf.init_filter(data);
    } else if ( !kf_qp.is_inited()) {
        // filter may not be inited even after the callback is called multiple times
        dt = t- curr_t;
        data.input_dt(dt);
        curr_t = t;
    } else {
        dt = t- curr_t;
        
        data.input_dt(dt);
        kf_qp.update_filter(data);
        kf.update_filter(data);
        curr_t = t;
    }
    // debug print filtered data
    sensor_msgs::Imu filterd_imu_msg;
    sensor_msgs::JointState filterd_joint_msg;
    filterd_imu_msg.header.stamp = ros::Time::now();
    filterd_imu_msg.linear_acceleration.x = data.acc[0];
    filterd_imu_msg.linear_acceleration.y = data.acc[1];
    filterd_imu_msg.linear_acceleration.z = data.acc[2];

    filterd_imu_msg.angular_velocity.x = data.ang_vel[0];
    filterd_imu_msg.angular_velocity.y = data.ang_vel[1];
    filterd_imu_msg.angular_velocity.z = data.ang_vel[2];

    filterd_joint_msg.header.stamp = ros::Time::now();

    filterd_joint_msg.name = {"FL0", "FL1", "FL2",
                           "FR0", "FR1", "FR2",
                           "RL0", "RL1", "RL2",
                           "RR0", "RR1", "RR2",
                           "FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    filterd_joint_msg.position.resize(NUM_DOF + NUM_LEG);
    filterd_joint_msg.velocity.resize(NUM_DOF + NUM_LEG);
    filterd_joint_msg.effort.resize(NUM_DOF + NUM_LEG);
    for (int i = 0; i < NUM_DOF; ++i) {
        filterd_joint_msg.position[i] = data.joint_pos[i];
        filterd_joint_msg.velocity[i] = data.joint_vel[i];
    }
    Eigen::Vector4d estimated_contact = kf_qp.get_contacts();
    for (int i = 0; i < NUM_LEG; ++i) {
        filterd_joint_msg.velocity[NUM_DOF+i] = estimated_contact[i];
    }
    filterd_imu_pub.publish(filterd_imu_msg);
    filterd_joint_pub.publish(filterd_joint_msg);

    Eigen::Matrix<double, EKF_STATE_SIZE,1> kf_state = kf.get_state();
    nav_msgs::Odometry filterd_pos_msg;
    filterd_pos_msg.header.stamp = ros::Time::now();
    filterd_pos_msg.pose.pose.position.x = kf_state[0];
    filterd_pos_msg.pose.pose.position.y = kf_state[1];
    filterd_pos_msg.pose.pose.position.z = kf_state[2];
    filterd_pos_msg.twist.twist.linear.x = kf_state[3];
    filterd_pos_msg.twist.twist.linear.y = kf_state[4];
    filterd_pos_msg.twist.twist.linear.z = kf_state[5];

    filterd_pos_pub.publish(filterd_pos_msg);


    Eigen::Matrix<double, EKF_STATE_SIZE,1> kf_qp_state = kf_qp.get_state();
    nav_msgs::Odometry filterd_pos_qp_msg;
    filterd_pos_qp_msg.header.stamp = ros::Time::now();
    filterd_pos_qp_msg.pose.pose.position.x = kf_qp_state[0];
    filterd_pos_qp_msg.pose.pose.position.y = kf_qp_state[1];
    filterd_pos_qp_msg.pose.pose.position.z = kf_qp_state[2];
    filterd_pos_qp_msg.twist.twist.linear.x = kf_qp_state[3];
    filterd_pos_qp_msg.twist.twist.linear.y = kf_qp_state[4];
    filterd_pos_qp_msg.twist.twist.linear.z = kf_qp_state[5];

    filterd_pos_qp_pub.publish(filterd_pos_qp_msg);

    first_sensor_received = true;
    return;

}

// if optitrack data is available, use it to update the filter
// notice this is sort of an asynchronous callback
double opti_dt = 0;
double opti_curr_t = 0;
ros::Publisher filterd_opti_vel_pub;
bool opti_callback_first_received = false;
void opti_callback(const geometry_msgs::PoseStamped::ConstPtr& opti_msg) {
    // std::cout<<"opti_callback"<<std::endl;
    double opti_t = opti_msg->header.stamp.toSec();

    Eigen::Matrix<double, 3, 1> opti_pos; 
    opti_pos << opti_msg->pose.position.x, opti_msg->pose.position.y, opti_msg->pose.position.z;

    // only init data after sensor and opti are both received
    // if ( !kf.is_inited() && first_sensor_received == true) {
    //     kf.init_filter(data, opti_pos);
    // }
        
    // update sensor data
    if (opti_callback_first_received == false) {
        opti_curr_t = opti_t;
        opti_dt = 0;
        data.input_opti_dt(opti_dt);
        data.input_opti_pos(opti_pos);
    } else {
        // only send data to KF if it is initialized and optitrack generates reliable vel data

        opti_dt = opti_t - opti_curr_t;
        data.input_opti_dt(opti_dt);
        data.input_opti_pos(opti_pos);
    }

    // only update filter after filter init and data opti vel is ready
    // if (kf.is_inited() && data.opti_vel_ready()) {
    //     kf.update_filter_with_opti(data);
    // }

    // debug print
    nav_msgs::Odometry filterd_opti_vel_msg;
    if (data.opti_vel_ready()) {
        filterd_opti_vel_msg.header.stamp = ros::Time::now();
        filterd_opti_vel_msg.twist.twist.linear.x = data.opti_vel[0];
        filterd_opti_vel_msg.twist.twist.linear.y = data.opti_vel[1];
        filterd_opti_vel_msg.twist.twist.linear.z = data.opti_vel[2];
    }
    filterd_opti_vel_pub.publish(filterd_opti_vel_msg);

    opti_curr_t = opti_t;
    opti_callback_first_received = true;
}

std::ofstream myFile;
// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) {
   std::cout << "Caught signal " << signum << std::endl;
   myFile.close();
   // Terminate program
   exit(signum);
}

int main(int argc, char **argv) {
    // Register signal and signal handler
    signal(SIGINT, signal_callback_handler);
    ros::init(argc, argv, "run_bag");
    ros::NodeHandle nh;

    /* subscribers */
    // ros::Subscriber opti_sub = nh.subscribe("/mocap_node/Robot_1/pose", 30, opti_callback);

    message_filters::Subscriber<sensor_msgs::Imu> imu_sub;
    message_filters::Subscriber<sensor_msgs::JointState> joint_state_sub;

    // listen to hardware imu and joint foot
    // notice we know they have the same time stamp
    imu_sub.subscribe(nh,"/hardware_a1/imu", 30);
    joint_state_sub.subscribe(nh,"/hardware_a1/joint_foot", 30);
    // https://answers.ros.org/question/172772/writing-a-c-class-with-message_filters-member/
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Imu, sensor_msgs::JointState> MySyncPolicy;
    // ExactTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(30), imu_sub, joint_state_sub);

    sync.registerCallback(boost::bind(&sensor_callback, _1, _2));

    /* publishers */
    filterd_imu_pub = nh.advertise<sensor_msgs::Imu>("/a1_filterd_imu", 30);
    filterd_joint_pub = nh.advertise<sensor_msgs::JointState>("/a1_filterd_joint", 30);
    filterd_pos_pub = nh.advertise<nav_msgs::Odometry>("/a1_filterd_pos", 30);
    filterd_pos_qp_pub = nh.advertise<nav_msgs::Odometry>("/a1_filterd_pos_qp", 30);
    // this is the smoothed velocity we get from optitrack
    filterd_opti_vel_pub = nh.advertise<nav_msgs::Odometry>("/a1_opti_filterd_vel", 30);
    
    myFile = std::ofstream("/home/REXOperator/legged_ctrl_ws/bags/output/1017_aaron_lab1.csv");
    // ros loop 
    ros::Rate loop_rate(100);
    while (ros::ok()) {
        // Eigen::Matrix<double, 10,1> est_state = kf.get_state();
        Eigen::Matrix<double, EKF_STATE_SIZE,1> est_state = kf_qp.get_state();
        // std::cout << "est_state: " << est_state.transpose() << std::endl;
        // save position to a csv file
        myFile << est_state[0] << "," << est_state[1] << "," << est_state[2] << "\n";

        loop_rate.sleep();
        ros::spinOnce();
    }
    return 0;
}