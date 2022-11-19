#include <Eigen/Dense>
#include <iostream>
#include <casadi/casadi.hpp>
#include <pinocchio/math/rotation.hpp>
#include <pinocchio/spatial/se3.hpp>

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

    using namespace casadi;
    // play with some casad basics 
    // Variables
    SX x = SX::sym("x");
    SX y = SX::sym("y");

    // Simple function
    SX z = x*x + y*y;
    SX state = vertcat(x, y);
    SX dz = jacobian(z, state);
    std::cout << dz;

    // test some rotation stuff

    using namespace pinocchio;
    SX qw = SX::sym("qw");
    SX qx = SX::sym("qx");
    SX qy = SX::sym("qy");
    SX qz = SX::sym("qz");
    Eigen::Quaternion<SX> c_q(qw, qx, qy, qz);
    SX w0 = SX::sym("w0");
    SX w1 = SX::sym("w1");
    SX w2 = SX::sym("w2");
    Eigen::Matrix<SX,3,1> w(w0, w1, w2);

    // shitty cayley map 
    Eigen::Matrix<SX,4,1> dq_coeff; 
    dq_coeff[0] = 1;
    dq_coeff.segment(1,3) = w;
    dq_coeff = dq_coeff / (sqrt(1+w.squaredNorm()));
    Eigen::Quaternion<SX> dq(dq_coeff);
    Eigen::Quaternion<SX> next_q = c_q * dq;

    std::cout << next_q.w() << std::endl;
    std::cout << w[0] << std::endl;

    return 0;
}