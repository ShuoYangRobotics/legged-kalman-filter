#include "../../utils/Utils.h"

bool almost_equal(double a, double b, double epsilon) {
    return std::abs(a - b) < epsilon;
}

int main(int argc, char **argv) {
    // test quaternion to euler angle conversion
    Eigen::Vector3d euler_angles = Utils::quat_to_euler(Eigen::Quaterniond(0.707, 0.0, 0.0, 0.707));
    std::cout << euler_angles.transpose() << std::endl;
    assert  (almost_equal(euler_angles[0], 0.0, 0.01));
    assert  (almost_equal(euler_angles[1], 0.0, 0.01));
    assert  (almost_equal(euler_angles[2],M_PI/2, 0.01));

    Eigen::Matrix3d rotMat = Eigen::Quaterniond(0.707, 0.0, 0.0, 0.707).toRotationMatrix();
    std::cout << rotMat << std::endl;

    Eigen::Quaterniond quat = Utils::euler_to_quat(euler_angles);
    std::cout << quat.w() << " " << quat.x() << " " << quat.y() << " " << quat.z() << std::endl;
    assert (almost_equal(quat.w(),0.707, 0.01));
    assert (almost_equal(quat.x(),0.0, 0.01));
    assert (almost_equal(quat.y(),0.0, 0.01));
    assert (almost_equal(quat.z(),0.707, 0.01));


    rotMat = Utils::euler_to_rotMat(Eigen::Vector3d(0.6370,0.7920,0.8037));
    assert (almost_equal(rotMat(0,0), 0.4875, 0.01));
    assert (almost_equal(rotMat(0,1), -0.2849, 0.01));
    assert (almost_equal(rotMat(0,2), 0.8253, 0.01));
    assert (almost_equal(rotMat(1,0), 0.5057, 0.01));
    assert (almost_equal(rotMat(1,1), 0.8627, 0.01));
    assert (almost_equal(rotMat(1,2), -0.0009, 0.01));
    assert (almost_equal(rotMat(2,0), -0.7118, 0.01));
    assert (almost_equal(rotMat(2,1), 0.4178, 0.01));
    assert (almost_equal(rotMat(2,2), 0.5646, 0.01));

    std::cout << rotMat << std::endl;
    Eigen::Vector3d euler_angles_2 = Utils::rotMat_to_euler(rotMat);
    assert (almost_equal(euler_angles_2[0], 0.6370, 0.01));
    assert (almost_equal(euler_angles_2[1], 0.7920, 0.01));
    assert (almost_equal(euler_angles_2[2], 0.8037, 0.01));


    return 0;
}