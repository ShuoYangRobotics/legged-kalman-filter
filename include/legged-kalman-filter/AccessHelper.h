#include <Eigen/Core>

/*
%   - position        (0:2)
%   - velocity        (3:5)
%   - euler angle     (6:8)
%   - foot1 pos       (9:11)
%   - foot1 vel       (12:14)
%   - foot2 pos       (15:17)
%   - foot2 vel       (18:20)
%   - foot3 pos       (21:23)
%   - foot3 vel       (24:26)
%   - foot4 pos       (27:29)
%   - foot4 vel       (30:32)

% control u
%   - w      (0:2)     body IMU angular velocity
%   - a      (3:5)     body IMU acceleration
%   - a1     (6:8)     foot 1 IMU acceleration (already in body frame)
%   - a2     (9:11)    foot 2 IMU acceleration (already in body frame)
%   - a3     (12:14)   foot 3 IMU acceleration (already in body frame)
%   - a4     (15:17)   foot 4 IMU acceleration (already in body frame)
*/

enum FootIndex {
  FOOT_FL = 0,  // Front left
  FOOT_FR = 1,  // Front right
  FOOT_RL = 2,  // Rear left
  FOOT_RR = 3   // Rear right
};

namespace {
template <typename Derived, size_t Size>
using ConstVectorBlockReturnType = const Eigen::VectorBlock<const Derived, Size>;

template <typename Derived, size_t Size>
using VectorBlockReturnType = Eigen::VectorBlock<Derived, Size>;
}  // namespace

// State
template <typename Derived, size_t Size = 3>
VectorBlockReturnType<Derived, Size> getBodyPosition(Eigen::MatrixBase<Derived>& state) {
  return VectorBlockReturnType<Derived, Size>(state.derived(), 0);
}

template <typename Derived, size_t Size = 3>
VectorBlockReturnType<Derived, Size> getBodyVelocity(Eigen::MatrixBase<Derived>& state) {
  return VectorBlockReturnType<Derived, Size>(state.derived(), 3);
}

template <typename Derived, size_t Size = 3>
VectorBlockReturnType<Derived, Size> getBodyEulerAngle(Eigen::MatrixBase<Derived>& state) {
  return VectorBlockReturnType<Derived, Size>(state.derived(), 6);
}

template <typename Derived, size_t Size = 3>
VectorBlockReturnType<Derived, Size> getFootPosition(Eigen::MatrixBase<Derived>& state, int foot_id) {
  return VectorBlockReturnType<Derived, Size>(state.derived(), 9 + foot_id * 6);
}

template <typename Derived, size_t Size = 3>
VectorBlockReturnType<Derived, Size> getFootVelocity(Eigen::MatrixBase<Derived>& state, int foot_id) {
  return VectorBlockReturnType<Derived, Size>(state.derived(), 12 + foot_id * 6);
}

// Const accessors
template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getBodyPosition(const Eigen::MatrixBase<Derived>& state) {
  return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 0);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getBodyVelocity(const Eigen::MatrixBase<Derived>& state) {
  return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 3);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getBodyEulerAngle(const Eigen::MatrixBase<Derived>& state) {
  return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 6);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getFootPosition(const Eigen::MatrixBase<Derived>& state, int foot_id) {
  return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 9 + foot_id * 6);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getFootVelocity(const Eigen::MatrixBase<Derived>& state, int foot_id) {
  return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 12 + foot_id * 6);
}

// Input
template <typename Derived, size_t Size = 3>
VectorBlockReturnType<Derived, Size> getBodyAngularVelocity(Eigen::MatrixBase<Derived>& control) {
  return VectorBlockReturnType<Derived, Size>(control.derived(), 0);
}

template <typename Derived, size_t Size = 3>
VectorBlockReturnType<Derived, Size> getBodyAcceleration(Eigen::MatrixBase<Derived>& control) {
  return VectorBlockReturnType<Derived, Size>(control.derived(), 3);
}

template <typename Derived, size_t Size = 3>
VectorBlockReturnType<Derived, Size> getFootAcceleration(Eigen::MatrixBase<Derived>& control, int foot_id) {
  return VectorBlockReturnType<Derived, Size>(control.derived(), 6 + foot_id * 3);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getBodyAngularVelocity(const Eigen::MatrixBase<Derived>& control) {
  return ConstVectorBlockReturnType<Derived, control>(control.derived(), 0);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getBodyAcceleration(const Eigen::MatrixBase<Derived>& control) {
  return ConstVectorBlockReturnType<Derived, Size>(control.derived(), 3);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getFootAcceleration(const Eigen::MatrixBase<Derived>& control, int foot_id) {
  return ConstVectorBlockReturnType<Derived, Size>(control.derived(), 6 + foot_id * 3);
}
