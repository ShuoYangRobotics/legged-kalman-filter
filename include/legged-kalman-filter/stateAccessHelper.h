#include <Eigen/Core>

// %   - position        (0:2)
// %   - velocity        (3:5)
// %   - euler angle     (6:8)
// %   - foot1 pos       (9:11)
// %   - foot1 vel       (12:14)
// %   - foot2 pos       (15:17)
// %   - foot2 vel       (18:20)
// %   - foot3 pos       (21:23)
// %   - foot3 vel       (24:26)
// %   - foot4 pos       (27:29)
// %   - foot4 vel       (30:32)

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
ConstVectorBlockReturnType<Derived, Size> getBodyPosition(Eigen::MatrixBase<Derived>& state) {
    return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 0);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getBodyVelocity(Eigen::MatrixBase<Derived>& state) {
    return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 3);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getBodyEulerAngle(Eigen::MatrixBase<Derived>& state) {
    return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 6);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getFootPosition(Eigen::MatrixBase<Derived>& state, int foot_id) {
    return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 9 + foot_id * 6);
}

template <typename Derived, size_t Size = 3>
ConstVectorBlockReturnType<Derived, Size> getFootVelocity(Eigen::MatrixBase<Derived>& state, int foot_id) {
    return ConstVectorBlockReturnType<Derived, Size>(state.derived(), 12 + foot_id * 6);
}
