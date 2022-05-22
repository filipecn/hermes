/// Copyright (c) 2022, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file quaternion.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-02-09
///
///\brief Geometric quaternion class
///
///\ingroup geometry
///\addtogroup geometry
/// @{


#ifndef HERMES_HERMES_GEOMETRY_QUATERNION_H
#define HERMES_HERMES_GEOMETRY_QUATERNION_H

#include <hermes/geometry/transform.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                         Quaternion
// *********************************************************************************************************************
/// \brief Quaternion representation v.x i + v.y j + v.z k + r
/// \tparam T data type
template<typename T>
class Quaternion {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, Interval<f32>>::value || std::is_same<T, Interval<f64>>::value,
                "Quaternion must hold a float type!");
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  /// \brief Creates an identity quaternion (0,0,0,1)
  /// \return
  HERMES_DEVICE_CALLABLE static Quaternion<T> I() {
    return {{}, 1};
  }
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Default constructor
  HERMES_DEVICE_CALLABLE Quaternion() {}
  /// \brief Construct from component values
  /// \param x
  /// \param y
  /// \param z
  /// \param w
  HERMES_DEVICE_CALLABLE Quaternion(T x, T y, T z, T w) : v(x, y, z), r(w) {}
  /// \brief Construct from part values
  /// \param v vector part
  /// \param r scalar part
  HERMES_DEVICE_CALLABLE Quaternion(const hermes::Vector3<T> &v, T r = 0) : v(v), r(r) {}
  ///
  HERMES_DEVICE_CALLABLE ~Quaternion() {}
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  //                                                                                                           access
  /// \brief Get i-th component
  /// \warning `i` is not checked
  /// \param i component index in [0, 1]
  /// \return
  HERMES_DEVICE_CALLABLE const T &operator[](int i) const { return (&v[0])[i]; }
  /// \brief Get i-th component reference
  /// \warning `i` is not checked
  /// \param i component index in [0, 1]
  /// \return
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&v[0])[i]; }
  //                                                                                                       arithmetic
  //                                                                                                          boolean
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief
  /// \return
  HERMES_DEVICE_CALLABLE Matrix4x4<T> matrix() const {
    Matrix4x4<T> m;
    float Nv = v.x * v.x + v.y * v.y + v.z * v.z + r * r;
    float s = (Nv > 0.f) ? (2.f / Nv) : 0.f;
    float xs = v.x * s, ys = v.y * s, zs = v.z * s;
    float wx = r * xs, wy = r * ys, wz = r * zs;
    float xx = v.x * xs, xy = v.x * ys, xz = v.x * zs;
    float yy = v.y * ys, yz = v.y * zs, zz = v.z * zs;
    m[0][0] = 1.f - (yy + zz);
    m[1][0] = xy + wz;
    m[2][0] = xz - wy;
    m[0][1] = xy - wz;
    m[1][1] = 1.f - (xx + zz);
    m[2][1] = yz + wx;
    m[0][2] = xz + wy;
    m[1][2] = yz - wx;
    m[2][2] = 1.f - (xx + yy);
    m[0][3] = m[1][3] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
    m[3][3] = 1.f;
    return m;
  }
  /// \brief Computes inverse of this quaternion
  /// \return
  HERMES_DEVICE_CALLABLE Quaternion inverse() const {
    T d = 1 / (r * r + v.length2());
    return {-v * d, r * d};
  }
  /// \brief Computes conjugate of this quaternion
  /// \return
  HERMES_DEVICE_CALLABLE Quaternion conjugate() const {
    return {-v, r};
  }
  /// \brief Computes the squared norm
  /// \return
  HERMES_DEVICE_CALLABLE T length2() const {
    return v.length2() + r * r;
  }
  /// \brief Computes the norm
  /// \return
  HERMES_DEVICE_CALLABLE T length() const {
    return sqrtf(v.length2() + r * r);
  }
  /// \brief Computes normalized copy
  /// \return
  HERMES_DEVICE_CALLABLE Quaternion normalized() const {
    auto d = v.length2() + r * r;
    HERMES_CHECK_EXP(d != 0.0);
    return *this / d;
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  hermes::Vector3<T> v;  //!< vector part
  T r{0};                //!< scalar part
};

// *********************************************************************************************************************
//                                                                                                         ARITHMETIC
// *********************************************************************************************************************
/// \brief Scalar division
/// \tparam T
/// \param q
/// \param s
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE inline Quaternion<T> operator/(const Quaternion<T> &q, T s) {
  return {q.v / s, q.r / s};
}
/// \brief Scalar multiplication
/// \tparam T
/// \param q
/// \param s
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE inline Quaternion<T> operator*(const Quaternion<T> &q, T s) {
  return {q.v * s, q.r * s};
}
/// \brief Scalar multiplication
/// \tparam T
/// \param s
/// \param q
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE inline Quaternion<T> operator*(T s, const Quaternion<T> &q) {
  return {q.v * s, q.r * s};
}
/// \brief Adds two quaternions
/// \note q.v + p.v, q.r + p.r
/// \tparam T
/// \param q
/// \param p
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE inline Quaternion<T> operator+(const Quaternion<T> &q, const Quaternion<T> &p) {
  return {q.v + p.v, q.r + p.r};
}
/// \brief Subtracts p from q
/// \note q.v - p.v, q.r - p.r
/// \tparam T
/// \param q
/// \param p
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE inline Quaternion<T> operator-(const Quaternion<T> &q, const Quaternion<T> &p) {
  return {q.v - p.v, q.r - p.r};
}
/// \brief Multiplies quaternions
/// \tparam T
/// \param q
/// \param p
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE inline Quaternion<T> operator*(const Quaternion<T> &q, const Quaternion<T> &p) {
  return {q.r * p.v + p.r * q.v + cross(p.v, q.v), q.r * p.r - dot(q, p)};
}

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
/// \brief Quaternion support for `std::ostream::<<` operator
/// \tparam T
/// \param os
/// \param v
/// \return
template<typename T>
std::ostream &operator<<(std::ostream &os, const Quaternion<T> &v) {
  os << "Quaternion [(" << v[0] << " " << v[1] << " " << v[2] << "), " << v.w << "]";
  return os;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using quat = Quaternion<real_t>;
using quatf = Quaternion<f32>;
using quatd = Quaternion<f64>;

}

#endif //HERMES_HERMES_GEOMETRY_QUATERNION_H

/// @}
