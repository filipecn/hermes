/// Copyright (c) 2017, FilipeCN.
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
///\file normal.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2017-08-18
///
///\brief Geometric normal classes
///
///\ingroup geometry
///\addtogroup geometry
/// @{

#ifndef HERMES_GEOMETRY_NORMAL_H
#define HERMES_GEOMETRY_NORMAL_H

#include <iostream>
#include <hermes/geometry/vector.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                            Normal2
// *********************************************************************************************************************
/// \brief Geometric 2-dimensional normal (nx, ny)
/// \tparam T
template<typename T> class Normal2 : public MathElement<T, 2u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Normal2 must hold an float type!");
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Default constructor
  HERMES_DEVICE_CALLABLE Normal2() : x{0}, y{0} {};
  /// \brief Constructs from component values
  /// \param _x
  /// \param _y
  HERMES_DEVICE_CALLABLE Normal2(T _x, T _y) : x(_x), y(_y) {}
  /// \brief Constructs from vector
  /// \param v
  HERMES_DEVICE_CALLABLE Normal2(const Vector2 <T> &v) : x(v.x), y(v.y) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
  /// \brief Casts to vector
  /// \return
  HERMES_DEVICE_CALLABLE explicit operator Vector2<T>() const {
    return Vector2<T>(x, y);
  }
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE Normal2 operator-() const { return Normal2(-x, -y); }
  HERMES_DEVICE_CALLABLE Normal2 &operator*=(T f) {
    x *= f;
    y *= f;
    return *this;
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T x{0}; //!< 0-th normal component
  T y{0}; //!< 1-th normal component
};

// *********************************************************************************************************************
//                                                                                                            Normal3
// *********************************************************************************************************************
/// \brief Geometric 3-dimensional normal (nx, ny, nz)
/// \tparam T
template<typename T> class Normal3 : MathElement<T, 3u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Normal3 must hold an float type!");
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Default constructor
  HERMES_DEVICE_CALLABLE Normal3() { x = y = z = 0; }
  /// \brief Constructs from component values
  /// \param _x
  /// \param _y
  /// \param _z
  HERMES_DEVICE_CALLABLE Normal3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  /// \brief Constructs from vector
  /// \param v
  HERMES_DEVICE_CALLABLE explicit Normal3(const Vector3 <T> &v) : x(v.x), y(v.y), z(v.z) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
  /// \brief Casts to vector
  /// \return
  HERMES_DEVICE_CALLABLE explicit operator Vector3<T>() const {
    return Vector3<T>(x, y, z);
  }
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE Normal3 operator-() const { return Normal3(-x, -y, -z); }
  HERMES_DEVICE_CALLABLE Normal3 &operator*=(T f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }
  //                                                                                                          boolean
  HERMES_DEVICE_CALLABLE bool operator!=(const Normal3 &n) const {
    return n.x != x || n.y != y || n.z != z;
  }
  // *******************************************************************************************************************
  //                                                                                                         GEOMETRY
  // *******************************************************************************************************************
  /// \brief  reflects **v** from this
  /// \param v vector to be reflected
  /// \returns reflected **v**
  HERMES_DEVICE_CALLABLE Vector3 <T> reflect(const Vector3 <T> &v) {
    return reflect(v, *this);
  }
  /// \brief projects **v** on the surface with this normal
  /// \param v vector
  /// \returns projected **v**
  HERMES_DEVICE_CALLABLE Vector3 <T> project(const Vector3 <T> &v) {
    return project(v, *this);
  }
  /// \brief compute the two orthogonal-tangential vectors from this
  /// \param a **[out]** first tangent
  /// \param b **[out]** second tangent
  HERMES_DEVICE_CALLABLE void tangential(Vector3 <T> &a, Vector3 <T> &b) {
    //  hermes::tangential(Vector3<T>(x, y, z), a, b);
  }

  T x{0};  //!< 0-th normal component
  T y{0};  //!< 1-th normal component
  T z{0};  //!< 2-th normal component
};

// *********************************************************************************************************************
//                                                                                                          FUNCTIONS
// *********************************************************************************************************************
/// \brief  reflects **a** on **n**
/// \param a vector to be reflected
/// \param n axis of reflection
/// \returns reflected **a**
template<typename T>
HERMES_DEVICE_CALLABLE Vector2 <T> reflect(const Vector2 <T> &a, const Normal2<T> &n) {
  return a - 2 * dot(a, Vector2<T>(n)) * Vector2<T>(n);
}
/// \brief projects **v** on the surface with normal **n**
/// \param v vector
/// \param n surface's normal
/// \returns projected **v**
template<typename T>
HERMES_DEVICE_CALLABLE Vector2 <T> project(const Vector2 <T> &v, const Normal2<T> &n) {
  return v - dot(v, Vector2<T>(n)) * Vector2<T>(n);
}
/// \brief Computes normalized copy
/// \tparam T
/// \param normal
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE  Normal3<T> normalize(const Normal3<T> &normal) {
  T d = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
  if (d == 0.f)
    return normal;
  return Normal3<T>(normal.x / d, normal.y / d, normal.z / d);
}
/// \brief Computes absolute normal components
/// \tparam T
/// \param normal
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE  Normal3<T> abs(const Normal3<T> &normal) {
  return Normal3<T>(std::abs(normal.x), std::abs(normal.y), std::abs(normal.z));
}
/// \brief reflects **a** on **n**
/// \param a vector to be reflected
/// \param n axis of reflection
/// \returns reflected **a**
template<typename T>
HERMES_DEVICE_CALLABLE  Vector3 <T> reflect(const Vector3 <T> &a, const Normal3<T> &n) {
  return a - 2 * dot(a, Vector3<T>(n)) * Vector3<T>(n);
}
/// \brief projects **v** on the surface with normal **n**
/// \param v vector
/// \param n surface's normal
/// \returns projected **v**
template<typename S>
HERMES_DEVICE_CALLABLE  Vector3 <S> project(const Vector3 <S> &v, const Normal3<S> &n) {
  return v - dot(v, Vector3<S>(n)) * Vector3<S>(n);
}
/// \brief Computes dot product with vector
/// \tparam T
/// \param n
/// \param v
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE  T dot(const Normal3<T> &n, const Vector3 <T> &v) {
  return n.x * v.x + n.y * v.y + n.z * v.z;
}
/// \brief Computes dot product with vector
/// \tparam T
/// \param v
/// \param n
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE  T dot(const Vector3 <T> &v, const Normal3<T> &n) {
  return n.x * v.x + n.y * v.y + n.z * v.z;
}
///
/// \tparam T
/// \param v
/// \param n
/// \return v if is oriented along with n, -v otherwise
template<typename T>
HERMES_DEVICE_CALLABLE Vector3 <T> faceForward(const Vector3 <T> &v, const Normal3<T> &n) {
  return (dot(v, n) < 0.f) ? -v : v;
}

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
/// \brief Normal's support for `std::ostream::<<` operator
/// \tparam T
/// \param os
/// \param n
/// \return
template<typename T>
std::ostream &operator<<(std::ostream &os, const Normal2<T> &n) {
  os << "[Normal3] " << n.x << " " << n.y << std::endl;
  return os;
}
/// \brief Normal's support for `std::ostream::<<` operator
/// \tparam T
/// \param os
/// \param n
/// \return
template<typename T>
std::ostream &operator<<(std::ostream &os, const Normal3<T> &n) {
  os << "[Normal3] " << n.x << " " << n.y << " " << n.z << std::endl;
  return os;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using normal2 = Normal2<real_t>;
using normal2f = Normal2<float>;
using normal2d = Normal2<double>;
using normal3 = Normal3<real_t>;
using normal3f = Normal3<float>;
using normal3d = Normal3<double>;

} // namespace hermes

#endif

/// @}
