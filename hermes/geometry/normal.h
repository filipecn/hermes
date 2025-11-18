/* Copyright (c) 2017, FilipeCN.
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/// \file   normal.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2017-08-19
///  Geometric normal classes

#pragma once

#include <hermes/geometry/vector.h>

namespace hermes::geo {

// *****************************************************************************
//                                                                    Normal2
// *****************************************************************************
/// \brief Geometric 2-dimensional normal (nx, ny)
/// \tparam T
template <typename T> class Normal2 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Normal2 must hold an float type!");

public:
  /// \brief Default constructor
  HERMES_CPU_GPU Normal2() : x{0}, y{0} {};
  /// \brief Constructs from component values
  /// \param _x
  /// \param _y
  HERMES_CPU_GPU Normal2(T _x, T _y) : x(_x), y(_y) {}
  /// \brief Constructs from vector
  /// \param v
  HERMES_CPU_GPU Normal2(const Vector2<T> &v) : x(v.x), y(v.y) {}
  /// \brief Casts to vector
  /// \return
  HERMES_CPU_GPU explicit operator Vector2<T>() const {
    return Vector2<T>(x, y);
  }
  HERMES_CPU_GPU Normal2 operator-() const { return Normal2(-x, -y); }
  HERMES_CPU_GPU Normal2 &operator*=(T f) {
    x *= f;
    y *= f;
    return *this;
  }
  HERMES_CPU_GPU Normal2 &operator/=(T f) {
    x /= f;
    y /= f;
    return *this;
  }

  T x{0}; //!< 0-th normal component
  T y{0}; //!< 1-th normal component
};

// *****************************************************************************
//                                                                    Normal3
// *****************************************************************************
/// \brief Geometric 3-dimensional normal (nx, ny, nz)
/// \tparam T
template <typename T> class Normal3 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "Normal3 must hold an float type!");

public:
  /// \brief Default constructor
  HERMES_CPU_GPU Normal3() { x = y = z = 0; }
  /// \brief Constructs from component values
  /// \param _x
  /// \param _y
  /// \param _z
  HERMES_CPU_GPU Normal3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  /// \brief Constructs from vector
  /// \param v
  HERMES_CPU_GPU explicit Normal3(const Vector3<T> &v)
      : x(v.x), y(v.y), z(v.z) {}

  /// \brief Casts to vector
  /// \return
  HERMES_CPU_GPU explicit operator Vector3<T>() const {
    return Vector3<T>(x, y, z);
  }
  //                                                                                                       arithmetic
  HERMES_CPU_GPU Normal3 operator-() const { return Normal3(-x, -y, -z); }
  HERMES_CPU_GPU Normal3 &operator+=(const Vector3<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  HERMES_CPU_GPU Normal3 &operator*=(T f) {
    x *= f;
    y *= f;
    z *= f;
    return *this;
  }
  HERMES_CPU_GPU Normal3 &operator/=(T f) {
    x /= f;
    y /= f;
    z /= f;
    return *this;
  }

  HERMES_CPU_GPU bool operator!=(const Normal3 &n) const {
    return n.x != x || n.y != y || n.z != z;
  }

  /// \brief  reflects **v** from this
  /// \param v vector to be reflected
  /// \returns reflected **v**
  HERMES_CPU_GPU Vector3<T> reflect(const Vector3<T> &v) {
    return reflect(v, *this);
  }
  /// \brief projects **v** on the surface with this normal
  /// \param v vector
  /// \returns projected **v**
  HERMES_CPU_GPU Vector3<T> project(const Vector3<T> &v) {
    return project(v, *this);
  }
  /// \brief compute the two orthogonal-tangential vectors from this
  /// \param a **[out]** first tangent
  /// \param b **[out]** second tangent
  HERMES_CPU_GPU void tangential(Vector3<T> &a, Vector3<T> &b) {
    //  hermes::tangential(Vector3<T>(x, y, z), a, b);
  }

  T x{0}; //!< 0-th normal component
  T y{0}; //!< 1-th normal component
  T z{0}; //!< 2-th normal component
};

// *****************************************************************************
//                                                         EXTERNAL FUNCTIONS
// *****************************************************************************
/// \brief  reflects **a** on **n**
/// \param a vector to be reflected
/// \param n axis of reflection
/// \returns reflected **a**
template <typename T>
HERMES_CPU_GPU Vector2<T> reflect(const Vector2<T> &a, const Normal2<T> &n) {
  return a - 2 * dot(a, Vector2<T>(n)) * Vector2<T>(n);
}
/// \brief projects **v** on the surface with normal **n**
/// \param v vector
/// \param n surface's normal
/// \returns projected **v**
template <typename T>
HERMES_CPU_GPU Vector2<T> project(const Vector2<T> &v, const Normal2<T> &n) {
  return v - dot(v, Vector2<T>(n)) * Vector2<T>(n);
}
/// \brief Computes normalized copy
/// \tparam T
/// \param normal
/// \return
template <typename T>
HERMES_CPU_GPU Normal3<T> normalize(const Normal3<T> &normal) {
  T d = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
  if (d == 0.f)
    return normal;
  return Normal3<T>(normal.x / d, normal.y / d, normal.z / d);
}
/// \brief Computes absolute normal components
/// \tparam T
/// \param normal
/// \return
template <typename T> HERMES_CPU_GPU Normal3<T> abs(const Normal3<T> &normal) {
  return Normal3<T>(std::abs(normal.x), std::abs(normal.y), std::abs(normal.z));
}
/// \brief reflects **a** on **n**
/// \param a vector to be reflected
/// \param n axis of reflection
/// \returns reflected **a**
template <typename T>
HERMES_CPU_GPU Vector3<T> reflect(const Vector3<T> &a, const Normal3<T> &n) {
  return a - 2 * dot(a, Vector3<T>(n)) * Vector3<T>(n);
}
/// \brief projects **v** on the surface with normal **n**
/// \param v vector
/// \param n surface's normal
/// \returns projected **v**
template <typename S>
HERMES_CPU_GPU Vector3<S> project(const Vector3<S> &v, const Normal3<S> &n) {
  return v - dot(v, Vector3<S>(n)) * Vector3<S>(n);
}
/// \brief Computes dot product with vector
/// \tparam T
/// \param n
/// \param v
/// \return
template <typename T>
HERMES_CPU_GPU T dot(const Normal3<T> &n, const Vector3<T> &v) {
  return n.x * v.x + n.y * v.y + n.z * v.z;
}
/// \brief Computes dot product with vector
/// \tparam T
/// \param v
/// \param n
/// \return
template <typename T>
HERMES_CPU_GPU T dot(const Vector3<T> &v, const Normal3<T> &n) {
  return n.x * v.x + n.y * v.y + n.z * v.z;
}
///
/// \tparam T
/// \param v
/// \param n
/// \return v if is oriented along with n, -v otherwise
template <typename T>
HERMES_CPU_GPU Vector3<T> faceForward(const Vector3<T> &v,
                                      const Normal3<T> &n) {
  return (dot(v, n) < 0.f) ? -v : v;
}

using normal2 = Normal2<real_t>;
using normal2f = Normal2<f32>;
using normal2d = Normal2<f64>;
using normal3 = Normal3<real_t>;
using normal3f = Normal3<f32>;
using normal3d = Normal3<f64>;

} // namespace hermes::geo

namespace hermes {

HERMES_TYPE_LAYOUT_METHODS(geo::normal2, f32, 2)
HERMES_TYPE_LAYOUT_METHODS(geo::normal2d, f64, 2)
HERMES_TYPE_LAYOUT_METHODS(geo::normal3, f32, 3)
HERMES_TYPE_LAYOUT_METHODS(geo::normal3d, f64, 3)

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(geo::Normal2<T>, typename T)
HERMES_PUSH_DEBUG_LINE("N[{}, {}]", hermes::to_string(object.x),
                       hermes::to_string(object.y));
HERMES_TO_STRING_DEBUG_METHOD_END

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(geo::Normal3<T>, typename T)
HERMES_PUSH_DEBUG_LINE("N[{}, {}, {}]", hermes::to_string(object.x),
                       hermes::to_string(object.y),
                       hermes::to_string(object.z));
HERMES_TO_STRING_DEBUG_METHOD_END

} // namespace hermes
