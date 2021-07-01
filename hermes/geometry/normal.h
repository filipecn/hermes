/*
 * Copyright (c) 2017 FilipeCN
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef HERMES_GEOMETRY_NORMAL_H
#define HERMES_GEOMETRY_NORMAL_H

#include <iostream>
#include <hermes/numeric/math_element.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                            Normal2
// *********************************************************************************************************************
template<typename T> class Vector2;
template<typename T> class Normal2 : public MathElement<T, 2u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Normal2 must hold an float type!");
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  /// \brief  reflects **a** on **n**
  /// \param a vector to be reflected
  /// \param n axis of reflection
  /// \returns reflected **a**
  HERMES_DEVICE_CALLABLE friend Vector2<T> reflect(const Vector2<T> &a, const Normal2<T> &n) {
    return a - 2 * dot(a, Vector2<T>(n)) * Vector2<T>(n);
  }
  /// \brief projects **v** on the surface with normal **n**
  /// \param v vector
  /// \param n surface's normal
  /// \returns projected **v**
  HERMES_DEVICE_CALLABLE friend Vector2<T> project(const Vector2<T> &v, const Normal2<T> &n) {
    return v - dot(v, Vector2<T>(n)) * Vector2<T>(n);
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Normal2() : x{0}, y{0} {};
  /// \param _x
  /// \param _y
  HERMES_DEVICE_CALLABLE Normal2(T _x, T _y) : x(_x), y(_y) {}
  HERMES_DEVICE_CALLABLE Normal2(const Vector2<T> &v) : x(v.x), y(v.y) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
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
  T x{0}, y{0};
};

// *********************************************************************************************************************
//                                                                                                            Normal3
// *********************************************************************************************************************
template<typename T> class Vector3;
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
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE friend Normal3<T> normalize(const Normal3<T> &normal) {
    T d = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
    if (d == 0.f)
      return normal;
    return Normal3<T>(normal.x / d, normal.y / d, normal.z / d);
  }
  HERMES_DEVICE_CALLABLE friend Normal3<T> abs(const Normal3<T> &normal) {
    return Normal3<T>(std::abs(normal.x), std::abs(normal.y), std::abs(normal.z));
  }
  /// \brief reflects **a** on **n**
  /// \param a vector to be reflected
  /// \param n axis of reflection
  /// \returns reflected **a**
  HERMES_DEVICE_CALLABLE friend Vector3<T> reflect(const Vector3<T> &a, const Normal3<T> &n) {
    return a - 2 * dot(a, Vector3<T>(n)) * Vector3<T>(n);
  }
  /// \brief projects **v** on the surface with normal **n**
  /// \param v vector
  /// \param n surface's normal
  /// \returns projected **v**
  HERMES_DEVICE_CALLABLE friend Vector3<T> project(const Vector3<T> &v, const Normal3<T> &n) {
    return v - dot(v, Vector3<T>(n)) * Vector3<T>(n);
  }
  HERMES_DEVICE_CALLABLE friend T dot(const Normal3<T> &n, const Vector3<T> &v) {
    return n.x * v.x + n.y * v.y + n.z * v.z;
  }
  HERMES_DEVICE_CALLABLE friend T dot(const Vector3<T> &v, const Normal3<T> &n) {
    return n.x * v.x + n.y * v.y + n.z * v.z;
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Normal3() { x = y = z = 0; }
  HERMES_DEVICE_CALLABLE Normal3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  HERMES_DEVICE_CALLABLE explicit Normal3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
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
  HERMES_DEVICE_CALLABLE Vector3<T> reflect(const Vector3<T> &v) {
    return reflect(v, *this);
  }
  /// \brief projects **v** on the surface with this normal
  /// \param v vector
  /// \returns projected **v**
  HERMES_DEVICE_CALLABLE Vector3<T> project(const Vector3<T> &v) {
    return project(v, *this);
  }
  /// \brief compute the two orthogonal-tangential vectors from this
  /// \param a **[out]** first tangent
  /// \param b **[out]** second tangent
  HERMES_DEVICE_CALLABLE void tangential(Vector3<T> &a, Vector3<T> &b) {
    //  hermes::tangential(Vector3<T>(x, y, z), a, b);
  }

  T x{0}, y{0}, z{0};
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
std::ostream &operator<<(std::ostream &os, const Normal2<T> &n) {
  os << "[Normal3] " << n.x << " " << n.y << std::endl;
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Normal3<T> &n) {
  os << "[Normal3] " << n.x << " " << n.y << " " << n.z << std::endl;
  return os;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
typedef Normal2<real_t> normal2;
typedef Normal3<real_t> normal3;
typedef Normal3<float> normal3f;

} // namespace hermes

#endif
