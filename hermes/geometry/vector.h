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

/// \file   vector.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2017-08-19
///  Geometric vector classes

#pragma once

#include <hermes/base/index.h>
#include <hermes/numeric/interval.h>

#include <cstring>

namespace hermes::geo {

/// Forward declaration of Point2
/// \tparam T
template <typename T> class Point2;
// *****************************************************************************
//                                                                    Vector2
// *****************************************************************************
/// Geometric 2-dimensional vector (x, y)
/// \tparam T
template <typename T> class Vector2 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, Interval<f32>>::value ||
                    std::is_same<T, Interval<f64>>::value,
                "Vector2 must hold an float type!");

public:
  /// Default constructor
  HERMES_CPU_GPU Vector2() : x{0}, y{0} {};
  /// Constructs from component values
  /// \param _x
  /// \param _y
  HERMES_CPU_GPU Vector2(T _x, T _y) : x(_x), y(_y) {}
  /// Constructs from geometric point
  /// \param p
  HERMES_CPU_GPU explicit Vector2(const Point2<T> &p) : x(p.x), y(p.y) {}
  /// Constructs from single component value
  /// \param f
  HERMES_CPU_GPU explicit Vector2(T f) { x = y = f; }
  /// Constructs from component array
  /// \param f
  HERMES_CPU_GPU explicit Vector2(T *f) {
    x = f[0];
    y = f[1];
  }

  /// Get i-th component
  /// \warning `i` is not checked
  /// \param i component index in [0, 1]
  /// \return
  HERMES_CPU_GPU T operator[](size_t i) const { return (&x)[i]; }
  /// Get i-th component reference
  /// \warning `i` is not checked
  /// \param i component index in [0, 1]
  /// \return
  HERMES_CPU_GPU T &operator[](size_t i) { return (&x)[i]; }

#define ARITHMETIC_OP(OP)                                                      \
  HERMES_CPU_GPU Vector2 &operator OP## = (const Vector2 &v) {                 \
    x OP## = v.x;                                                              \
    y OP## = v.y;                                                              \
    return *this;                                                              \
  }                                                                            \
  HERMES_CPU_GPU Vector2 &operator OP## = (real_t f) {                         \
    x OP## = f;                                                                \
    y OP## = f;                                                                \
    return *this;                                                              \
  }                                                                            \
  HERMES_CPU_GPU Vector2 operator OP(const Vector2<T> &b) const {              \
    return {x OP b.x, y OP b.y};                                               \
  }                                                                            \
  HERMES_CPU_GPU Vector2 operator OP(real_t f) const {                         \
    return {x OP f, y OP f};                                                   \
  }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP
  HERMES_CPU_GPU Vector2 operator-() const { return Vector2(-x, -y); }

#define RELATIONAL_OP(OP, CO)                                                  \
  HERMES_CPU_GPU bool operator OP(const Vector2 &b) const {                    \
    return x OP b.x CO y OP b.y;                                               \
  }
  RELATIONAL_OP(<, &&)
  RELATIONAL_OP(>, &&)
  RELATIONAL_OP(<=, &&)
  RELATIONAL_OP(>=, &&)
  RELATIONAL_OP(!=, ||)
#undef RELATIONAL_OP
  HERMES_CPU_GPU bool operator==(const Vector2<T> &b) const {
    return numbers::cmp::is_equal(x, b.x) && numbers::cmp::is_equal(y, b.y);
  }

  /// Computes squared magnitude
  /// \return
  HERMES_CPU_GPU T length2() const { return x * x + y * y; }
  /// Computes magnitude
  /// \return
  HERMES_CPU_GPU T length() const { return sqrtf(length2()); }
  /// Gets orthogonal vector in _right_ direction
  /// \return
  HERMES_CPU_GPU Vector2 right() const { return Vector2(y, -x); }
  /// Gets orthogonal vector in _left_ direction
  /// \return
  HERMES_CPU_GPU Vector2 left() const { return Vector2(-y, x); }

  // swizzle

  /// Gets swizzle form (x, y)
  Vector2 xy() const { return {x, y}; }
  /// Gets swizzle form (y, x)
  Vector2 yx() const { return {y, x}; }
  /// Gets swizzle form (x, x)
  Vector2 xx() const { return {x, x}; }
  /// Gets swizzle form (y, y)
  Vector2 yy() const { return {y, y}; }

  T x = T(0.0); //!< 0-th component
  T y = T(0.0); //!< 1-th component
};

template <typename T> class Point3;

// *****************************************************************************
//                                                                    Vector3
// *****************************************************************************
/// Geometric 3-dimensional vector (x, y, z)
/// \tparam T
template <typename T> class Vector3 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, Interval<f32>>::value ||
                    std::is_same<T, Interval<f64>>::value,
                "Vector3 must hold an float type!");

public:
  /// Default constructor
  HERMES_CPU_GPU Vector3() : x{0}, y{0}, z{0} {}
  /// Constructs from single component value
  /// \param _f
  HERMES_CPU_GPU explicit Vector3(T _f) : x(_f), y(_f), z(_f) {}
  /// Constructs from component values
  /// \param _x
  /// \param _y
  /// \param _z
  HERMES_CPU_GPU Vector3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  /// Constructs from component array
  /// \param v
  HERMES_CPU_GPU explicit Vector3(const T *v) {
    x = v[0];
    y = v[1];
    z = v[2];
  }

  /// Casts from geometric point
  /// \param p
  HERMES_CPU_GPU explicit Vector3(const Point3<T> &p)
      : x(p.x), y(p.y), z(p.z) {}
  /// Constructs from interval
  /// \tparam S
  /// \tparam C
  /// \param vi
  template <typename S, typename C = T>
  HERMES_CPU_GPU explicit Vector3(
      const Vector3<Interval<S>> &vi,
      typename std::enable_if_t<!std::is_same_v<C, Interval<f32>> &&
                                !std::is_same_v<C, Interval<f64>>> * = nullptr)
      : x(vi.x), y(vi.y), z(vi.z) {}

  /// Copy assign
  /// \param v
  /// \return
  HERMES_CPU_GPU Vector3 &operator=(const T &v) {
    x = y = z = v;
    return *this;
  }

#define ARITHMETIC_OP(OP)                                                      \
  HERMES_CPU_GPU Vector3 &operator OP## = (const Vector3 &v) {                 \
    x OP## = v.x;                                                              \
    y OP## = v.y;                                                              \
    z OP## = v.z;                                                              \
    return *this;                                                              \
  }                                                                            \
  HERMES_CPU_GPU Vector3 &operator OP## = (real_t f) {                         \
    x OP## = f;                                                                \
    y OP## = f;                                                                \
    z OP## = f;                                                                \
    return *this;                                                              \
  }                                                                            \
  HERMES_CPU_GPU Vector3 operator OP(const Vector3<T> &b) const {              \
    return {x OP b.x, y OP b.y, z OP b.z};                                     \
  }                                                                            \
  HERMES_CPU_GPU Vector3 operator OP(T f) const {                              \
    return {x OP f, y OP f, z OP f};                                           \
  }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
#undef ARITHMETIC_OP
  HERMES_CPU_GPU Vector3<T> &operator/=(T f) {
    HERMES_CHECK_EXP(numbers::cmp::is_zero(f));
    T inv = 1.f / f;
    x *= inv;
    y *= inv;
    z *= inv;
    return *this;
  }
  HERMES_CPU_GPU Vector3<T> &operator/=(const Vector3<T> &v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
  }
  HERMES_CPU_GPU Vector3<T> operator/(const T &f) const {
    T inv = 1.f / f;
    return Vector3<T>(x * inv, y * inv, z * inv);
  }
  HERMES_CPU_GPU Vector3<T> operator-() const { return Vector3(-x, -y, -z); }

  HERMES_CPU_GPU bool operator==(const Vector3<T> &b) const {
    return numbers::cmp::is_equal(x, b.x) && numbers::cmp::is_equal(y, b.y) &&
           numbers::cmp::is_equal(z, b.z);
  }
  HERMES_CPU_GPU bool operator<(const Vector3<T> &b) const {
    if (x < b.x)
      return true;
    if (y < b.y)
      return true;
    return z < b.z;
  }
  HERMES_CPU_GPU bool operator>(const Vector3<T> &b) const {
    if (x > b.x)
      return true;
    if (y > b.y)
      return true;
    return z > b.z;
  }
  HERMES_CPU_GPU bool operator>=(const Vector3<T> &b) const {
    return x >= b.x && y >= b.y && z >= b.z;
  }
  HERMES_CPU_GPU bool operator<=(const Vector3<T> &b) const {
    return x <= b.x && y <= b.y && z <= b.z;
  }

  /// Get i-th component
  /// \warning `i` is not checked
  /// \param i component index in [0, 2]
  HERMES_CPU_GPU T operator[](int i) const { return (&x)[i]; }
  /// Get i-th component reference
  /// \warning `i` is not checked
  /// \param i component index in [0, 2]
  HERMES_CPU_GPU T &operator[](int i) { return (&x)[i]; }
  /// Gets 2-dimensional swizzle form
  /// \param i
  /// \param j
  HERMES_CPU_GPU Vector2<T> xy(int i = 0, int j = 1) const {
    return Vector2<T>((&x)[i], (&x)[j]);
  }

  /// Computes Manhattan distance
  /// \note Also called L1-norm, taxicab norm, Manhattan norm
  /// \note Defined as ||v||_1 = sum_i(|v_i|)
  /// \return L1-norm of this vector
  HERMES_CPU_GPU T mLength() const {
    return std::abs(x) + std::abs(y) + std::abs(z);
  }
  /// Computes vector magnitude
  /// \note Also called L2-norm, Euclidean norm, Euclidean distance, 2-norm
  /// \note Defined as ||v|| = (v_i * v_i)^(1/2)
  /// \return 2-norm of this vector
  template <typename C = T>
  HERMES_CPU_GPU T
  length(typename std::enable_if_t<!std::is_same_v<C, Interval<f32>> &&
                                   !std::is_same_v<C, Interval<f64>>> * =
             nullptr) const {
    return std::sqrt(length2());
  }
  /// Computes vector magnitude
  /// \tparam C
  /// \return
  template <typename C = T>
  HERMES_CPU_GPU T
  length(typename std::enable_if_t<std::is_same_v<C, Interval<f32>> ||
                                   std::is_same_v<C, Interval<f64>>> * =
             nullptr) const {
    return length2().sqrt();
  }
  /// Computes vector squared magnitude
  /// \note Also called squared Euclidean distance
  /// \note Defined as ||v||^2 = v_i * v_i
  /// \return squared 2-norm of this vector
  HERMES_CPU_GPU T length2() const { return x * x + y * y + z * z; }
  /// Gets maximum absolute component value
  /// \note Also called maximum norm, infinity norm
  /// \note Defined as ||v||_inf = argmax max(|v_i|)
  /// \return greatest absolute component value
  HERMES_CPU_GPU T maxAbs() const {
    if (std::abs(x) > std::abs(y) && std::abs(x) > std::abs(z))
      return x;
    if (std::abs(y) > std::abs(x) && std::abs(y) > std::abs(z))
      return y;
    return z;
  }
  /// Gets maximum component value
  /// \note Defined as argmax v_i
  /// \return greatest component value
  HERMES_CPU_GPU T max() const {
    if (x > y && x > z)
      return x;
    if (y > x && y > z)
      return y;
    return z;
  }
  /// Gets index of component with maximum value
  /// \note Defined as argmax_i v_i
  /// \return Index of component with greatest value
  HERMES_NODISCARD HERMES_CPU_GPU int maxDimension() const {
    if (x > y && x > z)
      return 0;
    if (y > x && y > z)
      return 1;
    return 2;
  }
  /// Gets index of component with maximum absolute value
  /// \note Defined as argmax_i |v_i|
  /// \return Index of dimension with greatest value
  HERMES_NODISCARD HERMES_CPU_GPU int maxAbsDimension() const {
    if (std::abs(x) > std::abs(y) && std::abs(x) > std::abs(z))
      return 0;
    if (std::abs(y) > std::abs(x) && std::abs(y) > std::abs(z))
      return 1;
    return 2;
  }

  /// Normalizes this vector
  /// \note Normalization by vector length
  /// \note Defined as v / ||v||
  HERMES_CPU_GPU void normalize() {
    auto l = length();
    if (l != 0.f) {
      x /= l;
      y /= l;
      z /= l;
    }
  }
  /// Gets a normalized copy of this vector
  /// \note Normalization by vector length
  /// \note Defined as v / ||v||
  /// \return Normalized vector of this vector
  HERMES_CPU_GPU Vector3 normalized() const {
    auto l = length();
    return (*this) / l;
  }
  /// Projects this vector onto b
  /// \note b * dot(v,b) / ||b||
  /// \param b vector to project onto
  /// \return projection of this vector onto **b**
  HERMES_CPU_GPU Vector3 projectOnto(const Vector3 &b) {
    return (dot(b, *this) / b.length2()) * b;
  }
  /// Rejects this vector on b
  /// \note v - b * dot(v,b) / ||b||
  /// \param b vector of rejection
  /// \return rejection of this vector on **b**
  HERMES_CPU_GPU Vector3 rejectOn(const Vector3 b) {
    return *this - (dot(b, *this) / b.length2()) * b;
  }

  /// Check for nans
  /// \return
  HERMES_CPU_GPU HERMES_NODISCARD bool hasNaNs() const {
    return numbers::is_nan(x) || numbers::is_nan(y) || numbers::is_nan(z);
  }

  T x = T(0.0); //!< 0-th component
  T y = T(0.0); //!< 1-th component
  T z = T(0.0); //!< 2-th component
};

// *****************************************************************************
//                                                                    Vector4
// *****************************************************************************
/// Geometric 4-dimensional point (x, y, z, w)
/// \tparam T
template <typename T> class Vector4 {
public:
  /// Default constructor
  HERMES_CPU_GPU Vector4() : x{0}, y{0}, z{0}, w{0} {}
  /// Construct from component values
  /// \param _x
  /// \param _y
  /// \param _z
  /// \param _w
  HERMES_CPU_GPU Vector4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}

  HERMES_CPU_GPU Vector4<T> &operator+=(const Vector4<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
  }
  HERMES_CPU_GPU Vector4<T> &operator-=(const Vector4<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
  }
  HERMES_CPU_GPU Vector4<T> &operator*=(T f) {
    x *= f;
    y *= f;
    z *= f;
    w *= f;
    return *this;
  }
  HERMES_CPU_GPU Vector4<T> &operator/=(T f) {
    T inv = 1.f / f;
    x *= inv;
    y *= inv;
    z *= inv;
    w *= inv;
    return *this;
  }
  HERMES_CPU_GPU Vector4<T> operator-() const {
    return Vector4(-x, -y, -z, -w);
  }
  HERMES_CPU_GPU Vector4<T> operator+(const Vector4<T> &b) {
    return Vector4<T>(x + b.x, y + b.y, z + b.z, w + b.w);
  }
  HERMES_CPU_GPU Vector4<T> operator-(const Vector4<T> &b) {
    return Vector4<T>(x - b.x, y - b.y, z - b.z, w - b.w);
  }
  HERMES_CPU_GPU Vector4<T> operator*(T f) {
    return Vector4<T>(x * f, y * f, z * f, w * f);
  }
  HERMES_CPU_GPU Vector4<T> operator/(T f) {
    T inv = 1.f / f;
    return Vector4<T>(x * inv, y * inv, z * inv, w * inv);
  }
  /// Get i-th component
  /// \warning `i` is not checked
  /// \param i component index in [0, 3]
  /// \return
  HERMES_CPU_GPU T operator[](int i) const { return (&x)[i]; }
  /// Get i-th component reference
  /// \warning `i` is not checked
  /// \param i component index in [0, 3]
  /// \return
  HERMES_CPU_GPU T &operator[](int i) { return (&x)[i]; }
  /// Gets first 2 components
  /// \return
  HERMES_CPU_GPU Vector2<T> xy() { return Vector2<T>(x, y); }
  /// Gets first 3 components
  /// \return
  HERMES_CPU_GPU Vector3<T> xyz() { return Vector3<T>(x, y, z); }
  /// Computes vector squared magnitude
  /// \return
  HERMES_CPU_GPU T length2() const { return x * x + y * y + z * z + w * w; }
  /// Computes vector magnitude
  /// \return
  HERMES_CPU_GPU T length() const { return sqrtf(length2()); }

  T x = T(0.0); //!< 0-th component
  T y = T(0.0); //!< 1-th component
  T z = T(0.0); //!< 2-th component
  T w = T(0.0); //!< 3-th component
};

// *****************************************************************************
//                                                         EXTERNAL FUNCTIONS
// *****************************************************************************
/// Computes the dot product between two vectors
/// \tparam T
/// \param a
/// \param b
/// \return
template <typename T>
HERMES_CPU_GPU T dot(const Vector2<T> &a, const Vector2<T> &b) {
  return a.x * b.x + a.y * b.y;
}
/// Computes the dot product between two vectors
/// \tparam T
/// \param a
/// \param b
/// \return
template <typename T>
HERMES_CPU_GPU T dot(const Vector3<T> &a, const Vector3<T> &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
/// Computes the cross product between two vectors
/// \tparam T
/// \param a
/// \param b
/// \return
template <typename T>
HERMES_CPU_GPU T cross(const Vector2<T> &a, const Vector2<T> &b) {
  return a.x * b.y - a.y * b.x;
}
/// Computes the cross product between two vectors
/// \tparam T
/// \param a
/// \param b
/// \return
template <typename T>
HERMES_CPU_GPU Vector3<T> cross(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>((a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z),
                    (a.x * b.y) - (a.y * b.x));
}
/// Computes the triple product between 3 vectors
/// \tparam T
/// \param a
/// \param b
/// \param c
/// \return
template <typename T>
HERMES_CPU_GPU T triple(const Vector3<T> &a, const Vector3<T> &b,
                        const Vector3<T> &c) {
  return dot(a, cross(b, c));
}
/// Computes normalized copy from vector
/// \tparam T
/// \param v
/// \return
template <typename T> HERMES_CPU_GPU Vector2<T> normalize(const Vector2<T> &v) {
  return v / v.length();
}
/// Computes normalized copy from vector
/// \tparam T
/// \param v
/// \return
template <typename T> HERMES_CPU_GPU Vector3<T> normalize(const Vector3<T> &v) {
  if (v.length2() == 0.f)
    return v;
  return v / v.length();
}
/// Computes vector orthonormal to given vector
/// \tparam T
/// \param v
/// \param first
/// \return
template <typename T>
HERMES_CPU_GPU Vector2<T> orthonormal(const Vector2<T> &v, bool first = true) {
  Vector2<T> n = normalize(v);
  if (first)
    return Vector2<T>(-n.y, n.x);
  return Vector2<T>(n.y, -n.x);
}
/// Projects a vector onto another.
/// \param a **[in]**
/// \param b **[in]**
/// \returns the projection of **a** onto **b**
template <typename T>
HERMES_CPU_GPU Vector2<T> project(const Vector2<T> &a, const Vector2<T> &b) {
  return (dot(b, a) / b.length2()) * b;
}
/// Projects one vector into another
/// \note b * dot(a,b) / ||b||
/// \tparam T
/// \param a
/// \param b
/// \return projection of **a** onto **b**
template <typename T>
HERMES_CPU_GPU Vector3<T> project(const Vector3<T> &a, const Vector3<T> &b) {
  return (dot(b, a) / b.length2()) * b;
}
/// Rejects one vector on another
/// \tparam T
/// \param a
/// \param b
/// \return rejection of **a** onto **b**
template <typename T>
HERMES_CPU_GPU Vector3<T> reject(const Vector3<T> &a, const Vector3<T> &b) {
  return a - (dot(b, a) / b.length2()) * b;
}
/// compute the two orthogonal-tangential vectors from a
/// \param a **[in]** normal
/// \param b **[out]** first tangent
/// \param c **[out]** second tangent
template <typename T>
HERMES_CPU_GPU void tangential(const Vector3<T> &a, Vector3<T> &b,
                               Vector3<T> &c) {
  b = hermes::geo::normalize(cross(
      a, ((std::abs(a.y) > 0.f || std::abs(a.z) > 0.f) ? Vector3<T>(1, 0, 0)
                                                       : Vector3<T>(0, 1, 1))));
  c = hermes::geo::normalize(cross(a, b));
}

#define DOP2(OP) f OP v.x, f OP v.y
#define DOP3(OP) f OP v.x, f OP v.y, f OP v.z
#define MATH_OP(D, OP)                                                         \
  template <typename T>                                                        \
  HERMES_CPU_GPU Vector##D<T> operator OP(T f, const Vector##D<T> &v) {        \
    return Vector##D<T>(DOP##D(OP));                                           \
  }
MATH_OP(2, *)
MATH_OP(2, /)
MATH_OP(3, *)
MATH_OP(3, /)
#undef MATH_OP
#undef DOP2
#undef DOP3

#define DOP2(OP) OP(a.x, b.x), OP(a.y, b.y)
#define DOP3(OP) OP(a.x, b.x), OP(a.y, b.y), OP(a.z, b.z)
#define MATH_OP(D, NAME, OP)                                                   \
  template <typename T>                                                        \
  HERMES_CPU_GPU Vector##D<T> NAME(const Vector##D<T> &a,                      \
                                   const Vector##D<T> &b) {                    \
    return Vector##D<T>(DOP##D(OP));                                           \
  }
#ifdef HERMES_DEVICE_ENABLED
MATH_OP(2, min, min)
MATH_OP(2, max, max)
MATH_OP(3, min, min)
MATH_OP(3, max, max)
#else
MATH_OP(2, min, (std::min))
MATH_OP(2, max, (std::max))
MATH_OP(3, min, (std::min))
MATH_OP(3, max, (std::max))
#endif
#undef MATH_OP
#undef DOP2
#undef DOP3

#define DOP2(OP) OP(v.x), OP(v.y)
#define DOP3(OP) OP(v.x), OP(v.y), OP(v.z)
#define MATH_OP(NAME, OP, D)                                                   \
  template <typename T>                                                        \
  HERMES_CPU_GPU Vector##D<T> NAME(const Vector##D<T> &v) {                    \
    return Vector##D<T>(DOP##D(OP));                                           \
  }
#ifdef HERMES_DEVICE_ENABLED
MATH_OP(floor, ::floor, 2)
MATH_OP(ceil, ::ceil, 2)
MATH_OP(abs, ::abs, 2)
MATH_OP(cos, ::cos, 2)
MATH_OP(floor, ::floor, 3)
MATH_OP(ceil, ::ceil, 3)
MATH_OP(abs, ::abs, 3)
MATH_OP(cos, ::cos, 3)
#else
MATH_OP(floor, std::floor, 2)
MATH_OP(ceil, std::ceil, 2)
MATH_OP(abs, std::abs, 2)
MATH_OP(cos, std::cos, 2)
MATH_OP(floor, std::floor, 3)
MATH_OP(ceil, std::ceil, 3)
MATH_OP(abs, std::abs, 3)
MATH_OP(cos, std::cos, 3)
#endif
#undef MATH_OP
#undef DOP2
#undef DOP3

using vec2 = Vector2<real_t>;
using vec3 = Vector3<real_t>;
using vec4 = Vector4<real_t>;
using vec3d = Vector3<f64>;
using vec3f = Vector3<f32>;
using vec2f = Vector2<f32>;
using vec2d = Vector2<f64>;
using vec2i = Vector2<Interval<real_t>>;
using vec3i = Vector3<Interval<real_t>>;

} // namespace hermes::geo

namespace hermes {

HERMES_TYPE_LAYOUT_METHODS(geo::vec2, f32, 2)
HERMES_TYPE_LAYOUT_METHODS(geo::vec2d, f64, 2)
HERMES_TYPE_LAYOUT_METHODS(geo::vec3, f32, 3)
HERMES_TYPE_LAYOUT_METHODS(geo::vec3d, f64, 3)
HERMES_TYPE_LAYOUT_METHODS(geo::vec4, real_t, 4)
HERMES_TYPE_LAYOUT_METHODS(geo::vec2i, Interval<real_t>, 2)
HERMES_TYPE_LAYOUT_METHODS(geo::vec3i, Interval<real_t>, 3)

template <typename T> struct DebugTraits<geo::Vector2<T>> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Vector2<T> &data) {
    return DebugMessage("V[{}, {}]", hermes::to_string(data.x),
                        hermes::to_string(data.y));
  }
};

template <typename T> struct DebugTraits<geo::Vector3<T>> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Vector3<T> &data) {
    return DebugMessage("V[{}, {}, {}]", hermes::to_string(data.x),
                        hermes::to_string(data.y), hermes::to_string(data.z));
  }
};

template <typename T> struct DebugTraits<geo::Vector4<T>> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Vector4<T> &data) {
    return DebugMessage("V[{}, {}, {}, {}]", hermes::to_string(data.x),
                        hermes::to_string(data.y), hermes::to_string(data.z),
                        hermes::to_string(data.w));
  }
};

} // namespace hermes

// std hash support
namespace std {

/// Hash support for vector2
/// \tparam T
template <typename T> struct hash<hermes::geo::Vector2<T>> {
  /// Computes hash for a given vector
  /// \param v
  /// \return
  size_t operator()(hermes::geo::Vector2<T> const &v) const {
    hash<T> hasher;
    size_t s = 0;
    // inject x component
    size_t h = hasher(v.x);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    // inject y component
    h = hasher(v.y);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    return s;
  }
};

/// Hash support for vector3
/// \tparam T
template <typename T> struct hash<hermes::geo::Vector3<T>> {
  /// Computes hash for a given vector
  /// \param v
  /// \return
  size_t operator()(hermes::geo::Vector3<T> const &v) const {
    hash<T> hasher;
    size_t s = 0;
    // inject x component
    size_t h = hasher(v.x);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    // inject y component
    h = hasher(v.y);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    // inject y component
    h = hasher(v.z);
    h += 0x9e3779b9 + (s << 6) + (s >> 2);
    s ^= h;
    return s;
  }
};

} // namespace std
