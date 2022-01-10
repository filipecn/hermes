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
///\file vector.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2017-08-19
///
///\brief

#ifndef HERMES_GEOMETRY_VECTOR_H
#define HERMES_GEOMETRY_VECTOR_H

#include <hermes/numeric/numeric.h>
#include <hermes/common/debug.h>
#include <hermes/common/index.h>
#include <hermes/numeric/interval.h>

#include <cstring>
#include <initializer_list>
#include <vector>

namespace hermes {

/// Forward declaration of Point2
/// \tparam T
template<typename T> class Point2;
// *********************************************************************************************************************
//                                                                                                            Vector2
// *********************************************************************************************************************
/// \tparam T
template<typename T> class Vector2 : public MathElement<T, 2u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, Interval<f32>>::value || std::is_same<T, Interval<f64>>::value,
                "Vector2 must hold an float type!");

public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Vector2() : x{0}, y{0} {};
  HERMES_DEVICE_CALLABLE Vector2(T _x, T _y) : x(_x), y(_y) {}
  HERMES_DEVICE_CALLABLE explicit Vector2(const Point2<T> &p) : x(p.x), y(p.y) {}
  HERMES_DEVICE_CALLABLE explicit Vector2(T f) { x = y = f; }
  HERMES_DEVICE_CALLABLE explicit Vector2(T *f) {
    x = f[0];
    y = f[1];
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE T operator[](size_t i) const { return (&x)[i]; }
  HERMES_DEVICE_CALLABLE T &operator[](size_t i) { return (&x)[i]; }
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP)                                                                                           \
  HERMES_DEVICE_CALLABLE Vector2 &operator  OP##= (const Vector2 &v) {                                              \
    x OP##= v.x; y OP##= v.y; return *this;  }                                                                      \
  HERMES_DEVICE_CALLABLE Vector2 &operator  OP##= (real_t f) {                                                      \
    x OP##= f; y OP##= f; return *this;  }                                                                          \
  HERMES_DEVICE_CALLABLE Vector2 operator OP (const Vector2<T> &b) const {                                          \
    return {x OP b.x, y OP b.y}; }                                                                                  \
  HERMES_DEVICE_CALLABLE Vector2 operator OP (real_t f) const {                                                     \
    return {x OP f, y OP f}; }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP
  HERMES_DEVICE_CALLABLE Vector2 operator-() const { return Vector2(-x, -y); }
  //                                                                                                       relational
#define RELATIONAL_OP(OP, CO)                                                                                       \
  HERMES_DEVICE_CALLABLE bool operator OP (const Vector2 &b) const {                                                \
    return x OP b.x CO y OP b.y; }
  RELATIONAL_OP(<, &&)
  RELATIONAL_OP(>, &&)
  RELATIONAL_OP(<=, &&)
  RELATIONAL_OP(>=, &&)
  RELATIONAL_OP(!=, ||)
#undef RELATIONAL_OP
  HERMES_DEVICE_CALLABLE bool operator==(const Vector2<T> &b) const {
    return Check::is_equal(x, b.x) && Check::is_equal(y, b.y);
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE T length2() const { return x * x + y * y; }
  HERMES_DEVICE_CALLABLE T length() const { return sqrtf(length2()); }
  HERMES_DEVICE_CALLABLE Vector2 right() const { return Vector2(y, -x); }
  HERMES_DEVICE_CALLABLE Vector2 left() const { return Vector2(-y, x); }
  //                                                                                                          swizzle
  Vector2 xy() const { return {x, y}; }
  Vector2 yx() const { return {y, x}; }
  Vector2 xx() const { return {x, x}; }
  Vector2 yy() const { return {y, y}; }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T x = T(0.0);
  T y = T(0.0);
};

template<typename T> class Point3;

// *********************************************************************************************************************
//                                                                                                            Vector3
// *********************************************************************************************************************
template<typename T> class Vector3 : public MathElement<T, 3u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, Interval<f32>>::value || std::is_same<T, Interval<f64>>::value,
                "Vector3 must hold an float type!");

public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Vector3() : x{0}, y{0}, z{0} {}
  HERMES_DEVICE_CALLABLE explicit Vector3(T _f) : x(_f), y(_f), z(_f) {}
  HERMES_DEVICE_CALLABLE Vector3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  HERMES_DEVICE_CALLABLE explicit Vector3(const T *v) {
    x = v[0];
    y = v[1];
    z = v[2];
  }
  //                                                                                                       conversion
  HERMES_DEVICE_CALLABLE explicit Vector3(const Point3<T> &p) : x(p.x), y(p.y), z(p.z) {}
  template<typename S, typename C = T>
  HERMES_DEVICE_CALLABLE explicit Vector3(const Vector3<Interval<S>> &vi,
                                          typename std::enable_if_t<
                                              !std::is_same_v<C, Interval<f32>>
                                                  && !std::is_same_v<C, Interval<f64>>> * = nullptr) :
      x(vi.x), y(vi.y), z(vi.z) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  HERMES_DEVICE_CALLABLE Vector3 &operator=(const T &v) {
    x = y = z = v;
    return *this;
  }
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP)                                                                                           \
  HERMES_DEVICE_CALLABLE Vector3 &operator  OP##= (const Vector3 &v) {                                              \
    x OP##= v.x; y OP##= v.y; z OP##= v.z; return *this;  }                                                         \
  HERMES_DEVICE_CALLABLE Vector3 &operator  OP##= (real_t f) {                                                      \
    x OP##= f; y OP##= f; z OP##= f; return *this;  }                                                               \
  HERMES_DEVICE_CALLABLE Vector3 operator OP (const Vector3<T> &b) const {                                          \
    return {x OP b.x, y OP b.y, z OP b.z}; }                                                                        \
  HERMES_DEVICE_CALLABLE Vector3 operator OP (T f) const {                                                          \
    return {x OP f, y OP f, z OP f}; }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
#undef ARITHMETIC_OP
  HERMES_DEVICE_CALLABLE Vector3<T> &operator/=(T f) {
    HERMES_CHECK_EXP(Check::is_zero(f))
    T inv = 1.f / f;
    x *= inv;
    y *= inv;
    z *= inv;
    return *this;
  }
  HERMES_DEVICE_CALLABLE Vector3<T> &operator/=(const Vector3<T> &v) {
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
  }
  HERMES_DEVICE_CALLABLE Vector3<T> operator/(const T &f) const {
    T inv = 1.f / f;
    return Vector3<T>(x * inv, y * inv, z * inv);
  }
  HERMES_DEVICE_CALLABLE Vector3<T> operator-() const { return Vector3(-x, -y, -z); }
  //                                                                                                       relational
  HERMES_DEVICE_CALLABLE bool operator==(const Vector3<T> &b) const {
    return Check::is_equal(x, b.x) && Check::is_equal(y, b.y) &&
        Check::is_equal(z, b.z);
  }
  HERMES_DEVICE_CALLABLE bool operator<(const Vector3<T> &b) const {
    if (x < b.x)
      return true;
    if (y < b.y)
      return true;
    return z < b.z;
  }
  HERMES_DEVICE_CALLABLE bool operator>(const Vector3<T> &b) const {
    if (x > b.x)
      return true;
    if (y > b.y)
      return true;
    return z > b.z;
  }
  HERMES_DEVICE_CALLABLE bool operator>=(const Vector3<T> &b) const {
    return x >= b.x && y >= b.y && z >= b.z;
  }
  HERMES_DEVICE_CALLABLE bool operator<=(const Vector3<T> &b) const {
    return x <= b.x && y <= b.y && z <= b.z;
  }
  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE T operator[](int i) const { return (&x)[i]; }
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&x)[i]; }
  HERMES_DEVICE_CALLABLE Vector2<T> xy(int i = 0, int j = 1) const {
    return Vector2<T>((&x)[i], (&x)[j]);
  }
  // *******************************************************************************************************************
  //                                                                                                          METRICS
  // *******************************************************************************************************************
  /// \note Also called L1-norm, taxicab norm, Manhattan norm
  /// \note Defined as ||v||_1 = sum_i(|v_i|)
  /// \return L1-norm of this vector
  HERMES_DEVICE_CALLABLE T mLength() const { return std::abs(x) + std::abs(y) + std::abs(z); }
  /// \note Also called L2-norm, Euclidean norm, Euclidean distance, 2-norm
  /// \note Defined as ||v|| = (v_i * v_i)^(1/2)
  /// \return 2-norm of this vector
  template<typename C = T>
  HERMES_DEVICE_CALLABLE T length(typename std::enable_if_t<!std::is_same_v<C, Interval<f32>>
                                                                && !std::is_same_v<C,
                                                                                   Interval<f64>>> * = nullptr) const {
    return std::sqrt(length2());
  }
  template<typename C = T>
  HERMES_DEVICE_CALLABLE T length(typename std::enable_if_t<std::is_same_v<C, Interval<f32>>
                                                                || std::is_same_v<C,
                                                                                  Interval<f64>>> * = nullptr) const {
    return length2().sqrt();
  }
  /// \note Also called squared Euclidean distance
  /// \note Defined as ||v||^2 = v_i * v_i
  /// \return squared 2-norm of this vector
  HERMES_DEVICE_CALLABLE T length2() const { return x * x + y * y + z * z; }
  /// \note Also called maximum norm, infinity norm
  /// \note Defined as ||v||_inf = argmax max(|v_i|)
  /// \return greatest absolute component value
  HERMES_DEVICE_CALLABLE T maxAbs() const {
    if (std::abs(x) > std::abs(y) && std::abs(x) > std::abs(z))
      return x;
    if (std::abs(y) > std::abs(x) && std::abs(y) > std::abs(z))
      return y;
    return z;
  }
  /// \note Defined as argmax v_i
  /// \return greatest component value
  HERMES_DEVICE_CALLABLE T max() const {
    if (x > y && x > z)
      return x;
    if (y > x && y > z)
      return y;
    return z;
  }
  /// \note Defined as argmax_i v_i
  /// \return Index of component with greatest value
  [[nodiscard]] HERMES_DEVICE_CALLABLE int maxDimension() const {
    if (x > y && x > z)
      return 0;
    if (y > x && y > z)
      return 1;
    return 2;
  }
  /// \note Defined as argmax_i |v_i|
  /// \return Index of dimension with greatest value
  [[nodiscard]] HERMES_DEVICE_CALLABLE int maxAbsDimension() const {
    if (std::abs(x) > std::abs(y) && std::abs(x) > std::abs(z))
      return 0;
    if (std::abs(y) > std::abs(x) && std::abs(y) > std::abs(z))
      return 1;
    return 2;
  }
  // *******************************************************************************************************************
  //                                                                                                       OPERATIONS
  // *******************************************************************************************************************
  /// \note Normalization by vector length
  /// \note Defined as v / ||v||
  HERMES_DEVICE_CALLABLE void normalize() {
    auto l = length();
    if (l != 0.f) {
      x /= l;
      y /= l;
      z /= l;
    }
  }
  /// \note Normalization by vector length
  /// \note Defined as v / ||v||
  /// \return Normalized vector of this vector
  HERMES_DEVICE_CALLABLE Vector3 normalized() const {
    auto l = length();
    return (*this) / l;
  }
  /// \note b * dot(v,b) / ||b||
  /// \param b vector to project onto
  /// \return projection of this vector onto **b**
  HERMES_DEVICE_CALLABLE Vector3 projectOnto(const Vector3 &b) {
    return (dot(b, *this) / b.length2()) * b;
  }
  /// \note v - b * dot(v,b) / ||b||
  /// \param b vector of rejection
  /// \return rejection of this vector on **b**
  HERMES_DEVICE_CALLABLE Vector3 rejectOn(const Vector3 b) {
    return *this - (dot(b, *this) / b.length2()) * b;
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T x = T(0.0);
  T y = T(0.0);
  T z = T(0.0);
};

// *********************************************************************************************************************
//                                                                                                            Vector4
// *********************************************************************************************************************
template<typename T> class Vector4 : public MathElement<T, 4> {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Vector4() : x{0}, y{0}, z{0}, w{0} {}
  HERMES_DEVICE_CALLABLE Vector4(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE Vector4<T> &operator+=(const Vector4<T> &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
  }
  HERMES_DEVICE_CALLABLE Vector4<T> &operator-=(const Vector4<T> &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
  }
  HERMES_DEVICE_CALLABLE Vector4<T> &operator*=(T f) {
    x *= f;
    y *= f;
    z *= f;
    w *= f;
    return *this;
  }
  HERMES_DEVICE_CALLABLE Vector4<T> &operator/=(T f) {
    T inv = 1.f / f;
    x *= inv;
    y *= inv;
    z *= inv;
    w *= inv;
    return *this;
  }
  HERMES_DEVICE_CALLABLE Vector4<T> operator-() const { return Vector4(-x, -y, -z, -w); }
  HERMES_DEVICE_CALLABLE Vector4<T> operator+(const Vector4<T> &b) {
    return Vector4<T>(x + b.x, y + b.y, z + b.z, w + b.w);
  }
  HERMES_DEVICE_CALLABLE Vector4<T> operator-(const Vector4<T> &b) {
    return Vector4<T>(x - b.x, y - b.y, z - b.z, w - b.w);
  }
  HERMES_DEVICE_CALLABLE Vector4<T> operator*(T f) {
    return Vector4<T>(x * f, y * f, z * f, w * f);
  }
  HERMES_DEVICE_CALLABLE Vector4<T> operator/(T f) {
    T inv = 1.f / f;
    return Vector4<T>(x * inv, y * inv, z * inv, w * inv);
  }
  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE T operator[](int i) const { return (&x)[i]; }
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&x)[i]; }
  HERMES_DEVICE_CALLABLE Vector2<T> xy() { return Vector2<T>(x, y); }
  HERMES_DEVICE_CALLABLE Vector3<T> xyz() { return Vector3<T>(x, y, z); }
  // *******************************************************************************************************************
  //                                                                                                          METRICS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE T length2() const { return x * x + y * y + z * z + w * w; }
  HERMES_DEVICE_CALLABLE T length() const { return sqrtf(length2()); }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T x = T(0.0);
  T y = T(0.0);
  T z = T(0.0);
  T w = T(0.0);
};

// *********************************************************************************************************************
//                                                                                                 EXTERNAL FUNCTIONS
// *********************************************************************************************************************
//                                                                                                           geometry
template<typename T>
HERMES_DEVICE_CALLABLE  T dot(const Vector2<T> &a, const Vector2<T> &b) {
  return a.x * b.x + a.y * b.y;
}
template<typename T>
HERMES_DEVICE_CALLABLE  T dot(const Vector3<T> &a, const Vector3<T> &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template<typename T>
HERMES_DEVICE_CALLABLE  T cross(const Vector2<T> &a, const Vector2<T> &b) {
  return a.x * b.y - a.y * b.x;
}
template<typename T>
HERMES_DEVICE_CALLABLE  Vector3<T> cross(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>((a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z),
                    (a.x * b.y) - (a.y * b.x));
}
template<typename T>
HERMES_DEVICE_CALLABLE  T triple(const Vector3<T> &a, const Vector3<T> &b, const Vector3<T> &c) {
  return dot(a, cross(b, c));
}
template<typename T>
HERMES_DEVICE_CALLABLE  Vector2<T> normalize(const Vector2<T> &v) {
  return v / v.length();
}
template<typename T>
HERMES_DEVICE_CALLABLE  Vector3<T> normalize(const Vector3<T> &v) {
  if (v.length2() == 0.f)
    return v;
  return v / v.length();
}
template<typename T>
HERMES_DEVICE_CALLABLE  Vector2<T> orthonormal(const Vector2<T> &v, bool first = true) {
  Vector2<T> n = normalize(v);
  if (first)
    return Vector2<T>(-n.y, n.x);
  return Vector2<T>(n.y, -n.x);
}
/// Projects a vector onto another.
/// \param a **[in]**
/// \param b **[in]**
/// \returns the projection of **a** onto **b**
template<typename T>
HERMES_DEVICE_CALLABLE  Vector2<T> project(const Vector2<T> &a, const Vector2<T> &b) {
  return (dot(b, a) / b.length2()) * b;
}
/// \note b * dot(a,b) / ||b||
/// \tparam T
/// \param a
/// \param b
/// \return projection of **a** onto **b**
template<typename T>
HERMES_DEVICE_CALLABLE  Vector3<T> project(const Vector3<T> &a, const Vector3<T> &b) {
  return (dot(b, a) / b.length2()) * b;
}
/// \brief compute the two orthogonal-tangential vectors from a
/// \param a **[in]** normal
/// \param b **[out]** first tangent
/// \param c **[out]** second tangent
template<typename T>
HERMES_DEVICE_CALLABLE  void tangential(const Vector3<T> &a, Vector3<T> &b, Vector3<T> &c) {
  b = hermes::normalize(cross(a, ((std::abs(a.y) > 0.f || std::abs(a.z) > 0.f)
                                  ? Vector3<T>(1, 0, 0)
                                  : Vector3<T>(0, 1, 1))));
  c = hermes::normalize(cross(a, b));
}


//                                                                                                         arithmetic
#define DOP2(OP) f OP v.x, f OP v.y
#define DOP3(OP) f OP v.x, f OP v.y, f OP v.z
#define MATH_OP(D, OP)                                                                                                 \
template<typename T>                                                                                                   \
HERMES_DEVICE_CALLABLE  Vector##D<T> operator OP(T f, const Vector##D<T> &v) {                                         \
  return Vector##D<T>(DOP##D(OP)); }
MATH_OP(2, *)
MATH_OP(2, /)
MATH_OP(3, *)
MATH_OP(3, /)
#undef MATH_OP
#undef DOP2
#undef DOP3

#define DOP2(OP) OP(a.x, b.x), OP(a.y, b.y)
#define DOP3(OP) OP(a.x, b.x), OP(a.y, b.y), OP(a.z, b.z)
#define MATH_OP(D, NAME, OP) \
template<typename T> \
HERMES_DEVICE_CALLABLE  Vector##D<T> NAME(const Vector##D<T> &a, const Vector##D<T> &b) { \
  return Vector##D<T>(DOP##D(OP)); }
#ifdef HERMES_DEVICE_ENABLED
MATH_OP(2, min, min)
MATH_OP(2, max, max)
MATH_OP(3, min, min)
MATH_OP(3, max, max)
#else
MATH_OP(2, min, std::min)
MATH_OP(2, max, std::max)
MATH_OP(3, min, std::min)
MATH_OP(3, max, std::max)
#endif
#undef MATH_OP
#undef DOP2
#undef DOP3

//                                                                                                            numbers
#define DOP2(OP) OP(v.x), OP(v.y)
#define DOP3(OP) OP(v.x), OP(v.y), OP(v.z)
#define MATH_OP(NAME, OP, D)                                                                                          \
  template<typename T>                                                                                                 \
  HERMES_DEVICE_CALLABLE Vector##D<T> NAME(const Vector##D<T>& v) {                                                    \
    return Vector##D<T>(DOP##D(OP));  }
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
// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector2<T> &v) {
  os << "Vector2 [" << v.x << " " << v.y << "]";
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector3<T> &v) {
  os << "Vector3 [" << v.x << " " << v.y << " " << v.z << "]";
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector4<T> &v) {
  os << "Vector4 [" << v.x << " " << v.y << " " << v.z << " " << v.w << "]";
  return os;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using vec2 = Vector2<real_t>;
using vec3 = Vector3<real_t>;
using vec4 = Vector4<real_t>;
using vec3d = Vector3<double>;
using vec3f = Vector3<float>;
using vec2f = Vector2<float>;
using vec2i = Vector2<Interval<real_t>>;
using vec3i = Vector3<Interval<real_t>>;

} // namespace hermes

// std hash support
namespace std {

template<typename T> struct hash<hermes::Vector2<T>> {
  size_t operator()(hermes::Vector2<T> const &v) const {
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

template<typename T> struct hash<hermes::Vector3<T>> {
  size_t operator()(hermes::Vector3<T> const &v) const {
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

#endif
