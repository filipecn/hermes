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
///\file point.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-08-18
///
///\brief

#ifndef HERMES_GEOMETRY_POINT_H
#define HERMES_GEOMETRY_POINT_H

#include <initializer_list>

#include <hermes/geometry/vector.h>
#include <hermes/common/debug.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                             Point2
// *********************************************************************************************************************
///\tparam T
template<typename T> class Point2 : public MathElement<T, 2u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value || std::is_same<T, float>::value ||
      std::is_same<T, double>::value, "Point2 must hold a float type!");
public:
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE friend Point2<T> operator*(real_t f, const Point2<T> &a) {
    return Point2<T>(a.x * f, a.y * f);
  }
  //                                                                                                         geometry
  HERMES_DEVICE_CALLABLE friend real_t distance(const Point2<T> &a, const Point2<T> &b) {
    return (a - b).length();
  }
  HERMES_DEVICE_CALLABLE friend real_t distance2(const Point2<T> &a, const Point2<T> &b) {
    return (a - b).length2();
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE explicit Point2(T f = T(0)) { x = y = f; }
  HERMES_DEVICE_CALLABLE explicit Point2(const real_t *v) : x(v[0]), y(v[1]) {}
  HERMES_DEVICE_CALLABLE Point2(real_t _x, real_t _y) : x(_x), y(_y) {}
  template<typename U>
  HERMES_DEVICE_CALLABLE Point2(const Index2 <U> &index) : x{static_cast<T>(index.i)}, y{static_cast<T>(index.j)} {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  template<typename U>
  HERMES_DEVICE_CALLABLE Point2 &operator=(const Index2 <U> &index) {
    x = index.i;
    y = index.j;
    return *this;
  }
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE T operator[](int i) const { return (&x)[i]; }
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&x)[i]; }
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP)                                                                                           \
  HERMES_DEVICE_CALLABLE Point2 &operator OP##= (const Vector2 <T> &v) {                                            \
    x OP##= v.x; y OP##= v.y; return *this; }                                                                       \
  HERMES_DEVICE_CALLABLE Point2 &operator OP##= (real_t f) {                                                        \
    x OP##= f; y OP##= f; return *this; }                                                                           \
  HERMES_DEVICE_CALLABLE Point2 operator OP (const Vector2 <T> &v) const {                                          \
    return {x OP v.x, y OP v.y}; }                                                                                  \
  HERMES_DEVICE_CALLABLE Point2 operator OP (real_t f) const {                                                      \
    return {x OP f, y OP f}; }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(/)
  ARITHMETIC_OP(*)
#undef ARITHMETIC_OP
  HERMES_DEVICE_CALLABLE Vector2 <T> operator-(const Point2<T> &b) const {
    return Vector2<T>(x - b.x, y - b.y);
  }
  //                                                                                                       relational
#define RELATIONAL_OP(OP, CO)                                                                                       \
  HERMES_DEVICE_CALLABLE bool operator OP (const Point2<T> &b) const {                                              \
    return x OP b.x CO y OP b.y; }
  RELATIONAL_OP(<, &&)
  RELATIONAL_OP(>, &&)
  RELATIONAL_OP(<=, &&)
  RELATIONAL_OP(>=, &&)
  RELATIONAL_OP(!=, ||)
#undef RELATIONAL_OP
  HERMES_DEVICE_CALLABLE bool operator==(const Point2<T> &b) const {
    return Check::is_equal(x, b.x) && Check::is_equal(y, b.y);
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T x = T(0.0);
  T y = T(0.0);
};

// *********************************************************************************************************************
//                                                                                                             Point3
// *********************************************************************************************************************
template<typename T> class Point3 : public MathElement<T, 3u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value || std::is_same<T, float>::value ||
      std::is_same<T, double>::value, "Size2 must hold an float type!");
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE friend real_t distance(const Point3<T> &a, const Point3<T> &b) {
    return (a - b).length();
  }
  HERMES_DEVICE_CALLABLE friend real_t distance2(const Point3<T> &a, const Point3<T> &b) {
    return (a - b).length2();
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Point3() : x{0}, y{0}, z{0} {}
  HERMES_DEVICE_CALLABLE explicit Point3(real_t v) { x = y = z = v; }
  HERMES_DEVICE_CALLABLE Point3(real_t _x, real_t _y, real_t _z) : x(_x), y(_y), z(_z) {}
  HERMES_DEVICE_CALLABLE explicit Point3(const Vector3 <T> &v) : x(v.x), y(v.y), z(v.z) {}
  HERMES_DEVICE_CALLABLE explicit Point3(const Point2<T> &p) : x(p.x), y(p.y), z(0) {}
  HERMES_DEVICE_CALLABLE explicit Point3(const real_t *v) : x(v[0]), y(v[1]), z(v[2]) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
  HERMES_DEVICE_CALLABLE explicit operator Vector3<T>() const { return Vector3<T>(x, y, z); }
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE T operator[](int i) const { return (&x)[i]; }
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&x)[i]; }
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP)                                                                                           \
  HERMES_DEVICE_CALLABLE Point3 &operator OP##= (const Vector3 <T> &v) {                                            \
    x OP##= v.x; y OP##= v.y; z OP##= v.z; return *this; }                                                          \
  HERMES_DEVICE_CALLABLE Point3 &operator OP##= (real_t f) {                                                        \
    x OP##= f; y OP##= f; z OP##= f; return *this; }                                                                \
  HERMES_DEVICE_CALLABLE Point3 operator OP (const Vector3 <T> &v) const {                                          \
    return {x OP v.x, y OP v.y, z OP v.z}; }                                                                        \
  HERMES_DEVICE_CALLABLE Point3 operator OP (real_t f) const {                                                      \
    return {x OP f, y OP f, z OP f}; }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(/)
  ARITHMETIC_OP(*)
#undef ARITHMETIC_OP
  HERMES_DEVICE_CALLABLE Vector3 <T> operator-(const Point3<T> &b) const {
    return Vector3<T>(x - b.x, y - b.y, z - b.z);
  }
  //                                                                                                       relational
#define RELATIONAL_OP(OP, CO)                                                                                       \
  HERMES_DEVICE_CALLABLE bool operator OP (const Point3<T> &b) const {                                              \
    return x OP b.x CO y OP b.y CO z OP b.z; }
  RELATIONAL_OP(<, &&)
  RELATIONAL_OP(>, &&)
  RELATIONAL_OP(<=, &&)
  RELATIONAL_OP(>=, &&)
  RELATIONAL_OP(!=, ||)
#undef RELATIONAL_OP
  HERMES_DEVICE_CALLABLE bool operator==(const Point3<T> &b) const {
    return Check::is_equal(x, b.x) && Check::is_equal(y, b.y) &&
        Check::is_equal(z, b.z);
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE Point2<T> xy() const { return Point2<T>(x, y); }
  HERMES_DEVICE_CALLABLE Point2<T> yz() const { return Point2<T>(y, z); }
  HERMES_DEVICE_CALLABLE Point2<T> xz() const { return Point2<T>(x, z); }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T x = T(0.0);
  T y = T(0.0);
  T z = T(0.0);
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
std::ostream &operator<<(std::ostream &os, const Point2<T> &p) {
  os << "Point2[" << p.x << " " << p.y << "]";
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Point3<T> &p) {
  os << "Point3[" << p.x << " " << p.y << " " << p.z << "]";
  return os;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using point2 = Point2<real_t>;
using point2f = Point2<float>;
using point2d = Point2<double>;
using point3 = Point3<real_t>;
using point3f = Point3<float>;
using point3d = Point3<double>;

} // namespace hermes

// std hash support
namespace std {

template<typename T> struct hash<hermes::Point2<T>> {
  size_t operator()(hermes::Point2<T> const &v) const {
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

template<typename T> struct hash<hermes::Point3<T>> {
  size_t operator()(hermes::Point3<T> const &v) const {
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
