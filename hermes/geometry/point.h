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
///\date 2017-08-18
///
///\brief Geometric point classes
///
///\ingroup geometry
///\addtogroup geometry
/// @{

#ifndef HERMES_GEOMETRY_POINT_H
#define HERMES_GEOMETRY_POINT_H

#include <initializer_list>

#include <hermes/geometry/vector.h>
#include <hermes/common/debug.h>
#include <hermes/logging/memory_dump.h>
#include <hermes/numeric/interval.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                             Point2
// *********************************************************************************************************************
/// \brief Geometric 2-dimensional point (x, y)
/// \tparam T
template<typename T> class Point2 : public MathElement<T, 2u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, Interval<f32>>::value || std::is_same<T, Interval<f64>>::value,
                "Point2 must hold a float type!");
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Constructs from single component value
  /// \param f
  HERMES_DEVICE_CALLABLE explicit Point2(T f = T(0)) { x = y = f; }
  /// \brief Constructs from component array
  /// \param v
  HERMES_DEVICE_CALLABLE explicit Point2(const real_t *v) : x(v[0]), y(v[1]) {}
  /// \brief Constructs from component values
  /// \param _x
  /// \param _y
  HERMES_DEVICE_CALLABLE Point2(real_t _x, real_t _y) : x(_x), y(_y) {}
  /// \brief Constructs from index2
  /// \tparam U
  /// \param index
  template<typename U>
  HERMES_DEVICE_CALLABLE Point2(const Index2<U> &index) : x{static_cast<T>(index.i)}, y{static_cast<T>(index.j)} {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
  /// \brief Casts to index2
  /// \tparam U
  /// \return
  template<typename U>
  HERMES_DEVICE_CALLABLE operator Index2<U>() const {
    return Index2<U>(x, y);
  }
  //                                                                                                       assignment
  /// \brief Copy assigns from index 2
  /// \tparam U
  /// \param index
  /// \return
  template<typename U>
  HERMES_DEVICE_CALLABLE Point2 &operator=(const Index2<U> &index) {
    x = index.i;
    y = index.j;
    return *this;
  }
  //                                                                                                           access
  /// \brief Get i-th component
  /// \warning `i` is not checked
  /// \param i component index in [0, 1]
  /// \return
  HERMES_DEVICE_CALLABLE const T &operator[](int i) const { return (&x)[i]; }
  /// \brief Get i-th component reference
  /// \warning `i` is not checked
  /// \param i component index in [0, 1]
  /// \return
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
  HERMES_DEVICE_CALLABLE Vector2<T> operator-(const Point2<T> &b) const {
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
  T x = T(0.0); //!< 0-th component
  T y = T(0.0); //!< 1-th component
};

// *********************************************************************************************************************
//                                                                                                             Point3
// *********************************************************************************************************************
/// \brief Geometric 3-dimensional vector (x, y, z)
/// \tparam T
template<typename T> class Point3 : public MathElement<T, 3u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, Interval<f32>>::value || std::is_same<T, Interval<f64>>::value,
                "Size2 must hold an float type!");
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Default constructor
  HERMES_DEVICE_CALLABLE Point3() : x{0}, y{0}, z{0} {}
  /// \brief Constructs from single component value
  /// \param v
  HERMES_DEVICE_CALLABLE explicit Point3(T v) { x = y = z = v; }
  /// \brief Constructs from component values
  /// \param _x
  /// \param _y
  /// \param _z
  HERMES_DEVICE_CALLABLE Point3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
  /// \brief Constructs from point2
  /// \param p
  HERMES_DEVICE_CALLABLE explicit Point3(const Point2<T> &p, T _z = 0) : x(p.x), y(p.y), z(_z) {}
  /// \brief Constructs from component array
  /// \param v
  HERMES_DEVICE_CALLABLE explicit Point3(const real_t *v) : x(v[0]), y(v[1]), z(v[2]) {}
  /// \brief Constructs from interval center and radius
  /// \tparam S
  /// \tparam C
  /// \param c
  /// \param r
  template<typename S, typename C = T>
  HERMES_DEVICE_CALLABLE explicit Point3(const Point3<S> &c, const Vector3<S> &r,
                                         typename std::enable_if_t<
                                             std::is_same_v<C, Interval<f32>>
                                                 || std::is_same_v<C, Interval<f64>>> * = nullptr)
      :      x(Interval<S>::withRadius(c.x, r.x)),
             y(Interval<S>::withRadius(c.y, r.y)), z(Interval<S>::withRadius(c.z, r.z)) {}
  //                                                                                                       conversion
  /// \brief Constructs from vector
  /// \param v
  HERMES_DEVICE_CALLABLE explicit Point3(const Vector3<T> &v) : x(v.x), y(v.y), z(v.z) {}
  /// \brief Constructs from interval point3
  /// \tparam S
  /// \tparam C
  /// \param vi
  template<typename S, typename C = T>
  HERMES_DEVICE_CALLABLE explicit Point3(const Point3<Interval<S>> &vi,
                                         typename std::enable_if_t<
                                             !std::is_same_v<C, Interval<f32>>
                                                 && !std::is_same_v<C, Interval<f64>>> * = nullptr) :
      x(vi.x), y(vi.y), z(vi.z) {}
  /// \brief Constructs from index3
  /// \tparam U
  /// \param index
  template<typename U>
  HERMES_DEVICE_CALLABLE Point3(const Index3<U> &index) : x{static_cast<T>(index.i)}, y{static_cast<T>(index.j)},
                                                          z{static_cast<T>(index.k)} {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
  /// \brief Converts to vector3
  /// \return
  HERMES_DEVICE_CALLABLE explicit operator Vector3<T>() const { return Vector3<T>(x, y, z); }
  //                                                                                                           access
  /// \brief Get i-th component
  /// \warning `i` is not checked
  /// \param i component index in [0, 2]
  /// \return
  HERMES_DEVICE_CALLABLE T operator[](int i) const { return (&x)[i]; }
  /// \brief Get i-th component reference
  /// \warning `i` is not checked
  /// \param i component index in [0, 2]
  /// \return
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&x)[i]; }
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP)                                                                                           \
  HERMES_DEVICE_CALLABLE Point3 &operator OP##= (const Vector3 <T> &v) {                                            \
    x OP##= v.x; y OP##= v.y; z OP##= v.z; return *this; }                                                          \
  HERMES_DEVICE_CALLABLE Point3 &operator OP##= (real_t f) {                                                        \
    x OP##= f; y OP##= f; z OP##= f; return *this; }                                                                \
  HERMES_DEVICE_CALLABLE Point3 operator OP (const Vector3 <T> &v) const {                                          \
    return {x OP v.x, y OP v.y, z OP v.z}; }                                                                        \
  HERMES_DEVICE_CALLABLE Point3 operator OP (T f) const {                                                           \
    return {x OP f, y OP f, z OP f}; }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(/)
  ARITHMETIC_OP(*)
#undef ARITHMETIC_OP
  HERMES_DEVICE_CALLABLE Vector3<T> operator-(const Point3<T> &b) const {
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
  /// \brief Gets 2-dimensional swizzle (x, y)
  /// \return
  HERMES_DEVICE_CALLABLE Point2<T> xy() const { return Point2<T>(x, y); }
  /// \brief Gets 2-dimensional swizzle (y, z)
  /// \return
  HERMES_DEVICE_CALLABLE Point2<T> yz() const { return Point2<T>(y, z); }
  /// \brief Gets 2-dimensional swizzle (x, z)
  /// \return
  HERMES_DEVICE_CALLABLE Point2<T> xz() const { return Point2<T>(x, z); }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T x = T(0.0); //!< 0-th component
  T y = T(0.0); //!< 1-th component
  T z = T(0.0); //!< 2-th component
};

// *********************************************************************************************************************
//                                                                                                 EXTERNAL FUNCTIONS
// *********************************************************************************************************************
//                                                                                                         arithmetic
/// \brief Scalar multiplication operator
/// \tparam T
/// \param f
/// \param a
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE Point2<T> operator*(real_t f, const Point2<T> &a) {
  return Point2<T>(a.x * f, a.y * f);
}
//                                                                                                           geometry
/// \brief Computes the Euclidean distance between two points
/// \tparam T
/// \param a
/// \param b
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE real_t distance(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length();
}
/// \brief Computes the squared Euclidean distance between two points
/// \tparam T
/// \param a
/// \param b
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE real_t distance2(const Point2<T> &a, const Point2<T> &b) {
  return (a - b).length2();
}
/// \brief Computes the Euclidean distance between two points
/// \tparam T
/// \param a
/// \param b
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE real_t distance(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length();
}
/// \brief Computes the squared Euclidean distance between two points
/// \tparam T
/// \param a
/// \param b
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE real_t distance2(const Point3<T> &a, const Point3<T> &b) {
  return (a - b).length2();
}
//                                                                                                            numbers
#define MATH_OP(NAME, OP)                                                                                           \
  template<typename T>                                                                                              \
  HERMES_DEVICE_CALLABLE Point2<T> NAME(const Point2<T>& p) {                                                       \
    return Point2<T>(OP(p.x), OP(p.y));  }
#ifdef HERMES_DEVICE_ENABLED
MATH_OP(floor, ::floor)
MATH_OP(ceil, ::ceil)
MATH_OP(abs, ::abs)
#else
MATH_OP(floor, std::floor)
MATH_OP(ceil, std::ceil)
MATH_OP(abs, std::abs)
#endif
#undef MATH_OP

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
using point2i = Point2<Interval<real_t>>;
using point3i = Point3<Interval<real_t>>;

} // namespace hermes

// std hash support
namespace std {

/// brief Hash support for point2
/// \tparam T
template<typename T> struct hash<hermes::Point2<T>> {
  /// \brief Computes hash for a given vector
  /// \param v
  /// \return
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

/// brief Hash support for point3
/// \tparam T
template<typename T> struct hash<hermes::Point3<T>> {
  /// \brief Computes hash for a given vector
  /// \param v
  /// \return
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

/// @}
