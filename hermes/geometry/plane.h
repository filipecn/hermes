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

/// \file   line.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2017-08-19
/// Geometric plane classes

#pragma once

#include <hermes/geometry/normal.h>
#include <hermes/geometry/point.h>
#include <hermes/geometry/vector.h>
#include <type_traits>

namespace hermes::geo {

/** Implements the equation normal X = offset.
 */
template <typename T> class Plane3 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value,
                "Plane3 supports only f32 and f64");

public:
  using scalar_type = T;
  using normal_type = Normal3<T>;
  using point_type = Point3<T>;
  using vector_type = Vector3<T>;
  /// default_color constructor
  HERMES_CPU_GPU Plane3() = default;
  /** Constructor
   * \param n **[in]** normal
   * \param o **[in]** offset
   */
  HERMES_CPU_GPU Plane3(normal_type n, T o) : normal(n), offset(o) {}
  HERMES_CPU_GPU Plane3(normal_type n, point_type p)
      : normal(n), offset(dot(n, static_cast<vector_type>(p))) {}
  /// \param invert_normal
  /// \return a plane with offset = 0 and normal(0,0,1)
  HERMES_CPU_GPU static Plane3 XY(bool invert_normal = false) {
    return {normal_type(0, 0, invert_normal ? T(-1) : T(1)), T(0)};
  }
  /// \param invert_normal
  /// \return a plane with offset = 0 and normal(0,1,0)
  HERMES_CPU_GPU static Plane3 XZ(bool invert_normal = false) {
    return {normal_type(0, invert_normal ? T(-1) : T(1), 0), T(0)};
  }
  /// \param invert_normal
  /// \return a plane with offset = 0 and normal(1,0,0)
  HERMES_CPU_GPU static Plane3 YZ(bool invert_normal = false) {
    return {normal_type(invert_normal ? T(-1) : T(1), 0, 0), T(0)};
  }

  HERMES_NODISCARD HERMES_CPU_GPU point_type
  closestPoint(const point_type &p) const {
    vector_type n(normal);
    T t = (dot(n, vector_type(p)) - offset) / n.length2();
    return p - t * n;
  }
  /** \brief  projects **v** on plane
   * \param v
   * \returns projected **v**
   */
  HERMES_NODISCARD HERMES_CPU_GPU vector_type
  project(const vector_type &v) const {
    return hermes::geo::project(v, normal);
  }
  /** \brief  reflects **v** fron plane
   * \param v
   * \returns reflected **v**
   */
  HERMES_NODISCARD HERMES_CPU_GPU vector_type
  reflect(const vector_type &v) const {
    return hermes::geo::reflect(v, normal);
  }
  HERMES_NODISCARD HERMES_CPU_GPU bool
  isOnNormalSide(const point_type &p) const {
    return dot(vector_type(normal), p - closestPoint(p)) >= T(0);
  }

  normal_type normal;
  T offset{0};
};

using plane3 = Plane3<real_t>;
using plane3f = Plane3<f32>;
using plane3d = Plane3<f64>;

} // namespace hermes::geo

namespace hermes {

HERMES_TYPE_LAYOUT_METHODS(geo::Plane3<real_t>, real_t, 4)

#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
HERMES_TYPE_LAYOUT_METHODS(geo::Plane3<f32>, f32, 4)
#else
HERMES_TYPE_LAYOUT_METHODS(geo::Plane3<f64>, f64, 4)
#endif

template <typename T> struct DebugTraits<geo::Plane3<T>> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Plane3<T> &data) {
    return DebugMessage("Plane3[n={} o={}]", hermes::to_string(data.normal),
                        data.offset);
  }
};

} // namespace hermes

///** Implements the equation normal X = offset.
// */
// class ImplicitPlane2D : public ImplicitCurveInterface {
// public:
//  /// default_color constructor
//  ImplicitPlane2D() { offset = 0.f; }
//  /** Constructor
//   * \param n **[in]** normal
//   * \param o **[in]** offset
//   */
//  ImplicitPlane2D(normal2 n, real_t o) {
//    normal = n;
//    offset = o;
//  }
//  ImplicitPlane2D(point2 p, normal2 n) {
//    normal = n;
//    offset = dot(vec2(normal), vec2(p));
//  }
//  /** \brief  projects **v** on plane
//   * \param v
//   * \returns projected **v**
//   */
//  HERMES_NODISCARD vec2 project(const vec2 &v) const {
//    return hermes::project(v, normal);
//  }
//  /** \brief  reflects **v** fron plane
//   * \param v
//   * \returns reflected **v**
//   */
//  HERMES_NODISCARD vec2 reflect(const vec2 &v) const {
//    return hermes::reflect(v, normal);
//  }
//  HERMES_NODISCARD point2 closestPoint(const point2 &p) const override {
//    real_t t = (dot(vec2(normal), vec2(p)) - offset) / vec2(normal).length2();
//    return p - t * vec2(normal);
//  }
//  HERMES_NODISCARD normal2 closestNormal(const point2 &p) const override {
//    if (dot(vec2(normal), vec2(p)) < 0.f)
//      return -normal;
//    return normal;
//  }
//  HERMES_NODISCARD bbox2 boundingBox() const override { return bbox2(); }
//  void closestIntersection(const Ray2 &r,
//                           CurveRayIntersection *i) const override {
//    HERMES_UNUSED_VARIABLE(r);
//    HERMES_UNUSED_VARIABLE(i);
//  }
//  HERMES_NODISCARD double signedDistance(const point2 &p) const override {
//    return (dot(vec2(p), vec2(normal)) - offset) / vec2(normal).length();
//  }
//
//  friend std::ostream &operator<<(std::ostream &os, const ImplicitPlane2D &p)
//  {
//    os << "[Plane3] offset " << p.offset << " " << p.normal;
//    return os;
//  }
//
//  normal2 normal;
//  real_t offset;
//};
