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

namespace hermes::geo {

/** Implements the equation normal X = offset.
 */
class Plane {
public:
  /// default_color constructor
  Plane();
  /** Constructor
   * \param n **[in]** normal
   * \param o **[in]** offset
   */
  Plane(normal3 n, real_t o);
  Plane(normal3 n, point3 p);
  /// \param invert_normal
  /// \return a plane with offset = 0 and normal(0,0,1)
  static Plane XY(bool invert_normal = false);
  /// \param invert_normal
  /// \return a plane with offset = 0 and normal(0,1,0)
  static Plane XZ(bool invert_normal = false);
  /// \param invert_normal
  /// \return a plane with offset = 0 and normal(1,0,0)
  static Plane YZ(bool invert_normal = false);

  HERMES_NODISCARD point3 closestPoint(const point3 &p) const;
  /** \brief  projects **v** on plane
   * \param v
   * \returns projected **v**
   */
  HERMES_NODISCARD vec3 project(const vec3 &v) const;
  /** \brief  reflects **v** fron plane
   * \param v
   * \returns reflected **v**
   */
  HERMES_NODISCARD vec3 reflect(const vec3 &v) const;
  HERMES_NODISCARD bool isOnNormalSide(const point3 &p) const;

  normal3 normal;
  real_t offset;
};

} // namespace hermes::geo

namespace hermes {

HERMES_TYPE_LAYOUT_METHODS(geo::Plane, real_t, 4)

template <> struct DebugTraits<geo::Plane> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Plane &data) {
    return DebugMessage("Plane[n={} o={}]", hermes::to_string(data.normal),
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
//    os << "[Plane] offset " << p.offset << " " << p.normal;
//    return os;
//  }
//
//  normal2 normal;
//  real_t offset;
//};
