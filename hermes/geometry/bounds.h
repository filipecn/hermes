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

/// \file   bounds.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2017-08-19
/// \brief  Geometric bounds.

#pragma once

#include "hermes/base/index.h"
#include "hermes/core/types.h"
#include <hermes/geometry/point.h>

#include <algorithm>

namespace hermes::geo::bounds {

// *****************************************************************************
//                                                                BoundingBox1
// *****************************************************************************

/// Axis-aligned region of space.
/// \tparam T coordinates type
template <typename T> class BoundingBox1 {
public:
  HERMES_DEVICE_CALLABLE static BoundingBox1 Unit() {
    return BoundingBox1<T>(0, 1);
  }

  HERMES_DEVICE_CALLABLE BoundingBox1() {
    lower = math::numbers::greatest<T>();
    upper = math::numbers::lowest<T>();
  }
  HERMES_DEVICE_CALLABLE explicit BoundingBox1(const T &p)
      : lower(p), upper(p) {}
  HERMES_DEVICE_CALLABLE BoundingBox1(const T &p1, const T &p2)
      : lower(std::min(p1, p2)), upper(std::max(p1, p2)) {}
  HERMES_DEVICE_CALLABLE bool contains(const T &p) const {
    return p >= lower && p <= upper;
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE real_t size() const {
    return upper - lower;
  }
  HERMES_DEVICE_CALLABLE T extends() const { return upper - lower; }
  HERMES_DEVICE_CALLABLE T center() const {
    return lower + (upper - lower) * 0.5;
  }
  HERMES_DEVICE_CALLABLE T centroid() const { return lower * .5 + upper * .5; }
  HERMES_DEVICE_CALLABLE const T &operator[](int i) const {
    return (&lower)[i];
  }
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&lower)[i]; }

  T lower, upper;
};

// *****************************************************************************
//                                                                BoundingBox2
// *****************************************************************************

/// Axis-aligned region of space.
/// \tparam T coordinates type
template <typename T> class BoundingBox2 {
public:
  HERMES_DEVICE_CALLABLE static BoundingBox2<T> Unit() {
    return {Point2<T>(), Point2<T>(1, 1)};
  }

  HERMES_DEVICE_CALLABLE BoundingBox2() {
    lower = Point2<T>(math::numbers::greatest<T>());
    upper = Point2<T>(math::numbers::lowest<T>());
  }
  HERMES_DEVICE_CALLABLE explicit BoundingBox2(const Point2<T> &p)
      : lower(p), upper(p) {}
  HERMES_DEVICE_CALLABLE BoundingBox2(const Point2<T> &p1,
                                      const Point2<T> &p2) {
#ifdef HERMES_DEVICE_ENABLED
    lower = Point2<T>(fminf(p1.x, p2.x), fminf(p1.y, p2.y));
    upper = Point2<T>(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y));
#else
    lower = Point2<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
    upper = Point2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
#endif
  }
  template <typename U>
  HERMES_DEVICE_CALLABLE BoundingBox2(const Index2<U>::Range &range)
      : lower{range.lower()}, upper{range.upper() - Index2<U>(1, 1)} {}

  template <typename U>
  HERMES_DEVICE_CALLABLE BoundingBox2 &
  operator=(const Index2<U>::Range &range) {
    lower = range.lower();
    upper = range.upper();
    return *this;
  }

  template <typename U>
  HERMES_DEVICE_CALLABLE explicit operator typename Index2<U>::Range() const {
    return Index2<U>::Range(lower, Index2<U>(upper.x + 1, upper.y + 1));
  }

  HERMES_DEVICE_CALLABLE const Point2<T> &operator[](int i) const {
    return (i == 0) ? lower : upper;
  }
  HERMES_DEVICE_CALLABLE Point2<T> &operator[](int i) {
    return (i == 0) ? lower : upper;
  }

#define ARITHMETIC_OP(OP, O)                                                   \
  HERMES_DEVICE_CALLABLE BoundingBox2 &operator OP##=(const O & o) {           \
    *this = make_union(*this, o);                                              \
    return *this;                                                              \
  }                                                                            \
  HERMES_DEVICE_CALLABLE BoundingBox2 operator OP(const O &o) {                \
    return make_union(*this, o);                                               \
  }
  ARITHMETIC_OP(+, BoundingBox2)
  ARITHMETIC_OP(+, Point2<T>)
#undef ARITHMETIC_OP

  HERMES_DEVICE_CALLABLE bool operator==(const BoundingBox2 &b) const {
    return lower == b.lower && upper == b.upper;
  }

  HERMES_NODISCARD HERMES_DEVICE_CALLABLE bool
  contains(const Point2<T> &p) const {
    return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y &&
            p.y <= upper.y);
  }

  HERMES_NODISCARD HERMES_DEVICE_CALLABLE real_t size(int d) const {
#ifdef HERMES_DEVICE_ENABLED
    d = fmaxf(0, fminf(1, d));
#else
    d = std::max(0, std::min(1, d));
#endif
    return upper[d] - lower[d];
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Vector2<T> extends() const {
    return upper - lower;
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Point2<T> center() const {
    return lower + (upper - lower) * .5f;
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Point2<T> centroid() const {
    return lower * .5f + vec2(upper * .5f);
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE int maxExtent() const {
    Vector2<T> diag = upper - lower;
    if (diag.x > diag.y)
      return 0;
    return 1;
  }

  Point2<T> lower, upper;
};

// *****************************************************************************
//                                                                BoundingBox3
// *****************************************************************************

/// Axis-aligned region of space.
/// \tparam T coordinates type
template <typename T> class BoundingBox3 {
public:
  HERMES_DEVICE_CALLABLE static BoundingBox3
  unitBox(bool centroid_center = false) {
    if (centroid_center)
      return {Point3<T>(-0.5), Point3<T>(0.5)};
    return {Point3<T>(), Point3<T>(1, 1, 1)};
  }

  /// Creates an empty bounding box
  HERMES_DEVICE_CALLABLE BoundingBox3() {
    lower = Point3<T>(math::numbers::greatest<T>());
    upper = Point3<T>(math::numbers::lowest<T>());
  }
  /// Creates a bounding enclosing a single point
  /// \param p point
  HERMES_DEVICE_CALLABLE explicit BoundingBox3(const Point3<T> &p)
      : lower(p), upper(p) {}
  /// Creates a bounding box of 2r side centered at c
  /// \param c center point
  /// \param r radius
  HERMES_DEVICE_CALLABLE BoundingBox3(const Point3<T> &c, real_t r) {
    lower = c - Vector3<T>(r, r, r);
    upper = c + Vector3<T>(r, r, r);
  }
  /// Creates a bounding box enclosing two points
  /// \param p1 first point
  /// \param p2 second point
  HERMES_DEVICE_CALLABLE BoundingBox3(const Point3<T> &p1,
                                      const Point3<T> &p2) {
#ifdef HERMES_DEVICE_ENABLED
    lower = Point3<T>(fminf(p1.x, p2.x), fminf(p1.y, p2.y), fminf(p1.z, p2.z));
    upper = Point3<T>(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y), fmaxf(p1.z, p2.z));
#else
    lower = Point3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
                      std::min(p1.z, p2.z));
    upper = Point3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
                      std::max(p1.z, p2.z));
#endif
  }

  /// \param p
  /// \return true if this bounding box encloses **p**
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE bool
  contains(const Point3<T> &p) const {
    return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y &&
            p.y <= upper.y && p.z >= lower.z && p.z <= upper.z);
  }
  /// \param b bbox
  /// \return true if bbox is fully inside
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE bool
  contains(const BoundingBox3 &b) const {
    return contains(b.lower) && contains(b.upper);
  }
  /// Doesn't consider points on the upper boundary to be inside the bbox
  /// \param p point
  /// \return true if contains exclusive
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE bool
  containsExclusive(const Point3<T> &p) const {
    return (p.x >= lower.x && p.x < upper.x && p.y >= lower.y &&
            p.y < upper.y && p.z >= lower.z && p.z < upper.z);
  }

  /// Pads the bbox in both dimensions
  /// \param delta expansion factor (lower - delta, upper + delta)
  HERMES_DEVICE_CALLABLE void expand(real_t delta) {
    lower -= Vector3<T>(delta, delta, delta);
    upper += Vector3<T>(delta, delta, delta);
  }
  /// \return vector along the diagonal upper - lower
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Vector3<T> diagonal() const {
    return upper - lower;
  }
  /// \return index of longest axis
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE int maxExtent() const {
    Vector3<T> diag = upper - lower;
    if (diag.x > diag.y && diag.x > diag.z)
      return 0;
    else if (diag.y > diag.z)
      return 1;
    return 2;
  }
  /// \param p point
  /// \return position of **p** relative to the corners where lower has offset
  /// (0,0,0) and upper (1,1,1)
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Vector3<T>
  offset(const Point3<T> &p) const {
    hermes::geo::Vector3<T> o = p - lower;
    if (upper.x > lower.x)
      o.x /= upper.x - lower.x;
    if (upper.y > lower.y)
      o.y /= upper.y - lower.y;
    if (upper.z > lower.z)
      o.z /= upper.z - lower.z;
    return o;
  }
  /// \return surface area of the six faces
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE T surfaceArea() const {
    Vector3<T> d = upper - lower;
    return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
  }
  /// \return volume inside the bounds
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE T volume() const {
    Vector3<T> d = upper - lower;
    return d.x * d.y * d.z;
  }
  /**
   * y
   * |_ x
   * z
   *   /  2  /  3 /
   *  / 6  /  7  /
   *  ------------
   * |   0 |   1 |
   * | 4   | 5   |
   * ------------ */
  HERMES_NODISCARD std::vector<BoundingBox3> splitBy8() const {
    auto mid = center();
    std::vector<BoundingBox3<T>> children;
    children.emplace_back(lower, mid);
    children.emplace_back(Point3<T>(mid.x, lower.y, lower.z),
                          Point3<T>(upper.x, mid.y, mid.z));
    children.emplace_back(Point3<T>(lower.x, mid.y, lower.z),
                          Point3<T>(mid.x, upper.y, mid.z));
    children.emplace_back(Point3<T>(mid.x, mid.y, lower.z),
                          Point3<T>(upper.x, upper.y, mid.z));
    children.emplace_back(Point3<T>(lower.x, lower.y, mid.z),
                          Point3<T>(mid.x, mid.y, upper.z));
    children.emplace_back(Point3<T>(mid.x, lower.y, mid.z),
                          Point3<T>(upper.x, mid.y, upper.z));
    children.emplace_back(Point3<T>(lower.x, mid.y, mid.z),
                          Point3<T>(mid.x, upper.y, upper.z));
    children.emplace_back(Point3<T>(mid.x, mid.y, mid.z),
                          Point3<T>(upper.x, upper.y, upper.z));
    return children;
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Point3<T> center() const {
    return lower + (upper - lower) * .5f;
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Point3<T> centroid() const {
    return lower * .5f + vec3(upper * .5f);
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE T size(u32 d) const {
    return upper[d] - lower[d];
  }

  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  HERMES_DEVICE_CALLABLE const Point3<T> &operator[](int i) const {
    return (i == 0) ? lower : upper;
  }
  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  HERMES_DEVICE_CALLABLE Point3<T> &operator[](int i) {
    return (i == 0) ? lower : upper;
  }
  /// \param c corner index
  /// \return corner point
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Point3<T> corner(int c) const {
    return Point3<T>((*this)[(c & 1)].x, (*this)[(c & 2) ? 1 : 0].y,
                     (*this)[(c & 4) ? 1 : 0].z);
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE BoundingBox2<T> xy() const {
    return BoundingBox2<T>(lower.xy(), upper.xy());
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE BoundingBox2<T> yz() const {
    return BoundingBox2<T>(lower.yz(), upper.yz());
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE BoundingBox2<T> xz() const {
    return BoundingBox2<T>(lower.xz(), upper.xz());
  }

  Point3<T> lower, upper;
};

// *****************************************************************************
//                                                             BoundingSphere3
// *****************************************************************************

/// Bounding sphere.
/// \tparam T coordinates type.
template <typename T> struct BoundingSphere3 {

  HERMES_DEVICE_CALLABLE static BoundingSphere3 Unit() {
    return {{0, 0, 0}, 1};
  }

  HERMES_DEVICE_CALLABLE BoundingSphere3() = default;
  HERMES_DEVICE_CALLABLE ~BoundingSphere3() = default;

  HERMES_DEVICE_CALLABLE BoundingSphere3 &setCenter(const Point3<T> &p) {
    center = p;
    return *this;
  }

  HERMES_DEVICE_CALLABLE BoundingSphere3 &setRadius(T r) {
    radius = r;
    return *this;
  }

#define ARITHMETIC_OP(OP, O)                                                   \
  HERMES_DEVICE_CALLABLE BoundingSphere3 &operator OP##=(const O & o) {        \
    *this = make_union(*this, o);                                              \
    return *this;                                                              \
  }                                                                            \
  HERMES_DEVICE_CALLABLE BoundingSphere3 operator OP(const O &o) {             \
    return make_union(*this, o);                                               \
  }
  ARITHMETIC_OP(+, BoundingSphere3)
  ARITHMETIC_OP(+, Point3<T>)
#undef ARITHMETIC_OP

  BoundingBox3<T> extents() const { return BoundingBox3<T>(center, radius); }

  Point3<T> center;
  T radius{-1};
};

template <typename T>
HERMES_DEVICE_CALLABLE BoundingBox1<T> make_union(const BoundingBox1<T> &b,
                                                  const T &p) {
  BoundingBox1 ret = b;
  ret.lower = std::min(b.lower, p);
  ret.upper = std::max(b.upper, p);
  return ret;
}
template <typename T>
HERMES_DEVICE_CALLABLE BoundingBox1<T> make_union(const BoundingBox1<T> &a,
                                                  const BoundingBox1<T> &b) {
  BoundingBox1 ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}

template <typename T>
HERMES_DEVICE_CALLABLE inline BoundingBox2<T>
make_union(const BoundingBox2<T> &b, const Point2<T> &p) {
  BoundingBox2<T> ret = b;
#ifdef HERMES_DEVICE_ENABLED
  ret.lower.x = fminf(b.lower.x, p.x);
  ret.lower.y = fminf(b.lower.y, p.y);
  ret.upper.x = fmaxf(b.upper.x, p.x);
  ret.upper.y = fmaxf(b.upper.y, p.y);
#else
  ret.lower.x = std::min(b.lower.x, p.x);
  ret.lower.y = std::min(b.lower.y, p.y);
  ret.upper.x = std::max(b.upper.x, p.x);
  ret.upper.y = std::max(b.upper.y, p.y);
#endif
  return ret;
}

template <typename T>
HERMES_DEVICE_CALLABLE inline BoundingBox2<T>
make_union(const BoundingBox2<T> &a, const BoundingBox2<T> &b) {
  BoundingBox2<T> ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}

/// Checks if both bounding boxes overlap
/// \param a first bounding box
/// \param b second bounding box
/// \return true if they overlap
template <typename T>
HERMES_DEVICE_CALLABLE bool overlaps(const BoundingBox3<T> &a,
                                     const BoundingBox3<T> &b) {
  bool x = (a.upper.x >= b.lower.x) && (a.lower.x <= b.upper.x);
  bool y = (a.upper.y >= b.lower.y) && (a.lower.y <= b.upper.y);
  bool z = (a.upper.z >= b.lower.z) && (a.lower.z <= b.upper.z);
  return (x && y && z);
}
/// \tparam T coordinates type
/// \param b bounding box
/// \param p point
/// \return a new bounding box that encompasses **b** and **p**
template <typename T>
HERMES_DEVICE_CALLABLE BoundingBox3<T> make_union(const BoundingBox3<T> &b,
                                                  const Point3<T> &p) {
  BoundingBox3<T> ret = b;
#ifdef HERMES_DEVICE_ENABLED
  ret.lower.x = fminf(b.lower.x, p.x);
  ret.lower.y = fminf(b.lower.y, p.y);
  ret.lower.z = fminf(b.lower.z, p.z);
  ret.upper.x = fmaxf(b.upper.x, p.x);
  ret.upper.y = fmaxf(b.upper.y, p.y);
  ret.upper.z = fmaxf(b.upper.z, p.z);
#else
  ret.lower.x = std::min(b.lower.x, p.x);
  ret.lower.y = std::min(b.lower.y, p.y);
  ret.lower.z = std::min(b.lower.z, p.z);
  ret.upper.x = std::max(b.upper.x, p.x);
  ret.upper.y = std::max(b.upper.y, p.y);
  ret.upper.z = std::max(b.upper.z, p.z);
#endif
  return ret;
}
/// \tparam T coordinates type
/// \param a bounding box
/// \param b bounding box
/// \return a new bounding box that encompasses **a** and **b**
template <typename T>
HERMES_DEVICE_CALLABLE inline BoundingBox3<T>
make_union(const BoundingBox3<T> &a, const BoundingBox3<T> &b) {
  BoundingBox3<T> ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}
/// \tparam T coordinates type
/// \param a bounding sphere
/// \param b bounding sphere
/// \return a new bounding sphere that encompasses **a** and **b**
template <typename T>
HERMES_DEVICE_CALLABLE inline BoundingSphere3<T>
make_union(const BoundingSphere3<T> &a, const BoundingSphere3<T> &b) {
  auto ab_dist = hermes::geo::distance(a.center, b.center);
  if (ab_dist + a.radius < b.radius)
    return b;
  if (ab_dist + b.radius < a.radius)
    return a;
  auto radius = 0.5 * (a.radius + b.radius + ab_dist);
  return {.radius = radius,
          .center =
              a.center + (b.center - a.center) * (radius - a.radius) / ab_dist};
}

/// \tparam T coordinates type
/// \param a bounding box
/// \param b bounding box
/// \return a new bbox resulting from the intersection of **a** and **b**
template <typename T>
HERMES_DEVICE_CALLABLE BoundingBox3<T> intersect(const BoundingBox3<T> &a,
                                                 const BoundingBox3<T> &b) {
#ifdef HERMES_DEVICE_ENABLED
  return BoundingBox3<T>(
      Point3<T>(max(a.lower.x, b.lower.x), max(a.lower.x, b.lower.y),
                max(a.lower.z, b.lower.z)),
      Point3<T>(min(a.upper.x, b.upper.x), min(a.upper.x, b.upper.y),
                min(a.upper.z, b.upper.z)));
#else
  return BoundingBox3<T>(
      Point3<T>(std::max(a.lower.x, b.lower.x), std::max(a.lower.x, b.lower.y),
                std::max(a.lower.z, b.lower.z)),
      Point3<T>(std::min(a.upper.x, b.upper.x), std::min(a.upper.x, b.upper.y),
                std::min(a.upper.z, b.upper.z)));
#endif
}

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(BoundingBox1<T>, typename T)
HERMES_PUSH_DEBUG_CUSTOM_FIELD("BBox[{}, {}]", object.lower, object.upper);
HERMES_TO_STRING_DEBUG_METHOD_END

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(BoundingBox2<T>, typename T)
HERMES_PUSH_DEBUG_CUSTOM_FIELD("BBox[{}, {}]", hermes::to_string(object.lower),
                               hermes::to_string(object.upper));
HERMES_TO_STRING_DEBUG_METHOD_END

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(BoundingBox3<T>, typename T)
HERMES_PUSH_DEBUG_CUSTOM_FIELD("BBox[{}, {}]", hermes::to_string(object.lower),
                               hermes::to_string(object.upper));
HERMES_TO_STRING_DEBUG_METHOD_END

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(BoundingSphere3<T>, typename T)
HERMES_PUSH_DEBUG_CUSTOM_FIELD("BSphere[{}, {}]",
                               hermes::to_string(object.center), object.radius);
HERMES_TO_STRING_DEBUG_METHOD_END

typedef BoundingBox1<real_t> bbox1;
typedef BoundingBox2<real_t> bbox2;
typedef BoundingBox3<real_t> bbox3;
typedef BoundingBox3<float> bbox3f;

typedef BoundingSphere3<real_t> bsphere3;

} // namespace hermes::geo::bounds
