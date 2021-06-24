/*
 * Copyright (c) 2019 FilipeCN
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

#ifndef HERMES_GEOMETRY_CUDA_BBOX_H
#define HERMES_GEOMETRY_CUDA_BBOX_H

#include <hermes/geometry/point.h>

#include <algorithm>
#include <iostream>

namespace hermes {

template<typename T> class BBox2 {
public:
  // ***********************************************************************
  //                           STATIC METHODS
  // ***********************************************************************
  static __host__ __device__ BBox2<T> unitBox() {
    return {Point2<T>(), Point2<T>(1, 1)};
  }
  /// \param b
  /// \param p
  /// \return
  static __host__ __device__ inline BBox2<T> make_union(const BBox2<T> &b,
                                                        const Point2<T> &p) {
    BBox2<T> ret = b;
    ret.lower.x = std::min(b.lower.x, p.x);
    ret.lower.y = std::min(b.lower.y, p.y);
    ret.upper.x = std::max(b.upper.x, p.x);
    ret.upper.y = std::max(b.upper.y, p.y);
    return ret;
  }
  /// \param a
  /// \param b
  /// \return
  static __host__ __device__ inline BBox2<T> make_union(const BBox2<T> &a,
                                                        const BBox2<T> &b) {
    BBox2<T> ret = make_union(a, b.lower);
    return make_union(ret, b.upper);
  }
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  ///
  __host__ __device__ BBox2() {
    lower = Point2<T>(ponos::Constants::greatest<T>());
    upper = Point2<T>(ponos::Constants::lowest<T>());
  }
  /// \param p
  __host__ __device__ explicit BBox2(const Point2<T> &p) : lower(p), upper(p) {}
  /// \param p1
  /// \param p2
  __host__ __device__ BBox2(const Point2<T> &p1, const Point2<T> &p2) {
    lower = Point2<T>(fminf(p1.x, p2.x), fminf(p1.y, p2.y));
    upper = Point2<T>(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y));
  }
  // ***********************************************************************
  //                           QUERIES
  // ***********************************************************************
  /// \param p
  /// \return
  __host__ __device__ bool contains(const Point2<T> &p) const {
    return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y && p.y <= upper.y);
  }
  // ***********************************************************************
  //                           METRICS
  // ***********************************************************************
  /// \param d
  /// \return
  __host__ __device__ T size(int d) const {
    d = fmaxf(0, fminf(1, d));
    return upper[d] - lower[d];
  }
  /// \return
  [[nodiscard]] __host__ __device__ int maxExtent() const {
    Vector2<T> diag = upper - lower;
    if (diag.x > diag.y)
      return 0;
    return 1;
  }
  ///
  /// \return
  __host__ __device__ Vector2<T> extends() const {
    return upper - lower;
  }
  // ***********************************************************************
  //                           GEOMETRY
  // ***********************************************************************
  /// \return
  __host__ __device__ Point2<T> center() const {
    return lower + (upper - lower) * .5f;
  }
  /// \return
  __host__ __device__ Point2<T> centroid() const {
    return lower * .5f + vec2(upper * .5f);
  }
  // ***********************************************************************
  //                           FIELDS
  // ***********************************************************************
  /// \param i
  /// \return
  __host__ __device__ const Point2<T> &operator[](int i) const {
    return (i == 0) ? lower : upper;
  }
  /// \param i
  /// \return
  __host__ __device__ Point2<T> &operator[](int i) {
    return (i == 0) ? lower : upper;
  }

  Point2<T> lower; //!<
  Point2<T> upper; //!<
};

typedef BBox2<float> bbox2;

/// Axis-aligned region of space.
/// \tparam T coordinates type
template<typename T> class BBox3 {
public:
  // ***********************************************************************
  //                           STATIC METHODS
  // ***********************************************************************
  ///
  /// \return
  static __host__ __device__  BBox3 unitBox() {
    return {Point3<T>(), Point3<T>(1, 1, 1)};
  }
  /// \param b bounding box
  /// \param p point
  /// \return a new bounding box that encopasses **b** and **p**
  static __host__ __device__ BBox3 make_union(const BBox3 &b, const Point3<T> &p) {
    BBox3<T> ret = b;
    ret.lower.x = fminf(b.lower.x, p.x);
    ret.lower.y = fminf(b.lower.y, p.y);
    ret.lower.z = fminf(b.lower.z, p.z);
    ret.upper.x = fmaxf(b.upper.x, p.x);
    ret.upper.y = fmaxf(b.upper.y, p.y);
    ret.upper.z = fmaxf(b.upper.z, p.z);
    return ret;
  }
  /// \param a bounding box
  /// \param b bounding box
  /// \return a new bounding box that encompasses **a** and **b**
  static __host__ __device__ BBox3<T> make_union(const BBox3 &a, const BBox3 &b) {
    BBox3<T> ret = make_union(a, b.lower);
    return make_union(ret, b.upper);
  }
  /// Checks if both bounding boxes overlap
  /// \param a first bounding box
  /// \param b second bounding box
  /// \return true if they overlap
  __host__ __device__ bool overlaps(const BBox3 &a, const BBox3 &b) {
    bool x = (a.upper.x >= b.lower.x) && (a.lower.x <= b.upper.x);
    bool y = (a.upper.y >= b.lower.y) && (a.lower.y <= b.upper.y);
    bool z = (a.upper.z >= b.lower.z) && (a.lower.z <= b.upper.z);
    return (x && y && z);
  }
  /// \param a bounding box
  /// \param b bounding box
  /// \return a new bbox resulting from the intersection of **a** and **b**
  __host__ __device__ BBox3 intersect(const BBox3 &a, const BBox3 &b) {
    return BBox3<T>(
        Point3<T>(fmaxf(a.lower.x, b.lower.x), fmaxf(a.lower.x, b.lower.y),
                  fmaxf(a.lower.z, b.lower.z)),
        Point3<T>(fminf(a.lower.x, b.lower.x), fminf(a.lower.x, b.lower.y),
                  fminf(a.lower.z, b.lower.z)));
  }
  // ***********************************************************************
  //                           CONSTRUCTORS
  // ***********************************************************************
  /// Creates an empty bounding box
  __host__ __device__ BBox3() {
    lower = Point3<T>(ponos::Constants::greatest<T>());
    upper = Point3<T>(ponos::Constants::lowest<T>());
  }
  /// Creates a bounding enclosing a single point
  /// \param p point
  __host__ __device__ explicit BBox3(const Point3<T> &p) : lower(p), upper(p) {}
  /// Creates a bounding box of 2r side centered at c
  /// \param c center point
  /// \param r radius
  __host__ __device__ BBox3(const Point3<T> &c, T r) {
    lower = c - Vector3<T>(r, r, r);
    upper = c + Vector3<T>(r, r, r);
  }
  /// Creates a bounding box enclosing two points
  /// \param p1 first point
  /// \param p2 second point
  __host__ __device__ BBox3(const Point3<T> &p1, const Point3<T> &p2) {
    lower = Point3<T>(fminf(p1.x, p2.x), fminf(p1.y, p2.y), fminf(p1.z, p2.z));
    upper = Point3<T>(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y), fmaxf(p1.z, p2.z));
  }
  // ***********************************************************************
  //                           QUERIES
  // ***********************************************************************
  /// \param p
  /// \return true if this bounding box encloses **p**
  [[nodiscard]] __host__ __device__ bool contains(const Point3<T> &p) const {
    return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y &&
        p.y <= upper.y && p.z >= lower.z && p.z <= upper.z);
  }
  /// \param b bbox
  /// \return true if bbox is fully inside
  [[nodiscard]] __host__ __device__ bool contains(const BBox3 &b) const {
    return contains(b.lower) && contains(b.upper);
  }
  /// Doesn't consider points on the upper boundary to be inside the bbox
  /// \param p point
  /// \return true if contains exclusive
  [[nodiscard]] __host__ __device__ bool containsExclusive(const Point3<T> &p) const {
    return (p.x >= lower.x && p.x < upper.x && p.y >= lower.y && p.y < upper.y &&
        p.z >= lower.z && p.z < upper.z);
  }
  // ***********************************************************************
  //                           METRICS
  // ***********************************************************************
  /// \return index of longest axis
  [[nodiscard]] __host__ __device__ int maxExtent() const {
    Vector3<T> diag = upper - lower;
    if (diag.x > diag.y && diag.x > diag.z)
      return 0;
    else if (diag.y > diag.z)
      return 1;
    return 2;
  }
  [[nodiscard]] __host__ __device__ T size(size_t d) const {
    return upper[d] - lower[d];
  }
  /// \return surface area of the six faces
  [[nodiscard]] __host__ __device__ T surfaceArea() const {
    Vector3<T> d = upper - lower;
    return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
  }
  /// \return volume inside the bounds
  [[nodiscard]] __host__ __device__ T volume() const {
    Vector3<T> d = upper - lower;
    return d.x * d.y * d.z;
  }
  // ***********************************************************************
  //                           GEOMETRY
  // ***********************************************************************
  /// \return
  [[nodiscard]] __host__ __device__ BBox2<T> xy() const {
    return BBox2<T>(lower.xy(), upper.xy());
  }
  /// \return
  [[nodiscard]] __host__ __device__ BBox2<T> yz() const {
    return BBox2<T>(lower.yz(), upper.yz());
  }
  /// \return
  [[nodiscard]] __host__ __device__ BBox2<T> xz() const {
    return BBox2<T>(lower.xz(), upper.xz());
  }
  /// \return vector along the diagonal upper - lower
  [[nodiscard]] __host__ __device__ Vector3<T> diagonal() const {
    return upper - lower;
  }
  /// \param c corner index
  /// \return corner point
  [[nodiscard]] __host__ __device__ Point3<T> corner(int c) const {
    return Point3<T>((*this)[(c & 1)].x, (*this)[(c & 2) ? 1 : 0].y,
                     (*this)[(c & 4) ? 1 : 0].z);
  }
  /// \param p point
  /// \return position of **p** relative to the corners where lower has offset
  /// (0,0,0) and upper (1,1,1)
  [[nodiscard]] __host__ __device__ Vector3<T> offset(const Point3<T> &p) const {
    Vector3<T> o = p - lower;
    if (upper.x > lower.x)
      o.x /= upper.x - lower.x;
    if (upper.y > lower.y)
      o.y /= upper.y - lower.y;
    if (upper.z > lower.z)
      o.z /= upper.z - lower.z;
    return o;
  }
  [[nodiscard]] __host__ __device__ Point3<T> center() const {
    return lower + (upper - lower) * .5f;
  }
  ///
  /// \return
  [[nodiscard]] __host__ __device__ Point3<T> centroid() const {
    return lower * .5f + vec3(upper * .5f);
  }
  /// Pads the bbox in both dimensions
  /// \param delta expansion factor (lower - delta, upper + delta)
  __host__ __device__ void expand(T delta) {
    lower -= Vector3<T>(delta, delta, delta);
    upper += Vector3<T>(delta, delta, delta);
  }
  // ***********************************************************************
  //                           FIELDS
  // ***********************************************************************
  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  __host__ __device__ const Point3<T> &operator[](int i) const {
    return (i == 0) ? lower : upper;
  }
  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  __host__ __device__ Point3<T> &operator[](int i) {
    return (i == 0) ? lower : upper;
  }

  Point3<T> lower; //!<
  Point3<T> upper; //!<
};

using bbox2 = BBox2<float>;
using bbox2f = BBox2<float>;
using bbox3 = BBox3<float>;
using bbox3f = BBox3<float>;

} // namespace hermes

#endif
