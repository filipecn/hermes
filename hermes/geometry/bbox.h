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

#ifndef HERMES_GEOMETRY_BBOX_H
#define HERMES_GEOMETRY_BBOX_H

#include <hermes/geometry/point.h>

#include <algorithm>
#include <iostream>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                              BBox1
// *********************************************************************************************************************
template<typename T> class BBox1 {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE static BBox1 unitBox() { return BBox1<T>(0, 1); }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE BBox1() {
    lower = Numbers::greatest<T>();
    upper = Numbers::lowest<T>();
  }
  HERMES_DEVICE_CALLABLE explicit BBox1(const T &p) : lower(p), upper(p) {}
  HERMES_DEVICE_CALLABLE BBox1(const T &p1, const T &p2) : lower(std::min(p1, p2)),
                                                           upper(std::max(p1, p2)) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                          QUERIES
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE bool contains(const T &p) const { return p >= lower && p <= upper; }
  // *******************************************************************************************************************
  //                                                                                                         GEOMETRY
  // *******************************************************************************************************************
  [[nodiscard]] HERMES_DEVICE_CALLABLE real_t size() const {
    return upper - lower;
  }
  HERMES_DEVICE_CALLABLE T extends() const { return upper - lower; }
  HERMES_DEVICE_CALLABLE T center() const { return lower + (upper - lower) * 0.5; }
  HERMES_DEVICE_CALLABLE T centroid() const { return lower * .5 + upper * .5; }
  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE const T &operator[](int i) const { return (&lower)[i]; }
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&lower)[i]; }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T lower, upper;
};

// *********************************************************************************************************************
//                                                                                                              BBox2
// *********************************************************************************************************************
template<typename T> class BBox2 {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE static BBox2<T> unitBox() {
    return {Point2<T>(), Point2<T>(1, 1)};
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE BBox2() {
    lower = Point2<T>(Numbers::greatest<T>());
    upper = Point2<T>(Numbers::lowest<T>());
  }
  HERMES_DEVICE_CALLABLE explicit BBox2(const Point2 <T> &p) : lower(p), upper(p) {}
  HERMES_DEVICE_CALLABLE BBox2(const Point2 <T> &p1, const Point2 <T> &p2) {
#ifdef HERMES_DEVICE_ENABLED
    lower = Point2<T>(fminf(p1.x, p2.x), fminf(p1.y, p2.y));
    upper = Point2<T>(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y));
#else
    lower = Point2<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
    upper = Point2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
#endif
  }
  template<typename U>
  HERMES_DEVICE_CALLABLE BBox2(const Index2Range <U> &range) :
      lower{range.lower()}, upper{range.upper() - Index2<U>(1, 1)} {}
// *******************************************************************************************************************
//                                                                                                        OPERATORS
// *******************************************************************************************************************
  //                                                                                                       assignment
  template<typename U>
  HERMES_DEVICE_CALLABLE BBox2 &operator=(const Index2Range <U> &range) {
    lower = range.lower();
    upper = range.upper();
    return *this;
  }
  //                                                                                                          casting
  template<typename U>
  HERMES_DEVICE_CALLABLE explicit operator Index2Range<U>() const {
    return Index2Range<U>(lower, Index2<U>(upper.x + 1, upper.y + 1));
  }
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE const Point2 <T> &operator[](int i) const {
    return (i == 0) ? lower : upper;
  }
  HERMES_DEVICE_CALLABLE Point2 <T> &operator[](int i) { return (i == 0) ? lower : upper; }
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP, O)                                                                                        \
  HERMES_DEVICE_CALLABLE BBox2& operator OP##= (const O& o) { *this = make_union(*this, o); return *this; }         \
  HERMES_DEVICE_CALLABLE BBox2 operator OP (const O& o) { return make_union(*this, o); }
  ARITHMETIC_OP(+, BBox2)
  ARITHMETIC_OP(+, Point2 < T >)
#undef ARITHMETIC_OP
  //                                                                                                       relational
  HERMES_DEVICE_CALLABLE bool operator==(const BBox2 &b) const {
    return lower == b.lower && upper == b.upper;
  }
// *******************************************************************************************************************
//                                                                                                          QUERIES
// *******************************************************************************************************************
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool contains(const Point2 <T> &p) const {
    return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y
        && p.y <= upper.y);
  }
// *******************************************************************************************************************
//                                                                                                         GEOMETRY
// *******************************************************************************************************************
  [[nodiscard]] HERMES_DEVICE_CALLABLE real_t size(int d) const {
#ifdef HERMES_DEVICE_ENABLED
    d = fmaxf(0, fminf(1, d));
#else
    d = std::max(0, std::min(1, d));
#endif
    return upper[d] - lower[d];
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE Vector2 <T> extends() const {
    return upper - lower;
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE Point2 <T> center() const {
    return lower + (upper - lower) * .5f;
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE Point2 <T> centroid() const {
    return lower * .5f + vec2(upper * .5f);
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE int maxExtent() const {
    Vector2<T> diag = upper - lower;
    if (diag.x > diag.y)
      return 0;
    return 1;
  }
// *******************************************************************************************************************
//                                                                                                    PUBLIC FIELDS
// *******************************************************************************************************************
  Point2 <T> lower, upper;
};

// *********************************************************************************************************************
//                                                                                                              BBox3
// *********************************************************************************************************************
/// Axis-aligned region of space.
/// \tparam T coordinates type
template<typename T> class BBox3 {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE static BBox3 unitBox(bool centroid_center = false) {
    if (centroid_center)
      return {Point3<T>(-0.5), Point3<T>(0.5)};
    return {Point3<T>(), Point3<T>(1, 1, 1)};
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// Creates an empty bounding box
  HERMES_DEVICE_CALLABLE BBox3() {
    lower = Point3<T>(Numbers::greatest<T>());
    upper = Point3<T>(Numbers::lowest<T>());
  }
  /// Creates a bounding enclosing a single point
  /// \param p point
  HERMES_DEVICE_CALLABLE explicit BBox3(const Point3 <T> &p) : lower(p), upper(p) {}
  /// Creates a bounding box of 2r side centered at c
  /// \param c center point
  /// \param r radius
  HERMES_DEVICE_CALLABLE BBox3(const Point3 <T> &c, real_t r) {
    lower = c - Vector3<T>(r, r, r);
    upper = c + Vector3<T>(r, r, r);
  }
  /// Creates a bounding box enclosing two points
  /// \param p1 first point
  /// \param p2 second point
  HERMES_DEVICE_CALLABLE BBox3(const Point3 <T> &p1, const Point3 <T> &p2) {
#ifdef HERMES_DEVICE_ENABLED
    lower = Point3<T>(fminf(p1.x, p2.x), fminf(p1.y, p2.y),
                      fminf(p1.z, p2.z));
    upper = Point3<T>(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y),
                      fmaxf(p1.z, p2.z));
#else
    lower = Point3<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
                      std::min(p1.z, p2.z));
    upper = Point3<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
                      std::max(p1.z, p2.z));
#endif
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                          QUERIES
  // *******************************************************************************************************************
  /// \param p
  /// \return true if this bounding box encloses **p**
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool contains(const Point3 <T> &p) const {
    return (p.x >= lower.x && p.x <= upper.x && p.y >= lower.y &&
        p.y <= upper.y && p.z >= lower.z && p.z <= upper.z);
  }
  /// \param b bbox
  /// \return true if bbox is fully inside
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool contains(const BBox3 &b) const {
    return contains(b.lower) && contains(b.upper);
  }
  /// Doesn't consider points on the upper boundary to be inside the bbox
  /// \param p point
  /// \return true if contains exclusive
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool containsExclusive(const Point3 <T> &p) const {
    return (p.x >= lower.x && p.x < upper.x && p.y >= lower.y && p.y < upper.y
        && p.z >= lower.z && p.z < upper.z);
  }
  // *******************************************************************************************************************
  //                                                                                                         GEOMETRY
  // *******************************************************************************************************************
  /// Pads the bbox in both dimensions
  /// \param delta expansion factor (lower - delta, upper + delta)
  HERMES_DEVICE_CALLABLE void expand(real_t delta) {
    lower -= Vector3<T>(delta, delta, delta);
    upper += Vector3<T>(delta, delta, delta);
  }
  /// \return vector along the diagonal upper - lower
  [[nodiscard]] HERMES_DEVICE_CALLABLE Vector3 <T> diagonal() const {
    return upper - lower;
  }
  /// \return index of longest axis
  [[nodiscard]] HERMES_DEVICE_CALLABLE int maxExtent() const {
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
  [[nodiscard]] HERMES_DEVICE_CALLABLE Vector3 <T> offset(const Point3 <T> &p) const {
    hermes::Vector3<T> o = p - lower;
    if (upper.x > lower.x)
      o.x /= upper.x - lower.x;
    if (upper.y > lower.y)
      o.y /= upper.y - lower.y;
    if (upper.z > lower.z)
      o.z /= upper.z - lower.z;
    return o;
  }
  /// \return surface area of the six faces
  [[nodiscard]] HERMES_DEVICE_CALLABLE T surfaceArea() const {
    Vector3<T> d = upper - lower;
    return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
  }
  /// \return volume inside the bounds
  [[nodiscard]] HERMES_DEVICE_CALLABLE T volume() const {
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
  [[nodiscard]] std::vector<BBox3> splitBy8() const {
    auto mid = center();
    std::vector<BBox3 < T>>
    children;
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
  [[nodiscard]] HERMES_DEVICE_CALLABLE Point3 <T> center() const {
    return lower + (upper - lower) * .5f;
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE Point3 <T> centroid() const {
    return lower * .5f + vec3(upper * .5f);
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE T size(u32 d) const {
    return upper[d] - lower[d];
  }
  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  HERMES_DEVICE_CALLABLE const Point3 <T> &operator[](int i) const {
    return (i == 0) ? lower : upper;
  }
  /// \param i 0 = lower, 1 = upper
  /// \return lower or upper point
  HERMES_DEVICE_CALLABLE Point3 <T> &operator[](int i) {
    return (i == 0) ? lower : upper;
  }
  /// \param c corner index
  /// \return corner point
  [[nodiscard]] HERMES_DEVICE_CALLABLE Point3 <T> corner(int c) const {
    return Point3<T>((*this)[(c & 1)].x, (*this)[(c & 2) ? 1 : 0].y,
                     (*this)[(c & 4) ? 1 : 0].z);
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE BBox2<T> xy() const {
    return BBox2<T>(lower.xy(), upper.xy());
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE BBox2<T> yz() const {
    return BBox2<T>(lower.yz(), upper.yz());
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE BBox2<T> xz() const {
    return BBox2<T>(lower.xz(), upper.xz());
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  Point3 <T> lower, upper;
};

// *********************************************************************************************************************
//                                                                                                 EXTERNAL FUNCTIONS
// *********************************************************************************************************************
template<typename T>
HERMES_DEVICE_CALLABLE  BBox1<T> make_union(const BBox1<T> &b, const T &p) {
  BBox1 ret = b;
  ret.lower = std::min(b.lower, p);
  ret.upper = std::max(b.upper, p);
  return ret;
}
template<typename T>
HERMES_DEVICE_CALLABLE  BBox1<T> make_union(const BBox1<T> &a, const BBox1<T> &b) {
  BBox1 ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}

template<typename T>
HERMES_DEVICE_CALLABLE  inline BBox2<T> make_union(const BBox2<T> &b, const Point2 <T> &p) {
  BBox2<T> ret = b;
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

template<typename T>
HERMES_DEVICE_CALLABLE  inline BBox2<T> make_union(const BBox2<T> &a, const BBox2<T> &b) {
  BBox2<T> ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}

/// Checks if both bounding boxes overlap
/// \param a first bounding box
/// \param b second bounding box
/// \return true if they overlap
template<typename T>
HERMES_DEVICE_CALLABLE  bool overlaps(const BBox3<T> &a, const BBox3<T> &b) {
  bool x = (a.upper.x >= b.lower.x) && (a.lower.x <= b.upper.x);
  bool y = (a.upper.y >= b.lower.y) && (a.lower.y <= b.upper.y);
  bool z = (a.upper.z >= b.lower.z) && (a.lower.z <= b.upper.z);
  return (x && y && z);
}
/// \tparam T coordinates type
/// \param b bounding box
/// \param p point
/// \return a new bounding box that encompasses **b** and **p**
template<typename T>
HERMES_DEVICE_CALLABLE  BBox3<T> make_union(const BBox3<T> &b, const Point3 <T> &p) {
  BBox3 <T> ret = b;
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
template<typename T>
HERMES_DEVICE_CALLABLE  inline BBox3<T> make_union(const BBox3<T> &a, const BBox3<T> &b) {
  BBox3 <T> ret = make_union(a, b.lower);
  return make_union(ret, b.upper);
}
/// \tparam T coordinates type
/// \param a bounding box
/// \param b bounding box
/// \return a new bbox resulting from the intersection of **a** and **b**
template<typename T>
HERMES_DEVICE_CALLABLE  BBox3<T> intersect(const BBox3<T> &a, const BBox3<T> &b) {
#ifdef HERMES_DEVICE_ENABLED
  return BBox3<T>(
      Point3<T>(max(a.lower.x, b.lower.x), max(a.lower.x, b.lower.y),
                max(a.lower.z, b.lower.z)),
      Point3<T>(min(a.upper.x, b.upper.x), min(a.upper.x, b.upper.y),
                min(a.upper.z, b.upper.z)));
#else
  return BBox3<T>(
      Point3<T>(std::max(a.lower.x, b.lower.x), std::max(a.lower.x, b.lower.y),
                std::max(a.lower.z, b.lower.z)),
      Point3<T>(std::min(a.upper.x, b.upper.x), std::min(a.upper.x, b.upper.y),
                std::min(a.upper.z, b.upper.z)));
#endif
}
// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
std::ostream &operator<<(std::ostream &os, const BBox1<T> &b) {
  os << "BBox1(" << b.lower << ", " << b.upper << ")";
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const BBox2<T> &b) {
  os << "BBox2(" << b.lower << ", " << b.upper << ")";
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const BBox3<T> &b) {
  os << "BBox3(" << b.lower << ", " << b.upper << ")";
  return os;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
typedef BBox1<real_t> bbox1;
typedef BBox2<real_t> bbox2;
typedef BBox3<real_t> bbox3;
typedef BBox3<float> bbox3f;

} // namespace hermes

#endif
