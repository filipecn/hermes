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
/// Geometric line classes

#pragma once

#include <hermes/geometry/point.h>
#include <hermes/geometry/vector.h>
#include <type_traits>

namespace hermes::geo {

// *****************************************************************************
//                                                                       Line2
// *****************************************************************************
/// \brief Represents a 2D line by a point and a vector.
template <typename T = real_t> class Line2 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value,
                "Line2 supports only f32 and f64");

public:
  using scalar_type = T;
  using point_type = Point2<T>;
  using vector_type = Vector2<T>;
  Line2() = default;
  /// \param a line point
  /// \param d direction
  Line2(const point_type &a, const vector_type &d) : a(a), d(normalize(d)) {}
  /// \param t parametric coordinate
  /// \return euclidean point from parametric coordinate **t**
  point_type operator()(T t) const { return a + d * t; }
  /// \return unit vector representing line direction
  HERMES_NODISCARD vector_type direction() const { return normalize(d); }
  /// \param p point
  /// \return parametric coordinate of **p** projection into line
  HERMES_NODISCARD T projection(const point_type &p) const {
    return dot(p - a, d);
  }
  /// \param p point
  /// \return closest point in line from **p**
  HERMES_NODISCARD point_type closestPoint(const point_type &p) const {
    return (*this)(projection(p));
  }

  point_type a;  //!< line point
  vector_type d; //!< line direction
};

// *****************************************************************************
//                                                                       Line3
// *****************************************************************************
/// \brief Represents a line by a point and a vector.
template <typename T = real_t> class Line3 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value,
                "Line3 supports only f32 and f64");

public:
  using scalar_type = T;
  using point_type = Point3<T>;
  using vector_type = Vector3<T>;
  Line3() = default;
  /// \param _a line point
  /// \param _d direction
  Line3(const point_type &a, const vector_type &d) : a(a), d(normalize(d)) {}
  /// \param t parametric coordinate
  /// \return euclidean point from parametric coordinate **t**
  point_type operator()(T t) const { return a + d * t; }
  /// \return unit vector representing line direction
  HERMES_NODISCARD vector_type direction() const { return normalize(d); }
  /// \param p point
  /// \return parametric coordinate of **p** projection into line
  HERMES_NODISCARD T projection(const point_type &p) const {
    return dot(p - a, d);
  }
  /// \param p point
  /// \return closest point in line from **p**
  HERMES_NODISCARD point_type closestpoint(const point_type &p) const {
    return (*this)(projection(p));
  }

  point_type a;  //!< line point
  vector_type d; //!< line direction
};

using line2 = Line2<real_t>;
using line2f = Line2<f32>;
using line2d = Line2<f64>;
using line3 = Line3<real_t>;
using line3f = Line3<f32>;
using line3d = Line3<f64>;

} // namespace hermes::geo

namespace hermes {

HERMES_TYPE_LAYOUT_METHODS(geo::Line2<real_t>, real_t, 4)
HERMES_TYPE_LAYOUT_METHODS(geo::Line3<real_t>, real_t, 6)

#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
HERMES_TYPE_LAYOUT_METHODS(geo::Line2<f32>, f32, 4)
HERMES_TYPE_LAYOUT_METHODS(geo::Line3<f32>, f32, 6)
#else
HERMES_TYPE_LAYOUT_METHODS(geo::Line2<f64>, f64, 4)
HERMES_TYPE_LAYOUT_METHODS(geo::Line3<f64>, f64, 6)
#endif

template <typename T> struct DebugTraits<geo::Line2<T>> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Line2<T> &data) {
    return DebugMessage("Line2[p={} d={}]", hermes::to_string(data.a),
                        hermes::to_string(data.d));
  }
};

template <typename T> struct DebugTraits<geo::Line3<T>> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Line3<T> &data) {
    return DebugMessage("Line3[p={} d={}]", hermes::to_string(data.a),
                        hermes::to_string(data.d));
  }
};

} // namespace hermes
