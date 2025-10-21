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

#include "hermes/core/debug.h"
#include "hermes/core/types.h"
#include <hermes/geometry/point.h>
#include <hermes/geometry/vector.h>

namespace hermes::geo {

// *****************************************************************************
//                                                                       Line
// *****************************************************************************
/// \brief Represents a 2D line by a point and a vector.
class Line2 {
public:
  Line2() = default;
  /// \param a line point
  /// \param d direction
  Line2(const point2 &a, const vec2 &d);
  /// \param t parametric coordinate
  /// \return euclidean point from parametric coordinate **t**
  point2 operator()(real_t t) const;
  /// \return unit vector representing line direction
  HERMES_NODISCARD vec2 direction() const;
  /// \param p point
  /// \return parametric coordinate of **p** projection into line
  HERMES_NODISCARD real_t projection(const point2 &p) const;
  /// \param p point
  /// \return closest point in line from **p**
  HERMES_NODISCARD point2 closestPoint(const point2 &p) const;

  point2 a; //!< line point
  vec2 d;   //!< line direction
};

// *****************************************************************************
//                                                                       Line
// *****************************************************************************
/// \brief Represents a line by a point and a vector.
class Line {
public:
  Line() = default;
  /// \param _a line point
  /// \param _d direction
  Line(const point3 &a, const vec3 &d);
  /// \param t parametric coordinate
  /// \return euclidean point from parametric coordinate **t**
  point3 operator()(real_t t) const;
  /// \return unit vector representing line direction
  HERMES_NODISCARD vec3 direction() const;
  /// \param p point
  /// \return parametric coordinate of **p** projection into line
  HERMES_NODISCARD real_t projection(const point3 &p) const;
  /// \param p point
  /// \return closest point in line from **p**
  HERMES_NODISCARD point3 closestpoint(const point3 &p) const;

  point3 a; //!< line point
  vec3 d;   //!< line direction
};

} // namespace hermes::geo

namespace hermes {

HERMES_TYPE_LAYOUT_METHODS(geo::Line2, real_t, 4)
HERMES_TYPE_LAYOUT_METHODS(geo::Line, real_t, 6)

HERMES_DECLARE_TO_STRING_DEBUG_METHOD(geo::Line2)
HERMES_DECLARE_TO_STRING_DEBUG_METHOD(geo::Line)

} // namespace hermes
