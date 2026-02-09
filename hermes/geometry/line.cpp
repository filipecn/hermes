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

/// \file   line.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2017-08-19
/// Geometric line classes

#include <hermes/geometry/line.h>

namespace hermes::geo {

Line2::Line2(const point2 &_a, const vec2 &_d) {
  a = _a;
  d = normalize(_d);
}

point2 Line2::operator()(f32 t) const { return a + d * t; }

vec2 Line2::direction() const { return normalize(d); }

f32 Line2::projection(const point2 &p) const { return dot((p - a), d); }

point2 Line2::closestPoint(const point2 &p) const {
  return (*this)(projection(p));
}

Line::Line(const point3 &a, const vec3 &d) : a{a}, d{normalize(d)} {}

point3 Line::operator()(real_t t) const { return a + d * t; }

vec3 Line::direction() const { return normalize(d); }

real_t Line::projection(const point3 &p) const { return dot((p - a), d); }

point3 Line::closestpoint(const point3 &p) const {
  return (*this)(projection(p));
}

} // namespace hermes::geo
