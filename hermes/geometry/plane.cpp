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

/// \file   plane.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2017-08-19

#include <hermes/geometry/plane.h>

namespace hermes::geo {

Plane::Plane() { offset = 0; }

Plane::Plane(normal3 n, real_t o) {
  normal = n;
  offset = o;
}

Plane::Plane(normal3 n, point3 p) {
  normal = n;
  offset = dot(n, (vec3)p);
}

Plane Plane::XY(bool invert_normal) {
  return {normal3(0, 0, invert_normal ? -1 : 1), 0};
}

Plane Plane::XZ(bool invert_normal) {
  return {normal3(0, invert_normal ? -1 : 1, 0), 0};
}

Plane Plane::YZ(bool invert_normal) {
  return {normal3(invert_normal ? -1 : 1, 0, 0), 0};
}

point3 Plane::closestPoint(const point3 &p) const {
  float t = (dot(vec3(normal), vec3(p)) - offset) / vec3(normal).length2();
  return p - t * vec3(normal);
}

vec3 Plane::project(const vec3 &v) const {
  return hermes::geo::project(v, normal);
}

vec3 Plane::reflect(const vec3 &v) const {
  return hermes::geo::reflect(v, normal);
}

bool Plane::isOnNormalSide(const point3 &p) const {
  return dot(vec3(normal), p - closestPoint(p)) >= 0;
}

} // namespace hermes::geo
