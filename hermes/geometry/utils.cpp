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

#include <hermes/geometry/utils.h>
#include <hermes/numeric/numeric.h>

namespace hermes::coordinate_systems {

vec3f sphericalDirection(real_t sinTheta, real_t cosTheta,
                                   real_t phi) {
  return vec3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
}

vec3f sphericalDirection(real_t sinTheta, real_t cosTheta, real_t phi,
                                   const vec3f &x, const vec3f &y,
                                   const vec3f &z) {
  return sinTheta * std::cos(phi) * x + sinTheta * std::sin(phi) * y +
         cosTheta * z;
}

real_t sphericalTheta(const vec3f &v) {
  return std::acos(Numbers::clamp(v.z, -1.f, 1.f));
}

real_t sphericalPhi(const vec3f &v) {
  real_t p = std::atan2(v.y, v.x);
  return (p < 0) ? (p + 2 * Constants::pi) : p;
}

} // namespace hermes
