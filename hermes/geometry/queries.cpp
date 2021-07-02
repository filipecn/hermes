//
// Created by filipecn on 28/06/2021.
//


/// Copyright (c) 2021, FilipeCN.
///
/// The MIT License (MIT)
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.
///
///\file queries.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-28
///
///\brief

#include <hermes/geometry/matrix.h>
#include <hermes/numeric/numeric.h>
#include <hermes/geometry/queries.h>

#include <utility>

namespace hermes {

point3 GeometricQueries::closestPoint(const bbox3 &box, const point3 &p) {
  HERMES_UNUSED_VARIABLE(box);
  HERMES_UNUSED_VARIABLE(p);
  return hermes::point3();
}

std::optional<real_t> GeometricPredicates::intersect(const bbox3 &bounds,
                                                     const ray3 &ray,
                                                     const vec3 &inv_dir,
                                                     const i32 *dir_is_neg,
                                                     real_t max_t) {
  // check for ray intersection against x and y slabs
  real_t t_min = (bounds[dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
  real_t t_max = (bounds[1 - dir_is_neg[0]].x - ray.o.x) * inv_dir.x;
  real_t ty_min = (bounds[dir_is_neg[1]].y - ray.o.y) * inv_dir.y;
  real_t ty_max = (bounds[1 - dir_is_neg[1]].y - ray.o.y) * inv_dir.y;
  // update t_max and tyMax to ensure robust bounds intersection
  if (t_min > ty_max || ty_min > t_max)
    return std::nullopt;
  if (ty_min > t_min)
    t_min = ty_min;
  if (ty_max < t_max)
    t_max = ty_max;
  // check for ray intersection against z slab
  real_t tz_min = (bounds[dir_is_neg[2]].z - ray.o.z) * inv_dir.z;
  real_t tz_max = (bounds[1 - dir_is_neg[2]].z - ray.o.z) * inv_dir.z;
  if (t_min > tz_max || tz_min > t_max)
    return std::nullopt;
  if (tz_min > t_min)
    t_min = tz_min;
  if (tz_max < t_max)
    t_max = tz_max;
  if ((t_min < max_t) && (t_max > 0))
    return std::optional<real_t>{t_min};
  return std::nullopt;
}

std::optional<real_t> GeometricPredicates::intersect(const hermes::bbox3 &bounds,
                                                     const ray3 &ray, real_t *second_hit) {

  real_t t0 = 0.f, t1 = Constants::real_infinity;
  for (int i = 0; i < 3; i++) {
    real_t inv_ray_dir = 1.f / ray.d[i];
    real_t t_near = (bounds.lower[i] - ray.o[i]) * inv_ray_dir;
    real_t t_far = (bounds.upper[i] - ray.o[i]) * inv_ray_dir;
    if (t_near > t_far)
      std::swap(t_near, t_far);
    t0 = t_near > t0 ? t_near : t0;
    t1 = t_far < t1 ? t_far : t1;
    if (t0 > t1)
      return std::nullopt;
  }
  std::optional<real_t> hit(t0);
  if (second_hit)
    *second_hit = t1;
  return hit;
}

std::optional<real_t> GeometricPredicates::intersect(const point3 &p1,
                                                     const point3 &p2,
                                                     const point3 &p3,
                                                     const Ray3 &ray,
                                                     real_t *b0,
                                                     real_t *b1) {
  real_t max_t = Constants::real_infinity;
  // transform triangle vertices to ray coordinate space
  //    translate vertices based on ray origin
  auto p0t = p1 - vec3(ray.o);
  auto p1t = p2 - vec3(ray.o);
  auto p2t = p3 - vec3(ray.o);
  //    permute components of triangle vertices and ray direction
  int kz = abs(ray.d).maxDimension();
  int kx = (kz + 1) % 3;
  int ky = (kx + 1) % 3;
  vec3 d = {ray.d[kx], ray.d[ky], ray.d[kz]};
  p0t = {p0t[kx], p0t[ky], p0t[kz]};
  p1t = {p1t[kx], p1t[ky], p1t[kz]};
  p2t = {p2t[kx], p2t[ky], p2t[kz]};
  //    align ray direction with the z+ axis
  real_t sx = -d.x / d.z;
  real_t sy = -d.y / d.z;
  real_t sz = 1.f / d.z;
  p0t.x += sx * p0t.z;
  p0t.y += sy * p0t.z;
  p1t.x += sx * p1t.z;
  p1t.y += sy * p1t.z;
  p2t.x += sx * p2t.z;
  p2t.y += sy * p2t.z;
  // compute edge function coefficients e0, e1, and e2
  real_t e0 = p1t.x * p2t.y - p1t.y * p2t.x;
  real_t e1 = p2t.x * p0t.y - p2t.y * p0t.x;
  real_t e2 = p0t.x * p1t.y - p0t.y * p1t.x;
  // fall back to double precision test at triangle edges
  if (e0 == 0 && e1 == 0 && e2 == 0) {
    e0 = static_cast<f64>(p1t.x) * p2t.y - static_cast<f64>(p1t.y) * p2t.x;
    e1 = static_cast<f64>(p2t.x) * p0t.y - static_cast<f64>(p2t.y) * p0t.x;
    e2 = static_cast<f64>(p0t.x) * p1t.y - static_cast<f64>(p0t.y) * p1t.x;
  }
  // perform triangle edge and determinant tests
  if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
    return std::nullopt;
  real_t det = e0 + e1 + e2;
  if (det == 0)
    return std::nullopt;
  // compute scaled hit distance to triangle and test against ray t range
  p0t.z *= sz;
  p1t.z *= sz;
  p2t.z *= sz;
  real_t t_scaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
  if ((det < 0 && (t_scaled >= 0 || t_scaled < max_t * det)) ||
      (det > 0 && (t_scaled <= 0 || t_scaled > max_t * det)))
    return std::nullopt;
  // compute barycentric coordinates and t value for triangle intersection
  real_t inv_det = 1 / det;
  if (b0)
    *b0 = e0 * inv_det;
  if (b1)
    *b1 = e1 * inv_det;
  //  real_t b2 = e2 * inv_det;
  real_t t = t_scaled * inv_det;
  // ensure that computed triangle t is conservatively greater than zero
  //    compute dz term for triangle t error bounds
  real_t max_z_t = abs(vec3(p0t.z, p1t.z, p2t.z)).maxDimension();
  real_t delta_z = Numbers::gamma(3) * max_z_t;
  //    compute dx and dy terms for triangle t error bounds
  real_t max_x_t = abs(vec3(p0t.x, p1t.x, p2t.x)).maxDimension();
  real_t max_y_t = abs(vec3(p0t.y, p1t.y, p2t.y)).maxDimension();
  real_t delta_x = Numbers::gamma(5) * (max_x_t + max_z_t);
  real_t delta_y = Numbers::gamma(5) * (max_y_t + max_z_t);
  //    compute de term for triangle error bounds
  real_t delta_e = 2 * (Numbers::gamma(2) * max_x_t * max_y_t + delta_y * max_x_t + delta_x * max_y_t);
  //    compute dt term for triangle t error bounds and check t
  real_t max_e = abs(vec3(e0, e1, e2)).maxDimension();
  real_t
      delta_t = 3 * (Numbers::gamma(3) * max_e * max_z_t + delta_e * max_z_t + delta_z * max_e) * std::abs(inv_det);
  if (t <= delta_t)
    return std::nullopt;
  return std::optional<real_t>(t);
}

} // namespace hermes
