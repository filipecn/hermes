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
///\file queries.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-28
///
///\brief

#ifndef HERMES_GEOMETRY_QUERIES_H
#define HERMES_GEOMETRY_QUERIES_H

#include <hermes/geometry/bbox.h>
#include <hermes/geometry/ray.h>
#include <hermes/geometry/transform.h>
#include <hermes/geometry/utils.h>
#include <hermes/geometry/vector.h>
#include <optional>

namespace hermes {

class GeometricQueries {
public:
  ///
  /// \param box
  /// \param p
  /// \return
  static point3 closestPoint(const bbox3 &box, const point3 &p);
};

class GeometricPredicates {
public:
  ///
  /// \param bounds
  /// \param ray
  /// \param inv_dir
  /// \param dir_is_neg
  /// \param max_t
  /// \return
  static std::optional<real_t> intersect(const hermes::bbox3 &bounds,
                                         const ray3 &ray,
                                         const hermes::vec3 &inv_dir,
                                         const i32 dir_is_neg[3],
                                         real_t max_t = Constants::real_infinity);
  ///
  /// \param bounds
  /// \param ray
  /// \param second_hit
  /// \return
  static std::optional<real_t> intersect(const hermes::bbox3 &bounds,
                                         const ray3 &ray, real_t *second_hit = nullptr);

  /// BBox / Ray3 intersection test.
  /// **b1** and **b2**, if not null, receive the barycentric coordinates of the
  /// intersection point.
  /// \param p1 **[in]** first triangle's vertex
  /// \param p2 **[in]** second triangle's vertex
  /// \param p3 **[in]** third triangle's vertex
  /// \param ray **[in]**
  /// \param tHit **[out]** intersection (parametric coordinate)
  /// \param b1 **[out]** barycentric coordinate
  /// \param b2 **[out]** barycentric coordinate
  /// return **true** if intersection exists
  static std::optional<real_t> intersect(const point3 &p1, const point3 &p2,
                                         const point3 &p3, const Ray3 &ray,
                                         real_t *b1 = nullptr, real_t *b2 = nullptr);
};

} // namespace hermes

#endif // HERMES_GEOMETRY_QUERIES_H
