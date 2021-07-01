/*
 * Copyright (c) 2018 FilipeCN
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

#ifndef HERMES_GEOMETRY_TRANSFORM_H
#define HERMES_GEOMETRY_TRANSFORM_H

#include <hermes/geometry/bbox.h>
#include <hermes/geometry/matrix.h>
#include <hermes/geometry/normal.h>
#include <hermes/geometry/point.h>
#include <hermes/geometry/ray.h>
#include <hermes/common/debug.h>
#include <hermes/common/bitmask_operators.h>

namespace hermes {

enum class transform_options {
  none = 0x0,
  x_right = 0x1,
  y_right = 0x2,
  z_right = 0x4,
  left_handed = 0x8,
  x_left = 0x10,
  y_left = 0x20,
  z_left = 0x40,
  right_handed = 0x80,
  x_up = 0x100,
  y_up = 0x200,
  z_up = 0x400,
  zero_to_one = 0x800,
  x_down = 0x1000,
  y_down = 0x2000,
  z_down = 0x4000,
  transpose = 0x8000,
  flip_x = 0x10000,
  flip_y = 0x20000,
  flip_z = 0x40000,
};
HERMES_ENABLE_BITMASK_OPERATORS(transform_options);

// *********************************************************************************************************************
//                                                                                                         Transform2
// *********************************************************************************************************************
class Transform2 {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE static Transform2 scale(const vec2 &s);
  HERMES_DEVICE_CALLABLE static Transform2 rotate(real_t angle);
  HERMES_DEVICE_CALLABLE static Transform2 translate(const vec2 &v);
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  //                                                                                                          algebra
  friend Transform2 inverse(const Transform2 &t);
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Transform2();
  HERMES_DEVICE_CALLABLE Transform2(const mat3 &mat);
  HERMES_DEVICE_CALLABLE Transform2(const bbox2 &bbox);
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                        transform
  HERMES_DEVICE_CALLABLE void operator()(const point2 &p, point2 *r) const {
    real_t x = p.x, y = p.y;
    r->x = m[0][0] * x + m[0][1] * y + m[0][2];
    r->y = m[1][0] * x + m[1][1] * y + m[1][2];
    real_t wp = m[2][0] * x + m[2][1] * y + m[2][2];
    if (wp != 1.f)
      *r /= wp;
  }
  HERMES_DEVICE_CALLABLE void operator()(const vec2 &v, vec2 *r) const {
    real_t x = v.x, y = v.y;
    r->x = m[0][0] * x + m[0][1] * y;
    r->y = m[1][0] * x + m[1][1] * y;
  }
  HERMES_DEVICE_CALLABLE vec2 operator()(const vec2 &v) const {
    real_t x = v.x, y = v.y;
    return vec2(m[0][0] * x + m[0][1] * y, m[1][0] * x + m[1][1] * y);
  }
  HERMES_DEVICE_CALLABLE point2 operator()(const point2 &p) const {
    real_t x = p.x, y = p.y;
    real_t xp = m[0][0] * x + m[0][1] * y + m[0][2];
    real_t yp = m[1][0] * x + m[1][1] * y + m[1][2];
    real_t wp = m[2][0] * x + m[2][1] * y + m[2][2];
    if (wp == 1.f)
      return point2(xp, yp);
    return point2(xp / wp, yp / wp);
  }
  HERMES_DEVICE_CALLABLE bbox2 operator()(const bbox2 &b) const {
    const Transform2 &M = *this;
    bbox2 ret;
    ret = make_union(ret, M(point2(b.lower.x, b.lower.y)));
    ret = make_union(ret, M(point2(b.upper.x, b.lower.y)));
    ret = make_union(ret, M(point2(b.upper.x, b.upper.y)));
    ret = make_union(ret, M(point2(b.lower.x, b.upper.y)));
    return ret;
  }
  HERMES_DEVICE_CALLABLE Ray2 operator()(const Ray2 &r) {
    Ray2 ret = r;
    (*this)(ret.o, &ret.o);
    (*this)(ret.d, &ret.d);
    return ret;
  }
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE Transform2 operator*(const Transform2 &t) const {
    return m * t.m;
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE void reset();
  HERMES_DEVICE_CALLABLE [[nodiscard]] vec2 getTranslate() const { return vec2(m[0][2], m[1][2]); }
  HERMES_DEVICE_CALLABLE [[nodiscard]] vec2 getScale() const { return {0, 0}; }
  HERMES_DEVICE_CALLABLE [[nodiscard]] mat3 getMatrix() const { return m; }
  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE const real_t *operator[](u32 row_index) const { return m[row_index]; }
  HERMES_DEVICE_CALLABLE real_t *operator[](u32 row_index) { return m[row_index]; }

private:
  mat3 m;
};

Transform2 inverse(const Transform2 &t);

// *********************************************************************************************************************
//                                                                                                          Transform
// *********************************************************************************************************************
class Transform {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  //                                                                                                      projections
  /// Look At Transform
  /// \note This transform is commonly used (in graphics) to orient a camera so
  /// it looks at a certain **target** position from its **eye** position.
  /// Given an **up** vector to define the camera orientation, a new coordinate
  /// basis consisting of three vectors {r, u, v} is defined. Where
  /// v = (eye - target) / ||eye - target||
  /// r = -(v x up) / ||(v x up)||
  /// u = v x r
  /// \note The transform is then composed of a translation (to camera **eye**
  /// position) and a basis transform to align r with (1,0,0), u with (0,1,0)
  /// and v with (0,0,1). The final matrix is
  ///     rx  ry  rz  -dot(t, r)
  ///     rx  ry  rz  -dot(t, u)
  ///     rx  ry  rz  -dot(t, v)
  ///      0   0   0      1
  /// \note Note that this transform is built on a left handed coordinate system.
  /// \param eye camera position
  /// \param target camera target
  /// \param up orientation vector
  /// \param left_handed
  /// \return
  HERMES_DEVICE_CALLABLE static Transform lookAt(const point3 &eye, const point3 &target = {0, 0, 0},
                                                 const vec3 &up = {0, 1, 0},
                                                 transform_options options = transform_options::left_handed);
  /// Orthographic Projection
  /// \note In an orthographic projection, parallel lines remain parallel and objects
  /// maintain the same size regardless the distance.
  /// \note This transform projects points into the cube (-1,-1,-1) x (1, 1, 1). It is
  /// also possible to choose to project to (-1,-1, 0) x (1, 1, 1) with the
  /// zero_to_one option.
  /// \note The matrix takes the form:
  ///     2 / (r - l)       0             0         -(r + l) / (r - l)
  ///         0         2 / (t - b)       0         -(t + b) / (t - b)
  ///         0             0         2 / (f - n)   -(f + n) / (f - n)
  ///         0             0             0                  1
  /// \note In the case of zero_to_one == true, the matrix becomes:
  ///     2 / (r - l)       0             0         -(r + l) / (r - l)
  ///         0         2 / (t - b)       0         -(t + b) / (t - b)
  ///         0             0         1 / (f - n)          n / (f - n)
  ///         0             0             0                  1
  /// \note - Note that n > f. This function negates the values of near and
  /// far in case the given values are f > n. Because by default, this
  /// transform uses a left-handed coordinate system.
  /// \param left
  /// \param right
  /// \param bottom
  /// \param top
  /// \param near
  /// \param far
  /// \param left_handed
  /// \param zero_to_one
  /// \return
  HERMES_DEVICE_CALLABLE static Transform ortho(real_t left, real_t right, real_t bottom, real_t top,
                                                real_t near, real_t far,
                                                transform_options options = transform_options::left_handed);
  /// Perspective Projection
  /// \note The perspective projection transforms the view frustrum (a pyramid
  /// truncated by a near plane and a far plane, both orthogonal to the view
  /// direction) into the cube (-1,-1,-1) x (1, 1, 1).
  /// \note In a right-handed coordinate system when x points to the right, z
  /// points forward if y points downward and z points backwards if y points
  /// upwards.
  /// \note In a left-handed coordinate system when x points to the right, z
  /// points forwards if y points upward and z points backward if y points
  /// downwards.
  ///
  /// \note It is also possible to choose to project to (-1,-1, 0) x (1, 1, 1)
  /// with the zero_to_one option.
  /// \param fovy
  /// \param aspect_ratio
  /// \param near
  /// \param far
  /// \param left_handed
  /// \param zero_to_one
  /// \return
  HERMES_DEVICE_CALLABLE static Transform perspective(real_t fovy_in_degrees,
                                                      real_t aspect_ratio,
                                                      real_t near,
                                                      real_t far,
                                                      transform_options options = transform_options::left_handed);
  //                                                                                                        transform
  HERMES_DEVICE_CALLABLE static Transform segmentToSegmentTransform(point3 a, point3 b, point3 c, point3 d);
  HERMES_DEVICE_CALLABLE static Transform scale(real_t x, real_t y, real_t z);
  HERMES_DEVICE_CALLABLE static Transform translate(const vec3 &d);
  HERMES_DEVICE_CALLABLE static Transform rotateX(real_t angle);
  HERMES_DEVICE_CALLABLE static Transform rotateY(real_t angle);
  HERMES_DEVICE_CALLABLE static Transform rotateZ(real_t angle);
  HERMES_DEVICE_CALLABLE static Transform rotate(real_t angle, const vec3 &axis);
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  //                                                                                                          algebra
  HERMES_DEVICE_CALLABLE friend Transform inverse(const Transform &t);
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Transform();
  HERMES_DEVICE_CALLABLE Transform(const mat4 &mat);
  HERMES_DEVICE_CALLABLE explicit Transform(const real_t mat[4][4]);
  HERMES_DEVICE_CALLABLE Transform(const bbox3 &bbox);
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                        transform
  bbox3 operator()(const bbox3 &b) const {
    const Transform &M = *this;
    bbox3 ret(M(point3(b.lower.x, b.lower.y, b.lower.z)));
    ret = make_union(ret, M(point3(b.upper.x, b.lower.y, b.lower.z)));
    ret = make_union(ret, M(point3(b.lower.x, b.upper.y, b.lower.z)));
    ret = make_union(ret, M(point3(b.lower.x, b.lower.y, b.upper.z)));
    ret = make_union(ret, M(point3(b.lower.x, b.upper.y, b.upper.z)));
    ret = make_union(ret, M(point3(b.upper.x, b.upper.y, b.lower.z)));
    ret = make_union(ret, M(point3(b.upper.x, b.lower.y, b.upper.z)));
    ret = make_union(ret, M(point3(b.lower.x, b.upper.y, b.upper.z)));
    return ret;
  }
  point3 operator()(const point2 &p) const {
    real_t x = p.x, y = p.y, z = 0.f;
    real_t xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    real_t yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    real_t zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp == 1.f)
      return point3(xp, yp, zp);
    return point3(xp, yp, zp) / wp;
  }
  point3 operator()(const point3 &p) const {
    real_t x = p.x, y = p.y, z = p.z;
    real_t xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    real_t yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    real_t zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp == 1.f)
      return point3(xp, yp, zp);
    return point3(xp, yp, zp) / wp;
  }
  void operator()(const point3 &p, point3 *r) const {
    real_t x = p.x, y = p.y, z = p.z;
    r->x = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    r->y = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    r->z = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp != 1.f)
      *r /= wp;
  }
  vec3 operator()(const vec3 &v) const {
    real_t x = v.x, y = v.y, z = v.z;
    return vec3(m[0][0] * x + m[0][1] * y + m[0][2] * z,
                m[1][0] * x + m[1][1] * y + m[1][2] * z,
                m[2][0] * x + m[2][1] * y + m[2][2] * z);
  }
  normal3 operator()(const normal3 &n) const {
    real_t x = n.x, y = n.y, z = n.z;
    auto m_inv = inverse(*this);
    return normal3(m_inv[0][0] * x + m_inv[1][0] * y + m_inv[2][0] * z,
                   m_inv[0][1] * x + m_inv[1][1] * y + m_inv[2][1] * z,
                   m_inv[0][2] * x + m_inv[1][2] * y + m_inv[2][2] * z);
  }
  Ray3 operator()(const Ray3 &r) {
    Ray3 ret = r;
    (*this)(ret.o, &ret.o);
    ret.d = (*this)(ret.d);
    return ret;
  }
  void operator()(const Ray3 &r, Ray3 *ret) const {
    (*this)(r.o, &ret->o);
    ret->d = (*this)(ret->d);
  }
  //                                                                                                       arithmetic
  Transform &operator=(const Transform2 &t) {
    m.setIdentity();
    mat3 m3 = t.getMatrix();
    m[0][0] = m3[0][0];
    m[0][1] = m3[0][1];
    m[0][3] = m3[0][2];

    m[1][0] = m3[1][0];
    m[1][1] = m3[1][1];
    m[1][3] = m3[1][2];
    return *this;
  }
  Transform operator*(const Transform &t) const {
    mat4 m1 = m * t.m;
    return {m1};
  }
  point3 operator*(const point3 &p) const { return (*this)(p); }
  //                                                                                                          boolean
  bool operator==(const Transform &t) const { return t.m == m; }
  bool operator!=(const Transform &t) const { return t.m != m; }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  void reset();
  /// \return true if this transformation changes the coordinate system
  /// handedness
  HERMES_DEVICE_CALLABLE [[nodiscard]] bool swapsHandedness() const;
  HERMES_DEVICE_CALLABLE [[nodiscard]] vec3 getTranslate() const { return vec3(m[0][3], m[1][3], m[2][3]); }
  HERMES_DEVICE_CALLABLE bool isIdentity() { return m.isIdentity(); }
  HERMES_DEVICE_CALLABLE void applyToPoint(const real_t *p, real_t *r, size_t d = 3) const {
    real_t x = p[0], y = p[1], z = 0.f;
    if (d == 3)
      z = p[2];
    r[0] = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    r[1] = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    if (d == 3)
      r[2] = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp != 1.f) {
      real_t invwp = 1.f / wp;
      r[0] *= invwp;
      r[1] *= invwp;
      if (d == 3)
        r[2] *= invwp;
    }
  }

  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE [[nodiscard]] const real_t *c_matrix() const { return &m[0][0]; }
  HERMES_DEVICE_CALLABLE [[nodiscard]] const mat4 &matrix() const { return m; }
  HERMES_DEVICE_CALLABLE [[nodiscard]] mat3 upperLeftMatrix() const {
    return mat3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1],
                m[1][2], m[2][0], m[2][1], m[2][2]);
  }
  HERMES_DEVICE_CALLABLE const real_t *operator[](u32 row_index) const { return m[row_index]; }
  HERMES_DEVICE_CALLABLE real_t *operator[](u32 row_index) { return m[row_index]; }
protected:
  mat4 m;
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
std::ostream &operator<<(std::ostream &os, const Transform2 &m) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      os << m[i][j] << " ";
    os << std::endl;
  }
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Transform &m) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      os << m[i][j] << " ";
    os << std::endl;
  }
  return os;
}

} // namespace hermes

#endif
