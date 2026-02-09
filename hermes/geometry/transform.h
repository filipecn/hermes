/* Copyright (c) 2018, FilipeCN.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/// \file   transform.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2018-08-19
/// \brief  Geometric transform classes

#pragma once

#include <hermes/geometry/point.h>
#include <hermes/numeric/matrix.h>

namespace hermes::geo {

/// \brief Options for transform functions
/// \note You can use bitwise operators to combine these options
enum class transform_option_bits : u32 {
  none = 0x0,          //!< default behaviour
  x_right = 0x1,       //!< set x-axis to right
  y_right = 0x2,       //!< set y-axis to right
  z_right = 0x4,       //!< set z-axis to right
  left_handed = 0x8,   //!< use left-handed coordinate system
  x_left = 0x10,       //!< set x-axis to left
  y_left = 0x20,       //!< set y-axis to left
  z_left = 0x40,       //!< set z-axis to left
  right_handed = 0x80, //!< use right-handed coordinate system
  x_up = 0x100,        //!< set x-axis to up
  y_up = 0x200,        //!< set y-axis to up
  z_up = 0x400,        //!< set z-axis to up
  zero_to_one = 0x800, //!< maps projection range to [0,1], instead of [-1,1]
  x_down = 0x1000,     //!< set x-axis to down
  y_down = 0x2000,     //!< set y-axis to down
  z_down = 0x4000,     //!< set z-axis to down
  transpose = 0x8000,  //!< transpose transformation matrix
  flip_x = 0x10000,    //!< invert x-axis
  flip_y = 0x20000,    //!< invert y-axis
  flip_z = 0x40000,    //!< invert z-axis
};

using transform_options = hermes::Flags<geo::transform_option_bits>;

} // namespace hermes::geo

namespace hermes {
template <> struct FlagTraits<geo::transform_option_bits> {
  static HERMES_CONST_OR_CONSTEXPR bool is_bitmask = true;
  static HERMES_CONST_OR_CONSTEXPR geo::transform_options all_flags =
      geo::transform_option_bits::x_right |
      geo::transform_option_bits::y_right |
      geo::transform_option_bits::z_right |
      geo::transform_option_bits::left_handed |
      geo::transform_option_bits::x_left | geo::transform_option_bits::y_left |
      geo::transform_option_bits::z_left |
      geo::transform_option_bits::right_handed |
      geo::transform_option_bits::x_up | geo::transform_option_bits::y_up |
      geo::transform_option_bits::z_up |
      geo::transform_option_bits::zero_to_one |
      geo::transform_option_bits::x_down | geo::transform_option_bits::y_down |
      geo::transform_option_bits::z_down |
      geo::transform_option_bits::transpose |
      geo::transform_option_bits::flip_x | geo::transform_option_bits::flip_y |
      geo::transform_option_bits::flip_z;
};

} // namespace hermes

namespace hermes::geo {

class Transform;
class Transform2;
class bbox2;

/// \brief Computes inverse of a given transform
/// \param t
/// \return
HERMES_CPU_GPU Transform inverse(const Transform &t);
/// \brief Computes inverse of a given transform
/// \param t
/// \return
Transform2 inverse(const Transform2 &t);

// *****************************************************************************
//                                                                 Transform2
// *****************************************************************************
/// \brief Represents a 2-dimensional transformation
class Transform2 {
public:
  /// \brief Creates scale transform
  /// \param s
  /// \return
  HERMES_CPU_GPU static Transform2 scale(const vec2 &s);
  /// \brief Creates rotation transform
  /// \param angle
  /// \return
  HERMES_CPU_GPU static Transform2 rotate(real_t angle);
  /// \brief Creates translation transform
  /// \param v
  /// \return
  HERMES_CPU_GPU static Transform2 translate(const vec2 &v);

  /// \brief Gets inverse transform from t
  /// \param t
  /// \return
  friend Transform2 inverse(const Transform2 &t) { return inverse(t.m); }

  /// \brief Default constructor
  HERMES_CPU_GPU Transform2();
  /// \brief Constructs from matrix
  /// \param mat
  HERMES_CPU_GPU Transform2(const math::mat3 &mat);
  /// \brief Constructs from bounding box
  /// \param bbox
  HERMES_CPU_GPU Transform2(const bbox2 &bbox);

  /// \brief Applies this transform to geometric point
  /// \param p
  /// \param r
  HERMES_CPU_GPU void operator()(const point2 &p, point2 *r) const {
    real_t x = p.x, y = p.y;
    r->x = m[0][0] * x + m[0][1] * y + m[0][2];
    r->y = m[1][0] * x + m[1][1] * y + m[1][2];
    real_t wp = m[2][0] * x + m[2][1] * y + m[2][2];
    if (wp != 1.f)
      *r /= wp;
  }
  /// \brief Applies this transform to geometric vector
  /// \param v
  /// \param r
  HERMES_CPU_GPU void operator()(const vec2 &v, vec2 *r) const {
    real_t x = v.x, y = v.y;
    r->x = m[0][0] * x + m[0][1] * y;
    r->y = m[1][0] * x + m[1][1] * y;
  }
  /// \brief Applies this transform to geometric vector
  /// \param v
  /// \return
  HERMES_CPU_GPU vec2 operator()(const vec2 &v) const {
    real_t x = v.x, y = v.y;
    return vec2(m[0][0] * x + m[0][1] * y, m[1][0] * x + m[1][1] * y);
  }
  /// \brief Applies this transform to geometric point
  /// \param p
  /// \return
  HERMES_CPU_GPU point2 operator()(const point2 &p) const {
    real_t x = p.x, y = p.y;
    real_t xp = m[0][0] * x + m[0][1] * y + m[0][2];
    real_t yp = m[1][0] * x + m[1][1] * y + m[1][2];
    real_t wp = m[2][0] * x + m[2][1] * y + m[2][2];
    if (wp == 1.f)
      return point2(xp, yp);
    return point2(xp / wp, yp / wp);
  }
  /// \brief Applies this transform to geometric box
  /// \param b
  /// \return
  // HERMES_CPU_GPU bbox2 operator()(const bbox2 &b) const {
  //   const Transform2 &M = *this;
  //   bbox2 ret;
  //   ret = make_union(ret, M(point2(b.lower.x, b.lower.y)));
  //   ret = make_union(ret, M(point2(b.upper.x, b.lower.y)));
  //   ret = make_union(ret, M(point2(b.upper.x, b.upper.y)));
  //   ret = make_union(ret, M(point2(b.lower.x, b.upper.y)));
  //   return ret;
  // }
  /// \brief Applies this transform to geometric ray
  /// \param r
  /// \return
  // HERMES_CPU_GPU Ray2 operator()(const Ray2 &r) {
  //   Ray2 ret = r;
  //   (*this)(ret.o, &ret.o);
  //   (*this)(ret.d, &ret.d);
  //   return ret;
  // }
  //                                                                                                        arithmetic
  /// \brief Applies this transform to another transform
  /// \param t
  /// \return
  HERMES_CPU_GPU Transform2 operator*(const Transform2 &t) const {
    return m * t.m;
  }

  /// \brief Sets this transform back to identity
  HERMES_CPU_GPU void reset();
  /// \brief Extracts translation vector
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU vec2 getTranslate() const {
    return vec2(m[0][2], m[1][2]);
  }
  /// \brief Extracts scale vector
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU vec2 getScale() const { return {0, 0}; }
  /// \brief Extracts rotation matrix
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU math::mat3 getMatrix() const { return m; }

  /// \brief Gets transform matrix row
  /// \param row_index
  /// \return
  HERMES_CPU_GPU const real_t *operator[](u32 row_index) const {
    return m[row_index];
  }
  /// \brief Gets transform matrix row
  /// \param row_index
  /// \return
  HERMES_CPU_GPU real_t *operator[](u32 row_index) { return m[row_index]; }

private:
  math::mat3 m;
};

// *****************************************************************************
//                                                                  Transform
// *****************************************************************************
/// \brief Represents a 3-dimensional transformation
class Transform {
public:
  /// Creates a Look At Transform
  ///
  /// This transform is commonly used (in graphics) to orient a camera so
  /// it looks at a certain **target** position from its **eye** position.
  /// Given an **up** vector to define the camera orientation, a new coordinate
  /// basis consisting of three vectors {r, u, v} is defined. Where
  /// v = (eye - target) / ||eye - target||
  /// r = (up x v) / ||(up x v)||
  /// u = v x r
  ///
  /// The transform is then composed of a translation (to camera **eye**
  /// position) and a basis transform to align r with (1,0,0), u with (0,1,0)
  /// and v with (0,0,1). The final matrix is
  ///     rx   ry   rz  -dot(t, r)
  ///     ux   uy   uz  -dot(t, u)
  ///     vx   vy   vz  -dot(t, v)
  ///      0    0    0      1
  /// \note Note that v is negated in the matrix to orient the camera to -z.
  /// \note Note that this transform is built on a left handed coordinates. The
  ///       right-handed version can be selected with
  ///       transform_option_bits::right-handed
  /// \note This is a camera to world transform, and not the so called view
  ///       transform.
  /// \param eye camera position.
  /// \param target camera target.
  /// \param up camera orientation.
  /// \param options right/left handed versions.
  HERMES_CPU_GPU static Transform
  lookAt(const point3 &eye, const point3 &target = {0, 0, 0},
         const vec3 &up = {0, 1, 0},
         transform_options options = transform_option_bits::left_handed);
  /// Creates an Orthographic Projection
  ///
  /// In an orthographic projection, parallel lines remain parallel and objects
  /// maintain the same size regardless the distance. This transform projects
  /// points into the cube (-1,-1,-1) x (1, 1, 1).
  ///
  /// \note It is also possible to choose to project to (-1,-1, 0) x (1, 1, 1)
  ///       with the zero_to_one option.
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
  /// \note Note that n > f. This function negates the values of near and far in
  ///       case the given values are f > n. Because by default, this transform
  ///       uses a left-handed coordinate system.
  /// \param left
  /// \param right
  /// \param bottom
  /// \param top
  /// \param near
  /// \param far
  /// \param options
  /// \return
  HERMES_CPU_GPU static Transform
  ortho(real_t left, real_t right, real_t bottom, real_t top, real_t near,
        real_t far,
        transform_options options = transform_option_bits::left_handed);
  /// Creates a Perspective Projection
  ///
  /// The perspective projection transforms the view frustrum (a pyramid
  /// truncated by a near plane and a far plane, both orthogonal to the view
  /// direction) into the cube (-1,-1,-1) x (1, 1, 1).
  ///
  /// \note In a right-handed coordinate system when x points to the right, z
  ///       points forward if y points downward and z points backwards if y
  ///       points upwards.
  /// \note In a left-handed coordinate system when x points to the right, z
  ///       points forwards if y points upward and z points backward if y points
  ///       downwards.
  /// \note It is also possible to choose to project to (-1,-1, 0) x (1, 1, 1)
  ///       with the zero_to_one option.
  /// \param fovy_in_degrees
  /// \param aspect_ratio
  /// \param near
  /// \param far
  /// \param options
  /// \return
  HERMES_CPU_GPU static Transform
  perspective(real_t fovy_in_degrees, real_t aspect_ratio, real_t near,
              real_t far,
              transform_options options = transform_option_bits::left_handed);
  /// \brief Creates a scale transform
  /// \param x
  /// \param y
  /// \param z
  /// \return
  HERMES_CPU_GPU static Transform scale(real_t x, real_t y, real_t z);
  /// \brief Creates a translation transform
  /// \param d
  /// \return
  HERMES_CPU_GPU static Transform translate(const vec3 &d);
  /// \brief Creates a x-axis rotation transform
  /// \param angle_in_radians
  /// \return
  HERMES_CPU_GPU static Transform rotateX(real_t angle_in_radians);
  /// \brief Creates a y-axis rotation transform
  /// \param angle_in_radians
  /// \return
  HERMES_CPU_GPU static Transform rotateY(real_t angle_in_radians);
  /// \brief Creates a z-axis rotation transform
  /// \param angle_in_radians
  /// \return
  HERMES_CPU_GPU static Transform rotateZ(real_t angle_in_radians);
  /// \brief Creates a arbitrary-axis rotation transform
  /// \param angle_in_radians
  /// \param axis
  /// \return
  HERMES_CPU_GPU static Transform rotate(real_t angle_in_radians,
                                         const vec3 &axis);
  /// \brief Creates a transform that aligns vector a to vector b
  /// \param a source vector
  /// \param b destination vector
  /// \return
  HERMES_CPU_GPU static Transform alignVectors(const vec3 &a, const vec3 &b);

  /// \brief Computes inverse of a given transform
  /// \param t
  /// \return
  HERMES_CPU_GPU friend Transform inverse(const Transform &t);

  /// \brief Default constructor
  HERMES_CPU_GPU Transform() { m.setIdentity(); }
  /// \brief Constructs from matrix
  /// \param mat
  HERMES_CPU_GPU Transform(const math::mat4 &mat);
  /// \brief Constructs from array matrix
  /// \param mat
  HERMES_CPU_GPU explicit Transform(const real_t mat[4][4]);
  /// \brief Constructs from geometric box
  /// \param bbox
  // HERMES_CPU_GPU Transform(const bbox3 &bbox);

  /// \brief Applies this transform to geometric box
  /// \param b
  /// \return
  // HERMES_CPU_GPU bbox3 operator()(const bbox3 &b) const {
  //   const Transform &M = *this;
  //   bbox3 ret(M(point3(b.lower.x, b.lower.y, b.lower.z)));
  //   ret = make_union(ret, M(point3(b.upper.x, b.lower.y, b.lower.z)));
  //   ret = make_union(ret, M(point3(b.lower.x, b.upper.y, b.lower.z)));
  //   ret = make_union(ret, M(point3(b.lower.x, b.lower.y, b.upper.z)));
  //   ret = make_union(ret, M(point3(b.lower.x, b.upper.y, b.upper.z)));
  //   ret = make_union(ret, M(point3(b.upper.x, b.upper.y, b.lower.z)));
  //   ret = make_union(ret, M(point3(b.upper.x, b.lower.y, b.upper.z)));
  //   ret = make_union(ret, M(point3(b.lower.x, b.upper.y, b.upper.z)));
  //   return ret;
  // }
  /// \brief Applies this transform to geometric point
  /// \param p
  /// \return
  HERMES_CPU_GPU point3 operator()(const point2 &p) const {
    real_t x = p.x, y = p.y, z = 0.f;
    real_t xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    real_t yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    real_t zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp == 1.f)
      return point3(xp, yp, zp);
    return point3(xp, yp, zp) / wp;
  }
  /// \brief Applies this transform to geometric point
  /// \param p
  /// \return
  HERMES_CPU_GPU point3 operator()(const point3 &p) const {
    real_t x = p.x, y = p.y, z = p.z;
    real_t xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    real_t yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    real_t zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp == 1.f)
      return point3(xp, yp, zp);
    return point3(xp, yp, zp) / wp;
  }
  /// \brief Applies this transform to geometric point
  /// \param p
  /// \param r
  HERMES_CPU_GPU void operator()(const point3 &p, point3 *r) const {
    real_t x = p.x, y = p.y, z = p.z;
    r->x = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    r->y = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    r->z = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    real_t wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp != 1.f)
      *r /= wp;
  }
  /// \brief Applies this transform to geometric point
  /// \param v
  /// \return
  HERMES_CPU_GPU vec3 operator()(const vec3 &v) const {
    real_t x = v.x, y = v.y, z = v.z;
    return vec3(m[0][0] * x + m[0][1] * y + m[0][2] * z,
                m[1][0] * x + m[1][1] * y + m[1][2] * z,
                m[2][0] * x + m[2][1] * y + m[2][2] * z);
  }
  /// \brief Applies this transform to geometric normal
  /// \param n
  /// \return
  // HERMES_CPU_GPU normal3 operator()(const normal3 &n) const {
  //   real_t x = n.x, y = n.y, z = n.z;
  //   auto m_inv = inverse(*this);
  //   return normal3(m_inv[0][0] * x + m_inv[1][0] * y + m_inv[2][0] * z,
  //                  m_inv[0][1] * x + m_inv[1][1] * y + m_inv[2][1] * z,
  //                  m_inv[0][2] * x + m_inv[1][2] * y + m_inv[2][2] * z);
  // }
  /// \brief Applies this transform to geometric ray
  /// \param r
  /// \return
  // HERMES_CPU_GPU Ray3 operator()(const Ray3 &r) {
  //   Ray3 ret = r;
  //   (*this)(ret.o, &ret.o);
  //   ret.d = (*this)(ret.d);
  //   return ret;
  // }
  /// \brief Applies this transform to geometric ray
  /// \param r
  /// \param ret
  // HERMES_CPU_GPU void operator()(const Ray3 &r, Ray3 *ret) const {
  //   (*this)(r.o, &ret->o);
  //   ret->d = (*this)(ret->d);
  // }
  //                                                                                                        arithmetic
  /// \brief Copy assign from 2d transform
  /// \param t
  /// \return
  HERMES_CPU_GPU Transform &operator=(const Transform2 &t) {
    m.setIdentity();
    math::mat3 m3 = t.getMatrix();
    m[0][0] = m3[0][0];
    m[0][1] = m3[0][1];
    m[0][3] = m3[0][2];

    m[1][0] = m3[1][0];
    m[1][1] = m3[1][1];
    m[1][3] = m3[1][2];
    return *this;
  }
  /// \brief Applies this transform to t
  /// \param t
  /// \return
  HERMES_CPU_GPU Transform operator*(const Transform &t) const {
    math::mat4 m1 = m * t.m;
    return {m1};
  }
  /// \brief Applies this transform to geometric vector
  /// \param p
  /// \return
  HERMES_CPU_GPU point3 operator*(const point3 &p) const { return (*this)(p); }
  //                                                                                                          boolean
  HERMES_CPU_GPU bool operator==(const Transform &t) const { return t.m == m; }
  HERMES_CPU_GPU bool operator!=(const Transform &t) const { return t.m != m; }

  /// \brief Sets this transform back to identity
  HERMES_CPU_GPU void reset();
  /// \brief Checks if this transform swaps coordinate system handedness
  /// \return true if this transformation changes the coordinate system
  /// handedness
  HERMES_NODISCARD HERMES_CPU_GPU bool swapsHandedness() const;
  /// \brief Gets translation vector
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU vec3 getTranslate() const {
    return vec3(m[0][3], m[1][3], m[2][3]);
  }
  /// \brief Checks if this transform is identity
  /// \return
  HERMES_CPU_GPU bool isIdentity() { return m.isIdentity(); }
  /// \brief Applies transform to point (array)
  /// \param p
  /// \param r
  /// \param d
  HERMES_CPU_GPU void applyToPoint(const real_t *p, real_t *r,
                                   size_t d = 3) const {
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

  /// \brief Gets raw matrix pointer
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU const real_t *c_matrix() const {
    return &m[0][0];
  }
  /// \brief Gets transformation matrix
  HERMES_NODISCARD HERMES_CPU_GPU const math::mat4 &matrix() const { return m; }
  /// \brief Gets upper left matrix
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU math::mat3 upperLeftMatrix() const {
    return math::mat3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2],
                      m[2][0], m[2][1], m[2][2]);
  }
  /// \brief Gets transformation matrix row
  /// \param row_index
  /// \return
  HERMES_CPU_GPU const real_t *operator[](u32 row_index) const {
    return m[row_index];
  }
  /// \brief Gets transformation matrix row
  /// \param row_index
  /// \return
  HERMES_CPU_GPU real_t *operator[](u32 row_index) { return m[row_index]; }

  /// \brief Check for nans
  /// \return
  HERMES_CPU_GPU HERMES_NODISCARD bool hasNaNs() const {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        if (numbers::is_nan(m[i][j]))
          return true;
    return false;
  }

protected:
  math::mat4 m; //!< transformation matrix
};

} // namespace hermes::geo

namespace hermes {

HERMES_TYPE_LAYOUT_METHODS(geo::Transform2, f32, 9)
HERMES_TYPE_LAYOUT_METHODS(geo::Transform, f32, 16)

template <> struct DebugTraits<geo::transform_option_bits> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::transform_option_bits &data) {
    DebugMessage m;
#define BIT_NAME(B) else if (geo::transform_option_bits::B == data) m.addFmt(#B)
    if (geo::transform_option_bits::x_right == data)
      m.addFmt("x_right");
    BIT_NAME(y_right);
    BIT_NAME(z_right);
    BIT_NAME(left_handed);
    BIT_NAME(x_left);
    BIT_NAME(y_left);
    BIT_NAME(z_left);
    BIT_NAME(right_handed);
    BIT_NAME(x_up);
    BIT_NAME(y_up);
    BIT_NAME(z_up);
    BIT_NAME(zero_to_one);
    BIT_NAME(x_down);
    BIT_NAME(y_down);
    BIT_NAME(z_down);
    BIT_NAME(transpose);
    BIT_NAME(flip_x);
    BIT_NAME(flip_y);
    BIT_NAME(flip_z);
    return m;
#undef BIT_NAME
  }
};

template <> struct DebugTraits<geo::transform_options> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::transform_options &data) {
    DebugMessage m;
    std::vector<std::string> values;
#define CHECK_BIT(B)                                                           \
  if (geo::transform_option_bits::B & data)                                    \
  values.push_back(#B)
    CHECK_BIT(x_right);
    CHECK_BIT(y_right);
    CHECK_BIT(z_right);
    CHECK_BIT(left_handed);
    CHECK_BIT(x_left);
    CHECK_BIT(y_left);
    CHECK_BIT(z_left);
    CHECK_BIT(right_handed);
    CHECK_BIT(x_up);
    CHECK_BIT(y_up);
    CHECK_BIT(z_up);
    CHECK_BIT(zero_to_one);
    CHECK_BIT(x_down);
    CHECK_BIT(y_down);
    CHECK_BIT(z_down);
    CHECK_BIT(transpose);
    CHECK_BIT(flip_x);
    CHECK_BIT(flip_y);
    CHECK_BIT(flip_z);
    return DebugMessage("{}", hermes::cstr::join(values, " | "));
#undef CHECK_BIT
  }
};

template <> struct DebugTraits<geo::Transform2> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Transform2 &data) {
    DebugMessage m;
    for (int row = 0; row < 3; ++row) {
      m.addFmt("[{}, {}, {}]\n", data[row][0], data[row][1], data[row][2]);
    }
    return m;
  }
};

template <> struct DebugTraits<geo::Transform> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Transform &data) {
    DebugMessage m;
    for (int row = 0; row < 4; ++row) {
      m.addFmt("[{}, {}, {}, {}]\n", data[row][0], data[row][1], data[row][2],
               data[row][3]);
    }
    return m;
  }
};

} // namespace hermes
