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
#include <type_traits>

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

template <typename T = real_t> class Transform3;
template <typename T = real_t> class Transform2;
class bbox2;

/// \brief Computes inverse of a given transform
/// \param t
/// \return
template <typename T>
HERMES_CPU_GPU Transform3<T> inverse(const Transform3<T> &t);
/// \brief Computes inverse of a given transform
/// \param t
/// \return
template <typename T>
HERMES_CPU_GPU Transform2<T> inverse(const Transform2<T> &t);

// *****************************************************************************
//                                                                 Transform2
// *****************************************************************************
/// \brief Represents a 2-dimensional transformation
template <typename T> class Transform2 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value,
                "Transform2 supports only f32 and f64");

public:
  using scalar_type = T;
  using vector_type = Vector2<T>;
  using point_type = Point2<T>;
  using matrix_type = math::MatrixNxM<T, 3, 3>;
  /// \brief Creates scale transform
  /// \param s
  /// \return
  HERMES_CPU_GPU static Transform2 scale(const vector_type &s) {
    return matrix_type(s.x, T(0), T(0), T(0), s.y, T(0), T(0), T(0), T(1));
  }
  /// \brief Creates rotation transform
  /// \param angle
  /// \return
  HERMES_CPU_GPU static Transform2 rotate(T angle) {
    T a = math::degrees2radians(angle);
    T sn = static_cast<T>(sin(a)), cs = static_cast<T>(std::cos(a));
    return matrix_type(cs, -sn, T(0), sn, cs, T(0), T(0), T(0), T(1));
  }
  /// \brief Creates translation transform
  /// \param v
  /// \return
  HERMES_CPU_GPU static Transform2 translate(const vector_type &v) {
    return matrix_type(T(1), T(0), v.x, T(0), T(1), v.y, T(0), T(0), T(1));
  }

  /// \brief Gets inverse transform from t
  /// \param t
  /// \return
  /// \brief Default constructor
  HERMES_CPU_GPU Transform2() { m.setIdentity(); }
  /// \brief Constructs from matrix
  /// \param mat
  HERMES_CPU_GPU Transform2(const matrix_type &mat) : m(mat) {}
  /// \brief Constructs from bounding box
  /// \param bbox

  /// \brief Applies this transform to geometric point
  /// \param p
  /// \param r
  HERMES_CPU_GPU void operator()(const point_type &p, point_type *r) const {
    T x = p.x, y = p.y;
    r->x = m[0][0] * x + m[0][1] * y + m[0][2];
    r->y = m[1][0] * x + m[1][1] * y + m[1][2];
    T wp = m[2][0] * x + m[2][1] * y + m[2][2];
    if (wp != 1.f)
      *r /= wp;
  }
  /// \brief Applies this transform to geometric vector
  /// \param v
  /// \param r
  HERMES_CPU_GPU void operator()(const vector_type &v, vector_type *r) const {
    T x = v.x, y = v.y;
    r->x = m[0][0] * x + m[0][1] * y;
    r->y = m[1][0] * x + m[1][1] * y;
  }
  /// \brief Applies this transform to geometric vector
  /// \param v
  /// \return
  HERMES_CPU_GPU vector_type operator()(const vector_type &v) const {
    T x = v.x, y = v.y;
    return vector_type(m[0][0] * x + m[0][1] * y, m[1][0] * x + m[1][1] * y);
  }
  /// \brief Applies this transform to geometric point
  /// \param p
  /// \return
  HERMES_CPU_GPU point_type operator()(const point_type &p) const {
    T x = p.x, y = p.y;
    T xp = m[0][0] * x + m[0][1] * y + m[0][2];
    T yp = m[1][0] * x + m[1][1] * y + m[1][2];
    T wp = m[2][0] * x + m[2][1] * y + m[2][2];
    if (wp == 1.f)
      return point_type(xp, yp);
    return point_type(xp / wp, yp / wp);
  }
  /// \brief Applies this transform to geometric box
  /// \param b
  /// \return
  // HERMES_CPU_GPU bbox2 operator()(const bbox2 &b) const {
  //   const Transform2<T> &M = *this;
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
  HERMES_CPU_GPU void reset() { m.setIdentity(); }
  /// \brief Extracts translation vector
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU vector_type getTranslate() const {
    return vector_type(m[0][2], m[1][2]);
  }
  /// \brief Extracts scale vector
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU vector_type getScale() const {
    return {0, 0};
  }
  /// \brief Extracts rotation matrix
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU matrix_type getMatrix() const { return m; }

  /// \brief Gets transform matrix row
  /// \param row_index
  /// \return
  HERMES_CPU_GPU const T *operator[](u32 row_index) const {
    return m[row_index];
  }
  /// \brief Gets transform matrix row
  /// \param row_index
  /// \return
  HERMES_CPU_GPU T *operator[](u32 row_index) { return m[row_index]; }

private:
  matrix_type m;
};

template <typename T>
HERMES_CPU_GPU Transform2<T> inverse(const Transform2<T> &t) {
  return inverse(t.getMatrix());
}

// *****************************************************************************
//                                                                  Transform3
// *****************************************************************************
/// \brief Represents a 3-dimensional transformation
template <typename T> class Transform3 {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value,
                "Transform3 supports only f32 and f64");

public:
  using scalar_type = T;
  using vector_type = Vector3<T>;
  using point_type = Point3<T>;
  using point2_type = Point2<T>;
  using matrix_type = math::MatrixNxM<T, 4, 4>;
  /// Creates a Look At Transform3
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
  HERMES_CPU_GPU static Transform3
  lookAt(const point_type &eye, const point_type &target = {0, 0, 0},
         const vector_type &up = {0, 1, 0},
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
  /// \note Note that n > f. This function negates the values of near_plane and
  /// far_plane in
  ///       case the given values are f > n. Because by default, this transform
  ///       uses a left-handed coordinate system.
  /// \param left
  /// \param right
  /// \param bottom
  /// \param top
  /// \param near_plane
  /// \param far_plane
  /// \param options
  /// \return
  HERMES_CPU_GPU static Transform3
  ortho(T left, T right, T bottom, T top, T near_plane, T far_plane,
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
  /// \param near_plane
  /// \param far_plane
  /// \param options
  /// \return
  HERMES_CPU_GPU static Transform3
  perspective(T fovy_in_degrees, T aspect_ratio, T near_plane, T far_plane,
              transform_options options = transform_option_bits::left_handed);
  /// \brief Creates a scale transform
  /// \param x
  /// \param y
  /// \param z
  /// \return
  HERMES_CPU_GPU static Transform3 scale(T x, T y, T z);
  /// \brief Creates a scale transform
  /// \param s
  /// \return
  HERMES_CPU_GPU static Transform3 scale(const vector_type &s);
  /// \brief Creates a translation transform
  /// \param d
  /// \return
  HERMES_CPU_GPU static Transform3 translate(const vector_type &d);
  /// \brief Creates a x-axis rotation transform
  /// \param angle_in_radians
  /// \return
  HERMES_CPU_GPU static Transform3 rotateX(T angle_in_radians);
  /// \brief Creates a y-axis rotation transform
  /// \param angle_in_radians
  /// \return
  HERMES_CPU_GPU static Transform3 rotateY(T angle_in_radians);
  /// \brief Creates a z-axis rotation transform
  /// \param angle_in_radians
  /// \return
  HERMES_CPU_GPU static Transform3 rotateZ(T angle_in_radians);
  /// \brief Creates a arbitrary-axis rotation transform
  /// \param angle_in_radians
  /// \param axis
  /// \return
  HERMES_CPU_GPU static Transform3 rotate(T angle_in_radians,
                                          const vector_type &axis);
  /// \brief Creates a transform that aligns vector a to vector b
  /// \param a source vector
  /// \param b destination vector
  /// \return
  HERMES_CPU_GPU static Transform3 alignVectors(const vector_type &a,
                                                const vector_type &b);

  /// \brief Default constructor
  HERMES_CPU_GPU Transform3() { m.setIdentity(); }
  /// \brief Constructs from matrix
  /// \param mat
  HERMES_CPU_GPU Transform3(const matrix_type &mat);
  /// \brief Constructs from array matrix
  /// \param mat
  HERMES_CPU_GPU explicit Transform3(const T mat[4][4]);
  /// \brief Constructs from geometric box
  /// \param bbox
  // HERMES_CPU_GPU Transform3(const bbox3 &bbox);

  /// \brief Applies this transform to geometric box
  /// \param b
  /// \return
  // HERMES_CPU_GPU bbox3 operator()(const bbox3 &b) const {
  //   const Transform3 &M = *this;
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
  HERMES_CPU_GPU point_type operator()(const point2_type &p) const {
    T x = p.x, y = p.y, z = 0;
    T xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    T yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    T zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    T wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp == 1.f)
      return point_type(xp, yp, zp);
    return point_type(xp, yp, zp) / wp;
  }
  /// \brief Applies this transform to geometric point
  /// \param p
  /// \return
  HERMES_CPU_GPU point_type operator()(const point_type &p) const {
    T x = p.x, y = p.y, z = p.z;
    T xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    T yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    T zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    T wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp == 1.f)
      return point_type(xp, yp, zp);
    return point_type(xp, yp, zp) / wp;
  }
  /// \brief Applies this transform to geometric point
  /// \param p
  /// \param r
  HERMES_CPU_GPU void operator()(const point_type &p, point_type *r) const {
    T x = p.x, y = p.y, z = p.z;
    r->x = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    r->y = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    r->z = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    T wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp != 1.f)
      *r /= wp;
  }
  /// \brief Applies this transform to geometric point
  /// \param v
  /// \return
  HERMES_CPU_GPU vector_type operator()(const vector_type &v) const {
    T x = v.x, y = v.y, z = v.z;
    return vector_type(m[0][0] * x + m[0][1] * y + m[0][2] * z,
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
  HERMES_CPU_GPU Transform3 &operator=(const Transform2<T> &t) {
    m.setIdentity();
    auto m3 = t.getMatrix();
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
  HERMES_CPU_GPU Transform3 operator*(const Transform3 &t) const {
    matrix_type m1 = m * t.m;
    return {m1};
  }
  /// \brief Applies this transform to geometric vector
  /// \param p
  /// \return
  HERMES_CPU_GPU point_type operator*(const point_type &p) const {
    return (*this)(p);
  }
  //                                                                                                          boolean
  HERMES_CPU_GPU bool operator==(const Transform3 &t) const { return t.m == m; }
  HERMES_CPU_GPU bool operator!=(const Transform3 &t) const { return t.m != m; }

  /// \brief Sets this transform back to identity
  HERMES_CPU_GPU void reset() { m.setIdentity(); }
  /// \brief Checks if this transform swaps coordinate system handedness
  /// \return true if this transformation changes the coordinate system
  /// handedness
  HERMES_NODISCARD HERMES_CPU_GPU bool swapsHandedness() const;
  /// \brief Gets translation vector
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU vector_type getTranslate() const {
    return vector_type(m[0][3], m[1][3], m[2][3]);
  }
  /// \brief Checks if this transform is identity
  /// \return
  HERMES_CPU_GPU bool isIdentity() { return m.isIdentity(); }
  /// \brief Applies transform to point (array)
  /// \param p
  /// \param r
  /// \param d
  HERMES_CPU_GPU void applyToPoint(const T *p, T *r, size_t d = 3) const {
    T x = p[0], y = p[1], z = 0;
    if (d == 3)
      z = p[2];
    r[0] = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    r[1] = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    if (d == 3)
      r[2] = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    T wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp != 1.f) {
      T invwp = T(1) / wp;
      r[0] *= invwp;
      r[1] *= invwp;
      if (d == 3)
        r[2] *= invwp;
    }
  }

  /// \brief Gets raw matrix pointer
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU const T *c_matrix() const { return &m[0][0]; }
  /// \brief Gets transformation matrix
  HERMES_NODISCARD HERMES_CPU_GPU const matrix_type &matrix() const {
    return m;
  }
  /// \brief Gets upper left matrix
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU math::MatrixNxM<T, 3, 3>
  upperLeftMatrix() const {
    return math::MatrixNxM<T, 3, 3>(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1],
                                    m[1][2], m[2][0], m[2][1], m[2][2]);
  }
  /// \brief Gets transformation matrix row
  /// \param row_index
  /// \return
  HERMES_CPU_GPU const T *operator[](u32 row_index) const {
    return m[row_index];
  }
  /// \brief Gets transformation matrix row
  /// \param row_index
  /// \return
  HERMES_CPU_GPU T *operator[](u32 row_index) { return m[row_index]; }

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
  matrix_type m; //!< transformation matrix
};

template <typename T>
HERMES_CPU_GPU Transform3<T>::Transform3(const math::MatrixNxM<T, 4, 4> &mat)
    : m(mat) {}

template <typename T>
HERMES_CPU_GPU Transform3<T>::Transform3(const T mat[4][4]) {
  m = math::MatrixNxM<T, 4, 4>(mat[0][0], mat[0][1], mat[0][2], mat[0][3],
                               mat[1][0], mat[1][1], mat[1][2], mat[1][3],
                               mat[2][0], mat[2][1], mat[2][2], mat[2][3],
                               mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
}

// template <typename T> HERMES_CPU_GPU Transform3<T>::Transform3(const bbox3
// &bbox) {
//   m[0][0] = bbox.upper[0] - bbox.lower[0];
//   m[1][1] = bbox.upper[1] - bbox.lower[1];
//   m[2][2] = bbox.upper[2] - bbox.lower[2];
//   m[0][3] = bbox.lower[0];
//   m[1][3] = bbox.lower[1];
//   m[2][3] = bbox.lower[2];
// }

template <typename T>
HERMES_CPU_GPU bool Transform3<T>::swapsHandedness() const {
  T det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])) -
          (m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])) +
          (m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));
  return det < 0;
}

template <typename T>
HERMES_CPU_GPU Transform3<T>
Transform3<T>::lookAt(const Point3<T> &eye, const Point3<T> &target,
                      const Vector3<T> &up, transform_options options) {
  auto right_handed = options.contain(transform_option_bits::right_handed);

  math::MatrixNxM<T, 4, 4> m;
  Vector3<T> v = normalize(right_handed ? eye - target : target - eye);
  auto r = normalize(cross(up, v));
  auto u = cross(v, r);
  auto e = eye - Point3<T>();

  // row 0
  m[0][0] = r.x;
  m[0][1] = r.y;
  m[0][2] = r.z;
  m[0][3] = -dot(e, r);
  // row 1
  m[1][0] = u.x;
  m[1][1] = u.y;
  m[1][2] = u.z;
  m[1][3] = -dot(e, u);
  // row 2
  m[2][0] = v.x;
  m[2][1] = v.y;
  m[2][2] = v.z;
  m[2][3] = -dot(e, v);
  // row 3
  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;
  return {m};
}

template <typename T>
HERMES_CPU_GPU Transform3<T>
Transform3<T>::ortho(T left, T right, T bottom, T top, T near_plane,
                     T far_plane, transform_options options) {
  const auto zero_to_one = options.contain(transform_option_bits::zero_to_one);
  auto right_handed = options.contain(transform_option_bits::right_handed);
  auto flip_y = options.contain(transform_option_bits::flip_y);

  auto w_inv = 1 / (right - left);
  auto h_inv = 1 / (top - bottom);
  auto d_inv = 1 / (far_plane - near_plane);
  math::MatrixNxM<T, 4, 4> m;
  // row 0
  m[0][0] = 2 * w_inv;
  m[0][1] = 0;
  m[0][2] = 0;
  m[0][3] = -(right + left) * w_inv;
  // row 1
  m[1][0] = 0;
  m[1][1] = (flip_y ? -1.f : 1.f) * 2 * h_inv;
  m[1][2] = 0;
  m[1][3] = -(top + bottom) * h_inv;
  // row 2
  m[2][0] = 0;
  m[2][1] = 0;
  m[2][2] = (zero_to_one ? 1.f : 2.f) * (right_handed ? -1.f : 1.f) * d_inv;
  m[2][3] = -(zero_to_one ? near_plane : (far_plane + near_plane)) * d_inv;
  // row 3
  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;

  return {m};
}

template <typename T>
HERMES_CPU_GPU Transform3<T>
Transform3<T>::perspective(T fovy_in_degrees, T aspect_ratio, T near_plane,
                           T far_plane, transform_options options) {
  const auto zero_to_one = options.contain(transform_option_bits::zero_to_one);
  auto right_handed = options.contain(transform_option_bits::right_handed);
  auto flip_y = options.contain(transform_option_bits::flip_y);

  auto y_scale = 1.f / std::tan(math::degrees2radians(fovy_in_degrees) * 0.5f);
  auto d_inv = 1 / (near_plane - far_plane);

  math::MatrixNxM<T, 4, 4> m;
  // row 0
  m[0][0] = y_scale / aspect_ratio;
  m[0][1] = 0;
  m[0][2] = 0;
  m[0][3] = 0;
  // row 1
  m[1][0] = 0;
  m[1][1] = (flip_y ? -1.f : 1.f) * y_scale;
  m[1][2] = 0;
  m[1][3] = 0;
  // row 2
  m[2][0] = 0;
  m[2][1] = 0;
  m[2][2] = (right_handed ? -1.f : -1.f) *
            (zero_to_one ? near_plane : (far_plane + near_plane)) * d_inv;
  m[2][3] = (right_handed ? -1.f : 1.f) * (zero_to_one ? 1.f : 2.f) *
            near_plane * far_plane * d_inv;
  // row 3
  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = right_handed
                ? -1
                : 1; // this term 'copies' z into w for the perspective divide
  m[3][3] = 0;
  return {m};
}

template <typename T>
HERMES_CPU_GPU Transform3<T> Transform3<T>::translate(const Vector3<T> &d) {
  math::MatrixNxM<T, 4, 4> m(1.f, 0.f, 0.f, d.x, 0.f, 1.f, 0.f, d.y, 0.f, 0.f,
                             1.f, d.z, 0.f, 0.f, 0.f, 1.f);
  return {m};
}

template <typename T>
HERMES_CPU_GPU Transform3<T> Transform3<T>::scale(T x, T y, T z) {
  math::MatrixNxM<T, 4, 4> m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
  return {m};
}

template <typename T>
HERMES_CPU_GPU Transform3<T> Transform3<T>::scale(const Vector3<T> &s) {
  math::MatrixNxM<T, 4, 4> m(s.x, 0, 0, 0, 0, s.y, 0, 0, 0, 0, s.z, 0, 0, 0, 0,
                             1);
  return {m};
}

template <typename T>
HERMES_CPU_GPU Transform3<T> Transform3<T>::rotateX(T angle_in_radians) {
  T sin_a = std::sin(angle_in_radians);
  T cos_a = std::cos(angle_in_radians);
  math::MatrixNxM<T, 4, 4> m(1.f, 0.f, 0.f, 0.f, 0.f, cos_a, -sin_a, 0.f, 0.f,
                             sin_a, cos_a, 0.f, 0.f, 0.f, 0.f, 1.f);
  //  return {m, transpose(m)};
  return m;
}

template <typename T>
HERMES_CPU_GPU Transform3<T> Transform3<T>::rotateY(T angle_in_radians) {
  T sin_a = std::sin(angle_in_radians);
  T cos_a = std::cos(angle_in_radians);
  math::MatrixNxM<T, 4, 4> m(cos_a, 0.f, sin_a, 0.f, 0.f, 1.f, 0.f, 0.f, -sin_a,
                             0.f, cos_a, 0.f, 0.f, 0.f, 0.f, 1.f);
  //  return {m, transpose(m)};
  return m;
}

template <typename T>
HERMES_CPU_GPU Transform3<T> Transform3<T>::rotateZ(T angle_in_radians) {
  T sin_a = std::sin(angle_in_radians);
  T cos_a = std::cos(angle_in_radians);
  math::MatrixNxM<T, 4, 4> m(cos_a, -sin_a, 0.f, 0.f, sin_a, cos_a, 0.f, 0.f,
                             0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f);
  //  return {m, transpose(m)};
  return m;
}

template <typename T>
HERMES_CPU_GPU Transform3<T> Transform3<T>::rotate(T angle_in_radians,
                                                   const Vector3<T> &axis) {
  Vector3<T> a = normalize(axis);
  T s = std::sin(angle_in_radians);
  T c = std::cos(angle_in_radians);
  T m[4][4];

  m[0][0] = a.x * a.x + (1.f - a.x * a.x) * c;
  m[0][1] = a.x * a.y * (1.f - c) - a.z * s;
  m[0][2] = a.x * a.z * (1.f - c) + a.y * s;
  m[0][3] = 0;

  m[1][0] = a.x * a.y * (1.f - c) + a.z * s;
  m[1][1] = a.y * a.y + (1.f - a.y * a.y) * c;
  m[1][2] = a.y * a.z * (1.f - c) - a.x * s;
  m[1][3] = 0;

  m[2][0] = a.x * a.z * (1.f - c) - a.y * s;
  m[2][1] = a.y * a.z * (1.f - c) + a.x * s;
  m[2][2] = a.z * a.z + (1.f - a.z * a.z) * c;
  m[2][3] = 0;

  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;

  math::MatrixNxM<T, 4, 4> mat(m);
  return mat;
}

template <typename T>
HERMES_CPU_GPU Transform3<T> Transform3<T>::alignVectors(const Vector3<T> &a,
                                                         const Vector3<T> &b) {
  // based on
  // https://www.theochem.ru.nl/%7Epwormer/Knowino/knowino.org/wiki/Rotation_matrix.html#Vector_rotation

  auto m = math::MatrixNxM<T, 4, 4>::I();

  Vector3<T> na = normalize(a);
  Vector3<T> nb = normalize(b);

  Vector3<T> u = cross(na, nb);
  auto c = dot(na, nb);
  // parallel case
  if (numbers::cmp::is_equal(1., c))
    return Transform3<T>();
  // anti-parallel case
  if (numbers::cmp::is_equal(-1., c)) {
    if (numbers::cmp::is_equal(na.z, 1.)) {
      m[0][0] = -1;
      m[2][2] = -1;
    } else {
      auto fxx = na.x * na.x;
      auto fyy = na.y * na.y;
      auto fzz = na.z * na.z;
      auto fxy = na.x * na.y;
      auto h = 1 / (1 - fzz);
      m[0][0] = -(fxx - fyy);
      m[0][1] = -2 * fxy;
      m[1][0] = -2 * fxy;
      m[1][1] = (fxx - fyy);
      m[2][2] = -(1 - fzz);
      m *= h;
    }
  } else {
    auto h = (1 - c) / (1 - c * c);
    m[0][0] = c + h * u.x * u.x;
    m[0][1] = h * u.x * u.y - u.z;
    m[0][2] = h * u.x * u.z + u.y;

    m[1][0] = h * u.x * u.y + u.z;
    m[1][1] = c + h * u.y * u.y;
    m[1][2] = h * u.y * u.z - u.x;

    m[2][0] = h * u.x * u.z - u.y;
    m[2][1] = h * u.y * u.z + u.x;
    m[2][2] = c + h * u.z * u.z;
  }
  return {m};
}

template <typename T>
HERMES_CPU_GPU Transform3<T> inverse(const Transform3<T> &t) {
  return inverse(t.matrix());
}

template <typename T>
HERMES_CPU_GPU Transform3<T>
segmentToSegmentTransform(Point3<T> a, Point3<T> b, Point3<T> c, Point3<T> d) {
  HERMES_UNUSED_VARIABLE(a);
  HERMES_UNUSED_VARIABLE(b);
  HERMES_UNUSED_VARIABLE(c);
  HERMES_UNUSED_VARIABLE(d);
  // Consider two bases a b e f and c d g h
  // TODO implement
  return {};
}

using transform2 = Transform2<real_t>;
using transform3 = Transform3<real_t>;
using transform2f = Transform2<f32>;
using transform2d = Transform2<f64>;
using transform3f = Transform3<f32>;
using transform3d = Transform3<f64>;

} // namespace hermes::geo

namespace hermes {

HERMES_TYPE_LAYOUT_METHODS(geo::Transform2<real_t>, real_t, 9)
HERMES_TYPE_LAYOUT_METHODS(geo::Transform3<real_t>, real_t, 16)

#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
HERMES_TYPE_LAYOUT_METHODS(geo::Transform2<f32>, f32, 9)
HERMES_TYPE_LAYOUT_METHODS(geo::Transform3<f32>, f32, 16)
#else
HERMES_TYPE_LAYOUT_METHODS(geo::Transform2<f64>, f64, 9)
HERMES_TYPE_LAYOUT_METHODS(geo::Transform3<f64>, f64, 16)
#endif

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

template <typename T> struct DebugTraits<geo::Transform2<T>> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Transform2<T> &data) {
    DebugMessage m;
    for (int row = 0; row < 3; ++row) {
      m.addFmt("[{}, {}, {}]\n", data[row][0], data[row][1], data[row][2]);
    }
    return m;
  }
};

template <typename T> struct DebugTraits<geo::Transform3<T>> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const geo::Transform3<T> &data) {
    DebugMessage m;
    for (int row = 0; row < 4; ++row) {
      m.addFmt("[{}, {}, {}, {}]\n", data[row][0], data[row][1], data[row][2],
               data[row][3]);
    }
    return m;
  }
};

} // namespace hermes
