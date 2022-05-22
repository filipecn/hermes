#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnreachableCode"
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

#include <hermes/geometry/transform.h>
#include <hermes/numeric/numeric.h>
#include <hermes/geometry/quaternion.h>

namespace hermes {

HERMES_DEVICE_CALLABLE Transform2::Transform2() { m.setIdentity(); }

HERMES_DEVICE_CALLABLE Transform2::Transform2(const mat3 &mat)
    : m(mat) {}

HERMES_DEVICE_CALLABLE Transform2::Transform2(const bbox2 &bbox) {
  m[0][0] = bbox.upper[0] - bbox.lower[0];
  m[1][1] = bbox.upper[1] - bbox.lower[1];
  m[0][2] = bbox.lower[0];
  m[1][2] = bbox.lower[1];
}

HERMES_DEVICE_CALLABLE void Transform2::reset() { m.setIdentity(); }

HERMES_DEVICE_CALLABLE Transform2 Transform2::rotate(real_t angle) {
  real_t sin_a = sinf(Trigonometry::degrees2radians(angle));
  real_t cos_a = cosf(Trigonometry::degrees2radians(angle));
  mat3 m(cos_a, -sin_a, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 1.f);
  return m;
}

HERMES_DEVICE_CALLABLE Transform2 Transform2::translate(const vec2 &v) {
  mat3 m(1.f, 0.f, v.x, 0.f, 1.f, v.y, 0.f, 0.f, 1.f);
  return m;
}

HERMES_DEVICE_CALLABLE Transform::Transform() { m.setIdentity(); }

HERMES_DEVICE_CALLABLE Transform::Transform(const mat4 &mat) : m(mat) {}

HERMES_DEVICE_CALLABLE Transform::Transform(const real_t mat[4][4]) {
  m = mat4(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0], mat[1][1],
           mat[1][2], mat[1][3], mat[2][0], mat[2][1], mat[2][2], mat[2][3],
           mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
}

HERMES_DEVICE_CALLABLE Transform::Transform(const bbox3 &bbox) {
  m[0][0] = bbox.upper[0] - bbox.lower[0];
  m[1][1] = bbox.upper[1] - bbox.lower[1];
  m[2][2] = bbox.upper[2] - bbox.lower[2];
  m[0][3] = bbox.lower[0];
  m[1][3] = bbox.lower[1];
  m[2][3] = bbox.lower[2];
}

HERMES_DEVICE_CALLABLE void Transform::reset() { m.setIdentity(); }

HERMES_DEVICE_CALLABLE bool Transform::swapsHandedness() const {
  real_t det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])) -
      (m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])) +
      (m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));
  return det < 0;
}

HERMES_DEVICE_CALLABLE Transform2 Transform2::scale(const vec2 &s) {
  mat3 m(s.x, 0, 0, 0, s.y, 0, 0, 0, 1);
  mat3 inv(1.f / s.x, 0, 0, 0, 1.f / s.y, 0, 0, 0, 1);
  return {m};
}

HERMES_DEVICE_CALLABLE Transform segmentToSegmentTransform(point3 a, point3 b, point3 c, point3 d) {
  HERMES_UNUSED_VARIABLE(a);
  HERMES_UNUSED_VARIABLE(b);
  HERMES_UNUSED_VARIABLE(c);
  HERMES_UNUSED_VARIABLE(d);
  // Consider two bases a b e f and c d g h
  // TODO implement
  return {};
}

HERMES_DEVICE_CALLABLE Transform inverse(const Transform &t) { return inverse(t.m); }

HERMES_DEVICE_CALLABLE Transform Transform::lookAt(const point3 &eye, const point3 &target, const vec3 &up,
                                                   transform_options options) {
  auto right_handed = HERMES_MASK_BIT(options, transform_options::right_handed);

  Matrix4x4<real_t> m;
  vec3 v;
  if (right_handed)
    v = normalize(eye - target);
  else
    v = normalize(target - eye);
  auto r = -normalize(cross(v, up));
  auto u = cross(v, r);
  auto t = eye - point3();
  // row 0
  m[0][0] = r.x;
  m[0][1] = r.y;
  m[0][2] = r.z;
  m[0][3] = (right_handed ? 1. : -1.) * dot(t, r);
  // row 1
  m[1][0] = u.x;
  m[1][1] = u.y;
  m[1][2] = u.z;
  m[1][3] = (right_handed ? 1. : -1.) * dot(t, u);
  // row 2
  m[2][0] = v.x;
  m[2][1] = v.y;
  m[2][2] = v.z;
  m[2][3] = (right_handed ? 1. : -1.) * dot(t, v);
  // row 3
  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;
  return {m};
}

HERMES_DEVICE_CALLABLE Transform Transform::ortho(real_t left,
                                                  real_t right,
                                                  real_t bottom,
                                                  real_t top,
                                                  real_t near,
                                                  real_t far,
                                                  transform_options options) {
  const auto zero_to_one = HERMES_MASK_BIT(options, transform_options::zero_to_one);
  auto right_handed = HERMES_MASK_BIT(options, transform_options::right_handed);
  auto flip_y = HERMES_MASK_BIT(options, transform_options::flip_y);

  auto w_inv = 1 / (right - left);
  auto h_inv = 1 / (top - bottom);
  auto d_inv = 1 / (far - near);
  Matrix4x4<real_t> m;
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
  m[2][3] = -(zero_to_one ? near : (far + near)) * d_inv;
  // row 3
  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;

  return {m};
}

HERMES_DEVICE_CALLABLE Transform Transform::perspective(real_t fovy_in_degrees,
                                                        real_t aspect_ratio,
                                                        real_t near,
                                                        real_t far,
                                                        transform_options options) {
  const auto zero_to_one = HERMES_MASK_BIT(options, transform_options::zero_to_one);
  auto right_handed = HERMES_MASK_BIT(options, transform_options::right_handed);
  auto flip_y = HERMES_MASK_BIT(options, transform_options::flip_y);

  auto y_scale = 1.f / std::tan(Trigonometry::degrees2radians(fovy_in_degrees) * 0.5f);
  auto d_inv = 1 / (far - near);

  Matrix4x4<real_t> m;
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
  m[2][2] = (right_handed ? -1.f : 1.f) * (zero_to_one ? far : (far + near)) * d_inv;
  m[2][3] = (right_handed ? 1.f : -1.f) * (zero_to_one ? 1.f : 2.f) * near * far * d_inv;
  // row 3
  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = right_handed ? -1 : 1;
  m[3][3] = 0;
  return {m};
}

HERMES_DEVICE_CALLABLE Transform Transform::translate(const vec3 &d) {
  mat4 m(1.f, 0.f, 0.f, d.x, 0.f, 1.f, 0.f, d.y, 0.f, 0.f, 1.f, d.z, 0.f, 0.f,
         0.f, 1.f);
  mat4 m_inv(1.f, 0.f, 0.f, -d.x, 0.f, 1.f, 0.f, -d.y, 0.f, 0.f, 1.f, -d.z, 0.f,
             0.f, 0.f, 1.f);
  return {m};
}

HERMES_DEVICE_CALLABLE Transform Transform::scale(real_t x, real_t y, real_t z) {
  mat4 m(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
  mat4 inv(1.f / x, 0, 0, 0, 0, 1.f / y, 0, 0, 0, 0, 1.f / z, 0, 0, 0, 0, 1);
  return {m};
}

HERMES_DEVICE_CALLABLE Transform Transform::rotateX(real_t angle_in_radians) {
  real_t sin_a = sinf(angle_in_radians);
  real_t cos_a = cosf(angle_in_radians);
  mat4 m(1.f, 0.f, 0.f, 0.f, 0.f, cos_a, -sin_a, 0.f, 0.f, sin_a, cos_a, 0.f,
         0.f, 0.f, 0.f, 1.f);
//  return {m, transpose(m)};
  return m;
}

HERMES_DEVICE_CALLABLE Transform Transform::rotateY(real_t angle_in_radians) {
  real_t sin_a = sinf(angle_in_radians);
  real_t cos_a = cosf(angle_in_radians);
  mat4 m(cos_a, 0.f, sin_a, 0.f, 0.f, 1.f, 0.f, 0.f, -sin_a, 0.f, cos_a, 0.f,
         0.f, 0.f, 0.f, 1.f);
//  return {m, transpose(m)};
  return m;
}

HERMES_DEVICE_CALLABLE Transform Transform::rotateZ(real_t angle_in_radians) {
  real_t sin_a = sinf(angle_in_radians);
  real_t cos_a = cosf(angle_in_radians);
  mat4 m(cos_a, -sin_a, 0.f, 0.f, sin_a, cos_a, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f,
         0.f, 0.f, 0.f, 1.f);
//  return {m, transpose(m)};
  return m;
}

HERMES_DEVICE_CALLABLE Transform Transform::rotate(real_t angle_in_radians, const vec3 &axis) {
  quat q(sinf(angle_in_radians / 2) * axis, cosf(angle_in_radians / 2));
  return {q.normalized().matrix()};

  vec3 a = normalize(axis);
  real_t s = sinf(angle_in_radians);
  real_t c = cosf(angle_in_radians);
  real_t m[4][4];

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

  mat4 mat(m);
  return mat;
}

HERMES_DEVICE_CALLABLE Transform Transform::alignVectors(const vec3 &a, const vec3 &b) {
  // based on
  // https://www.theochem.ru.nl/%7Epwormer/Knowino/knowino.org/wiki/Rotation_matrix.html#Vector_rotation

  auto m = mat4::I();

  vec3 na = normalize(a);
  vec3 nb = normalize(b);

  vec3 u = cross(na, nb);
  auto c = dot(na, nb);
  // parallel case
  if (Check::is_equal(1.f, c))
    return Transform();
  // anti-parallel case
  if (Check::is_equal(-1.f, c)) {
    if (Check::is_equal(na.z, 1.f)) {
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

} // namespace hermes

#pragma clang diagnostic pop