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

#ifndef HERMES_GEOMETRY_MATRIX_H
#define HERMES_GEOMETRY_MATRIX_H

#include <hermes/geometry/vector.h>
#include <vector>
#include <hermes/numeric/math_element.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                AUXILIARY FUNCTIONS
// *********************************************************************************************************************
// function extracted from MESA implementation of the GLU library
template<typename T>
HERMES_DEVICE_CALLABLE bool gluInvertMatrix(const T m[16], T invOut[16]) {
  T inv[16], det;
  int i;

  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
      m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
      m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
      m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
      m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
      m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
      m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
      m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
      m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
      m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
      m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
      m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
      m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
      m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
      m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
      m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
      m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0f / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}

// *********************************************************************************************************************
//                                                                                                          Matrix4x4
// *********************************************************************************************************************
/// 4x4 Matrix representation
/// Access: m[ROW][COLUMN]
template<typename T> class Matrix4x4 : public MathElement<T, 16> {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param is_identity [optional | def = true] initialize as an identity matrix
  HERMES_DEVICE_CALLABLE explicit Matrix4x4(bool is_identity = false) {
    std::memset(m_, 0, sizeof(m_));
    if (is_identity)
      for (int i = 0; i < 4; i++)
        m_[i][i] = 1.f;
  }
  /// \param values list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  HERMES_DEVICE_CALLABLE Matrix4x4(std::initializer_list<T> values, bool columnMajor = false) {
    size_t l = 0, c = 0;
    for (auto v : values) {
      m_[l][c] = v;
      if (columnMajor) {
        l++;
        if (l >= 4)
          l = 0, c++;
      } else {
        c++;
        if (c >= 4)
          c = 0, l++;
      }
    }
  }
  /// \param mat list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  HERMES_DEVICE_CALLABLE explicit Matrix4x4(const T mat[16], bool columnMajor = false) {
    size_t k = 0;
    if (columnMajor)
      for (int c = 0; c < 4; c++)
        for (auto &l : m_)
          l[c] = mat[k++];
    else
      for (auto &l : m_)
        for (int c = 0; c < 4; c++)
          l[c] = mat[k++];
  }
  /// \param mat matrix entries in [ROW][COLUMN] form
  HERMES_DEVICE_CALLABLE explicit Matrix4x4(T mat[4][4]) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        m_[i][j] = mat[i][j];
  }
  /// \param m00 value of entry at row 0 column 0
  /// \param m01 value of entry at row 0 column 1
  /// \param m02 value of entry at row 0 column 2
  /// \param m03 value of entry at row 0 column 3
  /// \param m10 value of entry at row 1 column 0
  /// \param m11 value of entry at row 1 column 1
  /// \param m12 value of entry at row 1 column 2
  /// \param m13 value of entry at row 1 column 3
  /// \param m20 value of entry at row 2 column 0
  /// \param m21 value of entry at row 2 column 1
  /// \param m22 value of entry at row 2 column 2
  /// \param m23 value of entry at row 2 column 3
  /// \param m30 value of entry at row 3 column 0
  /// \param m31 value of entry at row 3 column 1
  /// \param m32 value of entry at row 3 column 2
  /// \param m33 value of entry at row 3 column 3
  HERMES_DEVICE_CALLABLE Matrix4x4(T m00, T m01, T m02, T m03, T m10, T m11, T m12, T m13, T m20,
                                   T m21, T m22, T m23, T m30, T m31, T m32, T m33) {
    m_[0][0] = m00;
    m_[0][1] = m01;
    m_[0][2] = m02;
    m_[0][3] = m03;
    m_[1][0] = m10;
    m_[1][1] = m11;
    m_[1][2] = m12;
    m_[1][3] = m13;
    m_[2][0] = m20;
    m_[2][1] = m21;
    m_[2][2] = m22;
    m_[2][3] = m23;
    m_[3][0] = m30;
    m_[3][1] = m31;
    m_[3][2] = m32;
    m_[3][3] = m33;
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE Matrix4x4<T> operator*(const Matrix4x4<T> &B) const {
    Matrix4x4<T> r;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        r[i][j] = m_[i][0] * B[0][j] + m_[i][1] * B[1][j] +
            m_[i][2] * B[2][j] + m_[i][3] * B[3][j];
    return r;
  }
  HERMES_DEVICE_CALLABLE Vector4 <T> operator*(const Vector4 <T> &v) const {
    Vector4<T> r;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        r[i] += m_[i][j] * v[j];
    return r;
  }
  //                                                                                                          boolean
  HERMES_DEVICE_CALLABLE bool operator==(const Matrix4x4<T> &B) const {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        if (!Check::is_equal(m_[i][j], B[i][j]))
          return false;
    return true;
  }
  HERMES_DEVICE_CALLABLE bool operator!=(const Matrix4x4<T> &B) const {
    return !((*this) == B);
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  ///
  HERMES_DEVICE_CALLABLE void setIdentity() {
#ifndef HERMES_DEVICE_CODE
    for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      m_[i][j] = 0.f;
#else
    std::memset(m_, 0, sizeof(m_));
#endif
    for (int i = 0; i < 4; i++)
      m_[i][i] = 1.f;
  }
  ///
  /// \param a
  HERMES_DEVICE_CALLABLE void row_major(T *a) const {
    int k = 0;
    for (auto &i : m_)
      for (int j = 0; j < 4; j++)
        a[k++] = i[j];
  }
  ///
  /// \param a
  HERMES_DEVICE_CALLABLE void column_major(T *a) const {
    int k = 0;
    for (int i = 0; i < 4; i++)
      for (auto &j : m_)
        a[k++] = j[i];
  }
  ///
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool isIdentity() const {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        if ((i != j && !Check::is_equal(m_[i][j], 0.f)) ||
            (i == j && !Check::is_equal(m_[i][j], 1.f)))
          return false;
    return true;
  }
  // *******************************************************************************************************************
  //                                                                                                            ACCESS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE T *operator[](u32 row_index) { return m_[row_index]; }
  HERMES_DEVICE_CALLABLE const T *operator[](u32 row_index) const { return m_[row_index]; }

private:
  T m_[4][4];
};

// *********************************************************************************************************************
//                                                                                                          Matrix3x3
// *********************************************************************************************************************
template<typename T> class Matrix3x3 : public MathElement<T, 9> {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Matrix3x3() { std::memset(m_, 0, sizeof(m_)); }
  HERMES_DEVICE_CALLABLE Matrix3x3(vec3 a, vec3 b, vec3 c)
      : Matrix3x3(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z) {}
  HERMES_DEVICE_CALLABLE Matrix3x3(T m00, T m01, T m02, T m10, T m11, T m12, T m20, T m21, T m22) {
    m_[0][0] = m00;
    m_[0][1] = m01;
    m_[0][2] = m02;
    m_[1][0] = m10;
    m_[1][1] = m11;
    m_[1][2] = m12;
    m_[2][0] = m20;
    m_[2][1] = m21;
    m_[2][2] = m22;
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE
  Vector3 <T> operator*(const Vector3 <T> &v) const {
    Vector3<T> r;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        r[i] += m_[i][j] * v[j];
    return r;
  }
  HERMES_DEVICE_CALLABLE
  Matrix3x3<T> operator*(const Matrix3x3<T> &B) const {
    Matrix3x3<T> r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        r[i][j] =
            m_[i][0] * B[0][j] + m_[i][1] * B[1][j] + m_[i][2] * B[2][j];
    return r;
  }
  HERMES_DEVICE_CALLABLE
  Matrix3x3<T> operator*(const T &f) const {
    return Matrix3x3<T>(m_[0][0] * f, m_[0][1] * f, m_[0][2] * f, m_[1][0] * f,
                        m_[1][1] * f, m_[1][2] * f, m_[2][0] * f, m_[2][1] * f,
                        m_[2][2] * f);
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE void setIdentity() {
    std::memset(m_, 0, sizeof(m_));
    for (int i = 0; i < 3; i++)
      m_[i][i] = 1.f;
  }
  HERMES_DEVICE_CALLABLE T determinant() const {
    return m_[0][0] * m_[1][1] * m_[2][2] + m_[0][1] * m_[1][2] * m_[2][0] +
        m_[0][2] * m_[1][0] * m_[2][1] - m_[2][0] * m_[1][1] * m_[0][2] -
        m_[2][1] * m_[1][2] * m_[0][0] - m_[2][2] * m_[1][0] * m_[0][1];
  }
  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE T *operator[](u32 row_index) { return m_[row_index]; }
  HERMES_DEVICE_CALLABLE const T *operator[](u32 row_index) const { return m_[row_index]; }

private:
  T m_[3][3];
};

// *********************************************************************************************************************
//                                                                                                          Matrix2x2
// *********************************************************************************************************************
template<typename T> class Matrix2x2 : public MathElement<T, 4> {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Matrix2x2() {
    std::memset(m_, 0, sizeof(m_));
    for (int i = 0; i < 2; i++)
      m_[i][i] = 1.f;
  }
  HERMES_DEVICE_CALLABLE Matrix2x2(T m00, T m01, T m10, T m11) {
    m_[0][0] = m00;
    m_[0][1] = m01;
    m_[1][0] = m10;
    m_[1][1] = m11;
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE Vector2 <T> operator*(const Vector2 <T> &v) const {
    Vector2<T> r;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
        r[i] += m_[i][j] * v[j];
    return r;
  }
  HERMES_DEVICE_CALLABLE Matrix2x2<T> operator*(const Matrix2x2<T> &B) const {
    Matrix2x2<T> r;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        r[i][j] = m_[i][0] * B[0][j] + m_[i][1] * B[1][j];
    return r;
  }
  HERMES_DEVICE_CALLABLE Matrix2x2<T> operator*(T f) const {
    return Matrix2x2<T>(m_[0][0] * f, m_[0][1] * f, m_[1][0] * f,
                        m_[1][1] * f);
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE void setIdentity() {
    std::memset(m_, 0, sizeof(m_));
    for (int i = 0; i < 2; i++)
      m_[i][i] = 1.f;
  }
  HERMES_DEVICE_CALLABLE T determinant() { return m_[0][0] * m_[1][1] - m_[0][1] * m_[1][0]; }
  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE const T &operator()(u32 i, u32 j) const { return m_[i][j]; }
  HERMES_DEVICE_CALLABLE T &operator()(u32 i, u32 j) { return m_[i][j]; }
  HERMES_DEVICE_CALLABLE T *operator[](u32 row_index) { return m_[row_index]; }
  HERMES_DEVICE_CALLABLE const T *operator[](u32 row_index) const { return m_[row_index]; }

private:
  T m_[2][2];
};

// *********************************************************************************************************************
//                                                                                                EXTERNAL FUNCTIONS
// *********************************************************************************************************************
template<typename T>
HERMES_DEVICE_CALLABLE
Matrix4x4<T> rowReduce(const Matrix4x4<T> &p, const Matrix4x4<T> &q) {
  Matrix4x4<T> l = p, r = q;
  // TODO implement with gauss jordan elimination
  return r;
}
template<typename T>
HERMES_DEVICE_CALLABLE  Matrix4x4<T> transpose(const Matrix4x4<T> &m) {
  return Matrix4x4<T>(m[0][0], m[1][0], m[2][0], m[3][0], m[0][1],
                      m[1][1], m[2][1], m[3][1], m[0][2], m[1][2],
                      m[2][2], m[3][2], m[0][3], m[1][3], m[2][3],
                      m[3][3]);
}
template<typename T>
HERMES_DEVICE_CALLABLE  Matrix4x4<T> inverse(const Matrix4x4<T> &m) {
  Matrix4x4<T> r;
  T mm[16], inv[16];
  m.row_major(mm);
  if (gluInvertMatrix(mm, inv)) {
    int k = 0;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        r[i][j] = inv[k++];
    return r;
  }

  T det = m[0][0] * m[1][1] * m[2][2] * m[3][3] +
      m[1][2] * m[2][3] * m[3][1] * m[1][3] +
      m[2][1] * m[3][2] * m[1][1] * m[2][3] +
      m[3][2] * m[1][2] * m[2][1] * m[3][3] +
      m[1][3] * m[2][2] * m[3][1] * m[0][1] +
      m[0][1] * m[2][3] * m[3][2] * m[0][2] +
      m[2][1] * m[3][3] * m[0][3] * m[2][2] +
      m[3][1] * m[0][1] * m[2][2] * m[3][3] +
      m[0][2] * m[2][3] * m[3][1] * m[0][3] +
      m[2][1] * m[3][2] * m[0][2] * m[0][1] +
      m[1][2] * m[3][3] * m[0][2] * m[1][3] +
      m[3][1] * m[0][3] * m[1][1] * m[3][2] -
      m[0][1] * m[1][3] * m[3][2] * m[0][2] -
      m[1][1] * m[3][3] * m[0][3] * m[1][2] -
      m[3][1] * m[0][3] * m[0][1] * m[1][3] -
      m[2][2] * m[0][2] * m[1][1] * m[2][3] -
      m[0][3] * m[1][2] * m[2][1] * m[0][1] -
      m[1][2] * m[2][3] * m[0][2] * m[1][3] -
      m[2][1] * m[0][3] * m[1][1] * m[2][2] -
      m[1][0] * m[1][0] * m[2][3] * m[3][2] -
      m[1][2] * m[2][0] * m[3][3] * m[1][3] -
      m[2][2] * m[3][0] * m[1][0] * m[2][2] -
      m[3][3] * m[1][2] * m[2][3] * m[3][0] -
      m[1][3] * m[2][0] * m[3][2] * m[1][1];
  if (fabs(det) < 1e-8)
    return r;

  r[0][0] =
      (m[1][1] * m[2][2] * m[3][3] + m[1][2] * m[2][3] * m[3][1] +
          m[1][3] * m[2][1] * m[3][2] - m[1][1] * m[2][3] * m[3][2] -
          m[1][2] * m[2][1] * m[3][3] - m[1][3] * m[2][2] * m[3][1]) /
          det;
  r[0][1] =
      (m[0][1] * m[2][3] * m[3][2] + m[0][2] * m[2][1] * m[3][3] +
          m[0][3] * m[2][2] * m[3][1] - m[0][1] * m[2][2] * m[3][3] -
          m[0][2] * m[2][3] * m[3][1] - m[0][3] * m[2][1] * m[3][2]) /
          det;
  r[0][2] =
      (m[0][1] * m[1][2] * m[3][3] + m[0][2] * m[1][3] * m[3][1] +
          m[0][3] * m[1][1] * m[3][2] - m[0][1] * m[1][3] * m[3][2] -
          m[0][2] * m[1][1] * m[3][3] - m[0][3] * m[1][2] * m[3][1]) /
          det;
  r[0][3] =
      (m[0][1] * m[1][3] * m[2][2] + m[0][2] * m[1][1] * m[2][3] +
          m[0][3] * m[1][2] * m[2][1] - m[0][1] * m[1][2] * m[2][3] -
          m[0][2] * m[1][3] * m[2][1] - m[0][3] * m[1][1] * m[2][2]) /
          det;
  r[1][0] =
      (m[1][0] * m[2][3] * m[3][2] + m[1][2] * m[2][0] * m[3][3] +
          m[1][3] * m[2][2] * m[3][0] - m[1][0] * m[2][2] * m[3][3] -
          m[1][2] * m[2][3] * m[3][0] - m[1][3] * m[2][0] * m[3][2]) /
          det;
  r[1][1] =
      (m[0][0] * m[2][2] * m[3][3] + m[0][2] * m[2][3] * m[3][0] +
          m[0][3] * m[2][0] * m[3][2] - m[0][0] * m[2][3] * m[3][2] -
          m[0][2] * m[2][0] * m[3][3] - m[0][3] * m[2][2] * m[3][0]) /
          det;
  r[1][2] =
      (m[0][0] * m[1][3] * m[3][2] + m[0][2] * m[1][0] * m[3][3] +
          m[0][3] * m[1][2] * m[3][0] - m[0][0] * m[1][2] * m[3][3] -
          m[0][2] * m[1][3] * m[3][0] - m[0][3] * m[1][0] * m[3][2]) /
          det;
  r[1][3] =
      (m[0][0] * m[1][2] * m[2][3] + m[0][2] * m[1][3] * m[2][0] +
          m[0][3] * m[1][0] * m[2][2] - m[0][0] * m[1][3] * m[2][2] -
          m[0][2] * m[1][0] * m[2][3] - m[0][3] * m[1][2] * m[2][0]) /
          det;
  r[2][0] =
      (m[1][0] * m[2][1] * m[3][3] + m[1][1] * m[2][3] * m[3][0] +
          m[1][3] * m[2][0] * m[3][1] - m[1][0] * m[2][3] * m[3][1] -
          m[1][1] * m[2][0] * m[3][3] - m[1][3] * m[2][1] * m[3][0]) /
          det;
  r[2][1] =
      (m[0][0] * m[2][3] * m[3][1] + m[0][1] * m[2][0] * m[3][3] +
          m[0][3] * m[2][1] * m[3][0] - m[0][0] * m[2][1] * m[3][3] -
          m[0][1] * m[2][3] * m[3][0] - m[0][3] * m[2][0] * m[3][1]) /
          det;
  r[2][2] =
      (m[0][0] * m[1][1] * m[3][3] + m[0][1] * m[1][3] * m[3][0] +
          m[0][3] * m[1][0] * m[3][1] - m[0][0] * m[1][3] * m[3][1] -
          m[0][1] * m[1][0] * m[3][3] - m[0][3] * m[1][1] * m[3][0]) /
          det;
  r[2][3] =
      (m[0][0] * m[1][3] * m[2][1] + m[0][1] * m[1][0] * m[2][3] +
          m[0][3] * m[1][1] * m[2][0] - m[0][0] * m[1][1] * m[2][3] -
          m[0][1] * m[1][3] * m[2][0] - m[0][3] * m[1][0] * m[2][1]) /
          det;
  r[3][0] =
      (m[1][0] * m[2][2] * m[3][1] + m[1][1] * m[2][0] * m[3][2] +
          m[1][2] * m[2][1] * m[3][0] - m[1][0] * m[2][1] * m[3][2] -
          m[1][1] * m[2][2] * m[3][0] - m[1][2] * m[2][0] * m[3][1]) /
          det;
  r[3][1] =
      (m[0][0] * m[2][1] * m[3][2] + m[0][1] * m[2][2] * m[3][0] +
          m[0][2] * m[2][0] * m[3][1] - m[0][0] * m[2][2] * m[3][1] -
          m[0][1] * m[2][0] * m[3][2] - m[0][2] * m[2][1] * m[3][0]) /
          det;
  r[3][2] =
      (m[0][0] * m[1][2] * m[3][1] + m[0][1] * m[1][0] * m[3][2] +
          m[0][2] * m[1][1] * m[3][0] - m[0][0] * m[1][1] * m[3][2] -
          m[0][1] * m[1][2] * m[3][0] - m[0][2] * m[1][0] * m[3][1]) /
          det;
  r[3][3] =
      (m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] +
          m[0][2] * m[1][0] * m[2][1] - m[0][0] * m[1][2] * m[2][1] -
          m[0][1] * m[1][0] * m[2][2] - m[0][2] * m[1][1] * m[2][0]) /
          det;

  return r;
}
template<typename T>
HERMES_DEVICE_CALLABLE  void decompose(const Matrix4x4<T> &m, Matrix4x4<T> &r, Matrix4x4<T> &s) {
  // extract rotation r from transformation matrix
  T norm;
  int count = 0;
  r = m;
  do {
    // compute next matrix in series
    Matrix4x4<T> Rnext;
    Matrix4x4<T> Rit = inverse(transpose(r));
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        Rnext[i][j] = .5f * (r[i][j] + Rit[i][j]);
    // compute norm difference between R and Rnext
    norm = 0.f;
    for (int i = 0; i < 3; i++) {
      T n = fabsf(r[i][0] - Rnext[i][0]) +
          fabsf(r[i][1] - Rnext[i][1]) + fabsf(r[i][2] - Rnext[i][2]);
      norm = std::max(norm, n);
    }
  } while (++count < 100 && norm > .0001f);
  // compute scale S using rotation and original matrix
  s = inverse(r) * m;
}
//                                                                                                       arithmetic
template<typename T>
HERMES_DEVICE_CALLABLE  Matrix2x2<T> operator*(T f, const Matrix2x2<T> &m) {
  return m * f;
}
//                                                                                                          algebra
template<typename T>
HERMES_DEVICE_CALLABLE Matrix2x2<T> inverse(const Matrix2x2<T> &m) {
  Matrix2x2<T> r;
  T det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  if (det == 0.f)
    return r;
  T k = 1.f / det;
  r[0][0] = m[1][1] * k;
  r[0][1] = -m[0][1] * k;
  r[1][0] = -m[1][0] * k;
  r[1][1] = m[0][0] * k;
  return r;
}

template<typename T>
HERMES_DEVICE_CALLABLE Matrix2x2<T> transpose(const Matrix2x2<T> &m) {
  return Matrix2x2<T>(m[0][0], m[1][0], m[0][1], m[1][1]);
}
//                                                                                                          algebra
template<typename T>
HERMES_DEVICE_CALLABLE  Matrix3x3<T> inverse(const Matrix3x3<T> &m) {
  Matrix3x3<T> r;
  // r.setIdentity();
  T det =
      m[0][0] * m[1][1] * m[2][2] + m[1][0] * m[2][1] * m[0][2] +
          m[2][0] * m[0][1] * m[1][2] - m[0][0] * m[2][1] * m[1][2] -
          m[2][0] * m[1][1] * m[0][2] - m[1][0] * m[0][1] * m[2][2];
  if (std::fabs(det) < 1e-8)
    return r;
  r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
  r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
  r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
  r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
  r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
  r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
  r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
  r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
  r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
  return r;
}
template<typename T>
HERMES_DEVICE_CALLABLE  Matrix3x3<T> transpose(const Matrix3x3<T> &m) {
  return Matrix3x3<T>(m[0][0], m[1][0], m[2][0], m[0][1], m[1][1],
                      m[2][1], m[0][2], m[1][2], m[2][2]);
}
template<typename T>
HERMES_DEVICE_CALLABLE  Matrix3x3<T> star(const Vector3 <T> a) {
  return Matrix3x3<T>(0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0);
}
//                                                                                                       arithmetic
template<typename T>
HERMES_DEVICE_CALLABLE  Matrix3x3<T> operator*(T f, const Matrix3x3<T> &m) {
  return m * f;
}
// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
std::ostream &operator<<(std::ostream &os, const Matrix4x4<T> &m) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      os << m[i][j] << " ";
    os << std::endl;
  }
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Matrix3x3<T> &m) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      os << m[i][j] << " ";
    os << std::endl;
  }
  return os;
}
template<typename T>
std::ostream &operator<<(std::ostream &os, const Matrix2x2<T> &m) {
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++)
      os << m[i][j] << " ";
    os << std::endl;
  }
  return os;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
typedef Matrix4x4<real_t> mat4;
typedef Matrix3x3<real_t> mat3;
typedef Matrix2x2<real_t> mat2;

} // namespace hermes

#endif
