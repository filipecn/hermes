/* Copyright (c) 2017, FilipeCN.
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

/// \file   matrix.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2017-08-18
/// \brief  Math matrix classes

#pragma once

#include <hermes/numeric/math_element.h>

namespace hermes {

/// \brief Inverts a 4x4 matrix
/// \note function extracted from MESA implementation of the GLU library
/// \tparam T
/// \param m
/// \param invOut
/// \return
template <typename T>
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

// *****************************************************************************
//                                                                  MatrixNxM
// *****************************************************************************

/// \brief NxM Matrix representation (N rows, M columns).
template <typename T, u32 N, u32 M>
class MatrixNxM : public MathElement<T, N * M> {
  static_assert(N >= 1 && M >= 1, "MatrixNxM can't have null size.");

public:
  // static

  /// Setup identity matrix.
  HERMES_DEVICE_CALLABLE static inline MatrixNxM I() {
    return MatrixNxM().setIdentity();
  }

  HERMES_DEVICE_CALLABLE static inline MatrixNxM One() {
    MatrixNxM<T, N, M> m;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        m[i][j] = 1;
    return m;
  }

  HERMES_DEVICE_CALLABLE static inline MatrixNxM Zero() { return {}; }

  HERMES_DEVICE_CALLABLE static inline MatrixNxM
  Diag(const MatrixNxM<T, N, 1> &d) {
    static_assert(N == M, "Can't create non-squared matrices from diagonal.");
    MatrixNxM<T, N, M> m;
    for (int i = 0; i < N; ++i)
      m[i][i] = d[i][0];
    return m;
  }

  // constructors

  HERMES_DEVICE_CALLABLE explicit MatrixNxM() {
    std::memset(m_, 0, sizeof(m_));
  }
  /// \param values list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  HERMES_DEVICE_CALLABLE MatrixNxM(std::initializer_list<T> values,
                                   bool columnMajor = false) {
    size_t l = 0, c = 0;
    for (auto v : values) {
      m_[l][c] = v;
      if (columnMajor) {
        l++;
        if (l >= N)
          l = 0, c++;
      } else {
        c++;
        if (c >= M)
          c = 0, l++;
      }
    }
  }
  /// \param mat list of values
  /// \param isColumnMajor [optional | default = false] values configuration
  HERMES_DEVICE_CALLABLE explicit MatrixNxM(const T mat[N * M],
                                            bool columnMajor = false) {
    size_t k = 0;
    if (columnMajor)
      for (int c = 0; c < M; c++)
        for (auto &l : m_)
          l[c] = mat[k++];
    else
      for (auto &l : m_)
        for (int c = 0; c < N; c++)
          l[c] = mat[k++];
  }
  /// \param mat matrix entries in [ROW][COLUMN] form
  HERMES_DEVICE_CALLABLE explicit MatrixNxM(T mat[N][M]) {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < M; j++)
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
  HERMES_DEVICE_CALLABLE MatrixNxM(T m00, T m01, T m02, T m03, T m10, T m11,
                                   T m12, T m13, T m20, T m21, T m22, T m23,
                                   T m30, T m31, T m32, T m33) {
    static_assert(N == 4 && M == 4, "This constructor works only for M4x4");
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
  HERMES_DEVICE_CALLABLE MatrixNxM(T m00, T m01, T m02, T m10, T m11, T m12,
                                   T m20, T m21, T m22) {
    static_assert(N == 3 && M == 3, "This constructor works only for M3x3");
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
  HERMES_DEVICE_CALLABLE MatrixNxM(T m00, T m01, T m10, T m11) {
    static_assert(N == 2 && M == 2, "This constructor works only for M2x2");
    m_[0][0] = m00;
    m_[0][1] = m01;
    m_[1][0] = m10;
    m_[1][1] = m11;
  }

  // operators

  template <u32 O>
  HERMES_DEVICE_CALLABLE MatrixNxM<T, N, M>
  operator*(const MatrixNxM<T, M, O> &B) const {
    MatrixNxM<T, N, O> r;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < O; ++j) {
        r[i][j] = 0;
        for (int k = 0; k < M; ++k)
          r[i][j] += m_[i][k] * B[k][i];
      }
    return r;
  }
  HERMES_DEVICE_CALLABLE MatrixNxM<T, N, 1>
  operator*(const MatrixNxM<T, N, 1> &v) const {
    MatrixNxM<T, N, 1> r;
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        r[i][0] += m_[i][j] * v[j][0];
    return r;
  }
#define ARITHMETIC_OP(OP)                                                      \
  HERMES_DEVICE_CALLABLE MatrixNxM<T, N, M> &operator OP##=(                   \
      const MatrixNxM<T, N, M> &B) {                                           \
    for (int i = 0; i < N; ++i)                                                \
      for (int j = 0; j < M; ++j)                                              \
        m_[i][j] OP## = B[i][j];                                               \
    return *this;                                                              \
  }                                                                            \
  HERMES_DEVICE_CALLABLE MatrixNxM<T, N, M> operator OP(                       \
      const MatrixNxM<T, N, M> &B) const {                                     \
    MatrixNxM<T, N, M> r;                                                      \
    for (int i = 0; i < N; ++i)                                                \
      for (int j = 0; j < M; ++j)                                              \
        r[i][j] = m_[i][j] OP B[i][j];                                         \
    return r;                                                                  \
  }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
#undef ARITHMETIC_OP

#define SCALAR_OP(OP)                                                          \
  HERMES_DEVICE_CALLABLE MatrixNxM<T, N, M> &operator OP##=(const T & s) {     \
    for (int i = 0; i < N; ++i)                                                \
      for (int j = 0; j < M; ++j)                                              \
        m_[i][j] OP## = s;                                                     \
    return *this;                                                              \
  }                                                                            \
  HERMES_DEVICE_CALLABLE MatrixNxM<T, N, M> operator OP(const T & s) const {   \
    MatrixNxM<T, N, M> r;                                                      \
    for (int i = 0; i < N; ++i)                                                \
      for (int j = 0; j < M; ++j)                                              \
        r[i][j] = m_[i][j] OP s;                                               \
  }
  SCALAR_OP(*)
  SCALAR_OP(/)
#undef SCALAR_OP

  template <u32 O, u32 P>
  HERMES_DEVICE_CALLABLE bool operator==(const MatrixNxM<T, O, P> &B) const {
    if (O != N || P != M)
      return false;
    for (int i = 0; i < N; i++)
      for (int j = 0; j < M; j++)
        if (!math::check::is_equal(m_[i][j], B[i][j]))
          return false;
    return true;
  }
  template <u32 O, u32 P>
  HERMES_DEVICE_CALLABLE bool operator!=(const MatrixNxM<T, O, P> &B) const {
    return !((*this) == B);
  }

  HERMES_DEVICE_CALLABLE MatrixNxM<T, N, M> &setIdentity() {
    static_assert(N == M, "Can't set identity for non-square matrices.");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    for (auto &i : m_)
      for (int j = 0; j < M; j++)
        i[j] = 0.f;
#else
    std::memset(m_, 0, sizeof(m_));
#endif
    for (int i = 0; i < M; i++)
      m_[i][i] = 1.f;
    return *this;
  }
  /// \param[out] a Receives matrix elements in row major.
  HERMES_DEVICE_CALLABLE void row_major(T *a) const {
    int k = 0;
    for (auto &i : m_)
      for (int j = 0; j < M; j++)
        a[k++] = i[j];
  }
  /// \param[out] a Receives matrix elements in column major.
  HERMES_DEVICE_CALLABLE void column_major(T *a) const {
    int k = 0;
    for (int i = 0; i < N; i++)
      for (auto &j : m_)
        a[k++] = j[i];
  }
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE bool isIdentity() const {
    static_assert(N == M, "Can't check identity for non-square matrices.");
    for (int i = 0; i < N; i++)
      for (int j = 0; j < M; j++)
        if ((i != j && !math::check::is_equal(m_[i][j], 0.f)) ||
            (i == j && !math::check::is_equal(m_[i][j], 1.f)))
          return false;
    return true;
  }

  HERMES_DEVICE_CALLABLE T determinant() const {
    return m_[0][0] * m_[1][1] - m_[0][1] * m_[1][0];
    return m_[0][0] * m_[1][1] * m_[2][2] + m_[0][1] * m_[1][2] * m_[2][0] +
           m_[0][2] * m_[1][0] * m_[2][1] - m_[2][0] * m_[1][1] * m_[0][2] -
           m_[2][1] * m_[1][2] * m_[0][0] - m_[2][2] * m_[1][0] * m_[0][1];
  }

  HERMES_DEVICE_CALLABLE MatrixNxM<T, N, 1> diagonal() const {
    static_assert(N == M, "Can't create non-squared matrices from diagonal.");
    MatrixNxM<T, N, 1> d;
    for (int i = 0; i < N; ++i)
      d[i][0] = m_[i][i];
    return d;
  }

  HERMES_DEVICE_CALLABLE T *operator[](u32 row_index) { return m_[row_index]; }
  HERMES_DEVICE_CALLABLE const T *operator[](u32 row_index) const {
    return m_[row_index];
  }

private:
  T m_[N][M];
};

template <typename T>
HERMES_DEVICE_CALLABLE MatrixNxM<T, 4, 4>
rowReduce(const MatrixNxM<T, 4, 4> &p, const MatrixNxM<T, 4, 4> &q) {
  MatrixNxM<T, 4, 4> l = p, r = q;
  // TODO implement with gauss jordan elimination
  HERMES_NOT_IMPLEMENTED;
  return r;
}
template <typename T, u32 N, u32 M>
HERMES_DEVICE_CALLABLE MatrixNxM<T, N, M>
transpose(const MatrixNxM<T, M, N> &m) {
  MatrixNxM<T, M, N> t;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < M; ++c)
      t[c][r] = m[r][c];
  return t;
}
template <typename T>
HERMES_DEVICE_CALLABLE MatrixNxM<T, 4, 4>
transpose(const MatrixNxM<T, 4, 4> &m) {
  return MatrixNxM<T, 4, 4>(
      m[0][0], m[1][0], m[2][0], m[3][0], m[0][1], m[1][1], m[2][1], m[3][1],
      m[0][2], m[1][2], m[2][2], m[3][2], m[0][3], m[1][3], m[2][3], m[3][3]);
}
template <typename T>
HERMES_DEVICE_CALLABLE MatrixNxM<T, 4, 4> inverse(const MatrixNxM<T, 4, 4> &m) {
  MatrixNxM<T, 4, 4> r;
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

  r[0][0] = (m[1][1] * m[2][2] * m[3][3] + m[1][2] * m[2][3] * m[3][1] +
             m[1][3] * m[2][1] * m[3][2] - m[1][1] * m[2][3] * m[3][2] -
             m[1][2] * m[2][1] * m[3][3] - m[1][3] * m[2][2] * m[3][1]) /
            det;
  r[0][1] = (m[0][1] * m[2][3] * m[3][2] + m[0][2] * m[2][1] * m[3][3] +
             m[0][3] * m[2][2] * m[3][1] - m[0][1] * m[2][2] * m[3][3] -
             m[0][2] * m[2][3] * m[3][1] - m[0][3] * m[2][1] * m[3][2]) /
            det;
  r[0][2] = (m[0][1] * m[1][2] * m[3][3] + m[0][2] * m[1][3] * m[3][1] +
             m[0][3] * m[1][1] * m[3][2] - m[0][1] * m[1][3] * m[3][2] -
             m[0][2] * m[1][1] * m[3][3] - m[0][3] * m[1][2] * m[3][1]) /
            det;
  r[0][3] = (m[0][1] * m[1][3] * m[2][2] + m[0][2] * m[1][1] * m[2][3] +
             m[0][3] * m[1][2] * m[2][1] - m[0][1] * m[1][2] * m[2][3] -
             m[0][2] * m[1][3] * m[2][1] - m[0][3] * m[1][1] * m[2][2]) /
            det;
  r[1][0] = (m[1][0] * m[2][3] * m[3][2] + m[1][2] * m[2][0] * m[3][3] +
             m[1][3] * m[2][2] * m[3][0] - m[1][0] * m[2][2] * m[3][3] -
             m[1][2] * m[2][3] * m[3][0] - m[1][3] * m[2][0] * m[3][2]) /
            det;
  r[1][1] = (m[0][0] * m[2][2] * m[3][3] + m[0][2] * m[2][3] * m[3][0] +
             m[0][3] * m[2][0] * m[3][2] - m[0][0] * m[2][3] * m[3][2] -
             m[0][2] * m[2][0] * m[3][3] - m[0][3] * m[2][2] * m[3][0]) /
            det;
  r[1][2] = (m[0][0] * m[1][3] * m[3][2] + m[0][2] * m[1][0] * m[3][3] +
             m[0][3] * m[1][2] * m[3][0] - m[0][0] * m[1][2] * m[3][3] -
             m[0][2] * m[1][3] * m[3][0] - m[0][3] * m[1][0] * m[3][2]) /
            det;
  r[1][3] = (m[0][0] * m[1][2] * m[2][3] + m[0][2] * m[1][3] * m[2][0] +
             m[0][3] * m[1][0] * m[2][2] - m[0][0] * m[1][3] * m[2][2] -
             m[0][2] * m[1][0] * m[2][3] - m[0][3] * m[1][2] * m[2][0]) /
            det;
  r[2][0] = (m[1][0] * m[2][1] * m[3][3] + m[1][1] * m[2][3] * m[3][0] +
             m[1][3] * m[2][0] * m[3][1] - m[1][0] * m[2][3] * m[3][1] -
             m[1][1] * m[2][0] * m[3][3] - m[1][3] * m[2][1] * m[3][0]) /
            det;
  r[2][1] = (m[0][0] * m[2][3] * m[3][1] + m[0][1] * m[2][0] * m[3][3] +
             m[0][3] * m[2][1] * m[3][0] - m[0][0] * m[2][1] * m[3][3] -
             m[0][1] * m[2][3] * m[3][0] - m[0][3] * m[2][0] * m[3][1]) /
            det;
  r[2][2] = (m[0][0] * m[1][1] * m[3][3] + m[0][1] * m[1][3] * m[3][0] +
             m[0][3] * m[1][0] * m[3][1] - m[0][0] * m[1][3] * m[3][1] -
             m[0][1] * m[1][0] * m[3][3] - m[0][3] * m[1][1] * m[3][0]) /
            det;
  r[2][3] = (m[0][0] * m[1][3] * m[2][1] + m[0][1] * m[1][0] * m[2][3] +
             m[0][3] * m[1][1] * m[2][0] - m[0][0] * m[1][1] * m[2][3] -
             m[0][1] * m[1][3] * m[2][0] - m[0][3] * m[1][0] * m[2][1]) /
            det;
  r[3][0] = (m[1][0] * m[2][2] * m[3][1] + m[1][1] * m[2][0] * m[3][2] +
             m[1][2] * m[2][1] * m[3][0] - m[1][0] * m[2][1] * m[3][2] -
             m[1][1] * m[2][2] * m[3][0] - m[1][2] * m[2][0] * m[3][1]) /
            det;
  r[3][1] = (m[0][0] * m[2][1] * m[3][2] + m[0][1] * m[2][2] * m[3][0] +
             m[0][2] * m[2][0] * m[3][1] - m[0][0] * m[2][2] * m[3][1] -
             m[0][1] * m[2][0] * m[3][2] - m[0][2] * m[2][1] * m[3][0]) /
            det;
  r[3][2] = (m[0][0] * m[1][2] * m[3][1] + m[0][1] * m[1][0] * m[3][2] +
             m[0][2] * m[1][1] * m[3][0] - m[0][0] * m[1][1] * m[3][2] -
             m[0][1] * m[1][2] * m[3][0] - m[0][2] * m[1][0] * m[3][1]) /
            det;
  r[3][3] = (m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] +
             m[0][2] * m[1][0] * m[2][1] - m[0][0] * m[1][2] * m[2][1] -
             m[0][1] * m[1][0] * m[2][2] - m[0][2] * m[1][1] * m[2][0]) /
            det;

  return r;
}

template <typename T>
HERMES_DEVICE_CALLABLE void decompose(const MatrixNxM<T, 4, 4> &m,
                                      MatrixNxM<T, 4, 4> &r,
                                      MatrixNxM<T, 4, 4> &s) {
  // extract rotation r from transformation matrix
  T norm;
  int count = 0;
  r = m;
  do {
    // compute next matrix in series
    MatrixNxM<T, 4, 4> Rnext;
    MatrixNxM<T, 4, 4> Rit = inverse(transpose(r));
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        Rnext[i][j] = .5f * (r[i][j] + Rit[i][j]);
    // compute norm difference between R and Rnext
    norm = 0.f;
    for (int i = 0; i < 3; i++) {
      T n = fabsf(r[i][0] - Rnext[i][0]) + fabsf(r[i][1] - Rnext[i][1]) +
            fabsf(r[i][2] - Rnext[i][2]);
      norm = std::max(norm, n);
    }
  } while (++count < 100 && norm > .0001f);
  // compute scale S using rotation and original matrix
  s = inverse(r) * m;
}
template <typename T, u32 N, u32 M>
HERMES_DEVICE_CALLABLE MatrixNxM<T, N, M>
operator*(T f, const MatrixNxM<T, N, M> &m) {
  return m * f;
}
template <typename T>
HERMES_DEVICE_CALLABLE MatrixNxM<T, 2, 2> inverse(const MatrixNxM<T, 2, 2> &m) {
  MatrixNxM<T, 2, 2> r;
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

template <typename T>
HERMES_DEVICE_CALLABLE MatrixNxM<T, 2, 2>
transpose(const MatrixNxM<T, 2, 2> &m) {
  return MatrixNxM<T, 2, 2>(m[0][0], m[1][0], m[0][1], m[1][1]);
}
template <typename T>
HERMES_DEVICE_CALLABLE MatrixNxM<T, 3, 3> inverse(const MatrixNxM<T, 3, 3> &m) {
  MatrixNxM<T, 3, 3> r;
  T det = m[0][0] * m[1][1] * m[2][2] + m[1][0] * m[2][1] * m[0][2] +
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
template <typename T>
HERMES_DEVICE_CALLABLE MatrixNxM<T, 3, 3>
transpose(const MatrixNxM<T, 3, 3> &m) {
  return MatrixNxM<T, 3, 3>(m[0][0], m[1][0], m[2][0], m[0][1], m[1][1],
                            m[2][1], m[0][2], m[1][2], m[2][2]);
}

template <typename T>
HERMES_DEVICE_CALLABLE MatrixNxM<T, 3, 3> star(const MatrixNxM<T, 3, 1> a) {
  return MatrixNxM<T, 3, 3>(0, -a[2][0], a[1][0], a[2][0], 0, -a[0][0],
                            -a[1][0], a[0][0], 0);
}

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(
    MatrixNxM<T HERMES_COMMA N HERMES_COMMA M>, typename T, u32 N, u32 M)
for (int row = 0; row < N; ++row) {
  hermes::Str<char> s;
  for (int col = 0; col < M; ++col) {
    s += object[row][col];
    s += " ";
  }
  HERMES_PUSH_DEBUG_LINE("| {}|", s.str());
}
HERMES_TO_STRING_DEBUG_METHOD_END

typedef MatrixNxM<real_t, 4, 4> mat4;
typedef MatrixNxM<real_t, 3, 3> mat3;
typedef MatrixNxM<real_t, 2, 2> mat2;

} // namespace hermes
