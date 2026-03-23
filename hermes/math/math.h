/* Copyright (c) 2019, FilipeCN.
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

/// \file   math.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2019-17-09
/// \brief  math functions

#pragma once

#include <hermes/core/debug.h>
#include <hermes/numeric/numeric.h>

#include <cmath>

namespace hermes::math {

struct constants {
  static constexpr real_t pi = static_cast<real_t>(3.14159265358979323846);
  static constexpr real_t two_pi = static_cast<real_t>(6.28318530718);
  static constexpr real_t inv_pi = static_cast<real_t>(0.31830988618379067154);
  static constexpr real_t inv_two_pi =
      static_cast<real_t>(0.15915494309189533577);
  static constexpr real_t inv_four_pi =
      static_cast<real_t>(0.07957747154594766788);
  static constexpr real_t pi_over_four = static_cast<real_t>(0.78539816339);
  static constexpr real_t pi_over_two = static_cast<real_t>(1.57079632679);
};

/// \tparam T
/// \param a
/// \return
template <typename T> HERMES_CPU_GPU static constexpr T sqr(T a) {
  return a * a;
}
/// Computes square
/// \tparam T
/// \param a
/// \return
template <typename T> HERMES_CPU_GPU static constexpr T cube(T a) {
  return a * a * a;
}
/// Computes sign
/// \tparam T
/// \param a
/// \return
template <typename T> HERMES_CPU_GPU static int sign(T a) {
  return a >= 0 ? 1 : -1;
}
/// Computes square root
/// \tparam T
/// \param a
/// \return
template <typename T> HERMES_CPU_GPU static constexpr T sqrt(T a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
  return sqrtf(a);
#else
  return std::sqrt(a);
#endif
}
/// Computes base 2 log
/// \param x **[in]** value
/// \return base-2 logarithm of **x**
HERMES_CPU_GPU static inline f32 log2(f32 x) {
#ifndef HERMES_DEVICE_ENABLED
  static f32 invLog2 = 1.f / logf(2.f);
#else
  f32 invLog2 = 1.f / logf(2.f);
#endif
  return logf(x) * invLog2;
}
/// Computes square root with clamped input
/// \param x
/// \return
HERMES_CPU_GPU [[maybe_unused]] static f32 safe_sqrt(f32 x) {
  HERMES_CHECK(x >= -1e-3f)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
  return sqrtf(fmaxf(0.f, x));
#else
  return std::sqrt((std::max)(0.f, x));
#endif
}
/// Computes b to the power of n
/// \tparam n
/// \param b
/// \return
template <int n> HERMES_CPU_GPU static inline constexpr real_t pow(real_t b) {
  if constexpr (n < 0)
    return 1 / pow<-n>(b);
  float n2 = pow<n / 2>(b);
  return n2 * n2 * pow<n & 1>(b);
}
/// Computes fast exponential
/// \param x
/// \return
HERMES_CPU_GPU static inline real_t fastExp(real_t x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
  return __expf(x);
#else
  // Compute $x'$ such that $\roman{e}^x = 2^{x'}$
  float xp = x * 1.442695041f;

  // Find integer and fractional components of $x'$
  float fxp = std::floor(xp), f = xp - fxp;
  int i = (int)fxp;

  // Evaluate polynomial approximation of $2^f$
  float twoToF = numeric::evaluatePolynomial(f, 1.f, 0.695556856f, 0.226173572f,
                                             0.0781455737f);

  // Scale $2^f$ by $2^i$ and return final result
  int exponent = numbers::floatExponent(twoToF) + i;
  if (exponent < -126)
    return 0;
  if (exponent > 127)
    return numeric::constants::real_infinity;
  uint32_t bits = numbers::float2bits(twoToF);
  bits &= 0b10000000011111111111111111111111u;
  bits |= (exponent + 127) << 23;
  return numbers::bitsToFloat(bits);
#endif
}
/// Computes v to the power of 1
/// \param v
/// \return
template <> HERMES_CPU_GPU inline constexpr float pow<1>(float v) {
  HERMES_UNUSED_VARIABLE(v);
  return v;
}
/// Computes v to the power of 0
/// \param v
/// \return
template <> HERMES_CPU_GPU inline constexpr float pow<0>(float v) {
  HERMES_UNUSED_VARIABLE(v);
  return 1;
}
/// Converts radians to degrees
/// \param a
/// \return
HERMES_CPU_GPU static constexpr real_t radians2degrees(real_t a) {
  return a * 180.f / constants::pi;
}
/// Converts degrees to radians
/// \param a
/// \return
HERMES_CPU_GPU static constexpr real_t degrees2radians(real_t a) {
  return a * constants::pi / 180.f;
}
/// Computes acos with clamped input
/// \param x
/// \return
HERMES_CPU_GPU static inline f32 safe_acos(f32 x) {
  return std::acos(numbers::clamp<f32>(x, -1, 1));
}
/// Computes asin with clamped input
/// \param x
/// \return
HERMES_CPU_GPU static inline f32 safe_asin(f32 x) {
  return std::asin(numbers::clamp<f32>(x, -1, 1));
}
/// Computes modulus
/// \param a **[in]**
/// \param b **[in]**
/// \return the remainder of a / b
HERMES_CPU_GPU static inline int mod(int a, int b) {
  int n = a / b;
  a -= n * b;
  if (a < 0)
    a += b;
  return a;
}

} // namespace hermes::math
