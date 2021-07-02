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

#ifndef HERMES_GEOMETRY_CUDA_NUMERIC_H
#define HERMES_GEOMETRY_CUDA_NUMERIC_H

#include <hermes/common/defs.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include <cstring>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                          Constants
// *********************************************************************************************************************
struct Constants {
  // *******************************************************************************************************************
  //                                                                                                    STATIC FIELDS
  // *******************************************************************************************************************
  static constexpr real_t pi = 3.14159265358979323846;
  static constexpr real_t two_pi = 6.28318530718;
  static constexpr real_t inv_pi = 0.31830988618379067154;
  static constexpr real_t inv_two_pi = 0.15915494309189533577;
  static constexpr real_t inv_four_pi = 0.07957747154594766788;
  static constexpr real_t machine_epsilon = std::numeric_limits<real_t>::epsilon() * .5;
  static constexpr real_t real_infinity = std::numeric_limits<real_t>::max();
};

// *********************************************************************************************************************
//                                                                                                       Trigonometry
// *********************************************************************************************************************
struct Trigonometry {
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE static constexpr real_t radians2degrees(real_t a) {
    return a * 180.f / Constants::pi;
  }
  HERMES_DEVICE_CALLABLE static constexpr real_t degrees2radians(real_t a) {
    return a * Constants::pi / 180.f;
  }
};

// *********************************************************************************************************************
//                                                                                                            Numbers
// *********************************************************************************************************************
struct Numbers {
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  /// Compute conservative bounds in error
  /// \param n
  /// \return
  static constexpr real_t gamma(i32 n) {
    return (n * Constants::machine_epsilon) / (1 - n * Constants::machine_epsilon);
  }
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T lowest() {
    return 0xfff0000000000000;
  }
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T greatest() {
    return 0x7ff0000000000000;
  }
  HERMES_DEVICE_CALLABLE static constexpr int lowest_int() { return -2147483647; }
  HERMES_DEVICE_CALLABLE static constexpr int greatest_int() { return 2147483647; }
  template<typename T> HERMES_DEVICE_CALLABLE T min(const T &a, const T &b) {
    if (a < b)
      return a;
    return b;
  }

  template<typename T> HERMES_DEVICE_CALLABLE T max(const T &a, const T &b) {
    if (a > b)
      return a;
    return b;
  }

  template<typename T> HERMES_DEVICE_CALLABLE void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
  }
  /// \brief round
  /// \param f **[in]**
  /// \return ceil of **f**
  static inline int ceil2Int(float f) { return static_cast<int>(f + 0.5f); }
  /// \brief round
  /// \param f **[in]**
  /// \return floor of **f**
  static inline int floor2Int(float f) { return static_cast<int>(f); }
  /// \brief round
  /// \param f **[in]**
  /// \return next integer greater or equal to **f**
  static inline int round2Int(float f) { return f + .5f; }
  /// \brief modulus
  /// \param a **[in]**
  /// \param b **[in]**
  /// \return the remainder of a / b
  static inline float mod(int a, int b) {
    int n = a / b;
    a -= n * b;
    if (a < 0)
      a += b;
    return a;
  }
  /// \param n **[in]** value
  /// \param l **[in]** low
  /// \param u **[in]** high
  /// \return clamp **b** to be in **[l, h]**
  template<typename T>
  HERMES_DEVICE_CALLABLE static T clamp(const T &n, const T &l, const T &u) {
    return fmaxf(l, fminf(n, u));
  }
  /// \param x **[in]** value
  /// \return base-2 logarithm of **x**
  HERMES_DEVICE_CALLABLE static inline f32 log2(f32 x) {
#ifndef HERMES_DEVICE_CODE
    static f32 invLog2 = 1.f / logf(2.f);
#else
    f32 invLog2 = 1.f / logf(2.f);
#endif
    return logf(x) * invLog2;
  }
  /// \param v **[in]** value
  /// \return **true** if **v** is power of 2
  HERMES_DEVICE_CALLABLE static inline bool isPowerOf2(int v) { return (v & (v - 1)) == 0; }
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T sqr(T a) { return a * a; }
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T cube(T a) { return a * a * a; }
  template<typename T> HERMES_DEVICE_CALLABLE static int sign(T a) {
    return a >= 0 ? 1 : -1;
  }
  HERMES_DEVICE_CALLABLE static inline u8 countDigits(u64 t, u8 base = 10) {
    u8 count{0};
    while (t) {
      count++;
      t /= base;
    }
    return count;
  }
  HERMES_DEVICE_CALLABLE static inline u32 separateBitsBy1(u32 n) {
    n = (n ^ (n << 8)) & 0x00ff00ff;
    n = (n ^ (n << 4)) & 0x0f0f0f0f;
    n = (n ^ (n << 2)) & 0x33333333;
    n = (n ^ (n << 1)) & 0x55555555;
    return n;
  }
  HERMES_DEVICE_CALLABLE static inline u32 separateBitsBy2(u32 n) {
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n << 8)) & 0x0300f00f;
    n = (n ^ (n << 4)) & 0x030c30c3;
    n = (n ^ (n << 2)) & 0x09249249;
    return n;
  }
  HERMES_DEVICE_CALLABLE static inline u32 interleaveBits(u32 x, u32 y, u32 z) {
    return (separateBitsBy2(z) << 2) + (separateBitsBy2(y) << 1) +
        separateBitsBy2(x);
  }
  HERMES_DEVICE_CALLABLE static inline u32 interleaveBits(u32 x, u32 y) {
    return (separateBitsBy1(y) << 1) + separateBitsBy1(x);
  }
  //                                                                                                   floating point
  /// Interprets a f32ing-point value into a integer type
  /// \param f f32 value
  /// \return  a 32 bit unsigned integer containing the bits of **f**
  HERMES_DEVICE_CALLABLE static inline uint32_t floatToBits(f32 f) {
    uint32_t ui(0);
    std::memcpy(&ui, &f, sizeof(f32));
    return ui;
  }
  /// Fills a f32 variable data
  /// \param ui bits
  /// \return a f32 built from bits of **ui**
  HERMES_DEVICE_CALLABLE static inline f32 bitsToFloat(uint32_t ui) {
    f32 f(0.f);
    std::memcpy(&f, &ui, sizeof(uint32_t));
    return f;
  }
  /// Interprets a f64-point value into a integer type
  /// \param d f64 value
  /// \return  a 64 bit unsigned integer containing the bits of **f**
  HERMES_DEVICE_CALLABLE static inline uint64_t floatToBits(f64 d) {
    uint64_t ui(0);
    std::memcpy(&ui, &d, sizeof(f64));
    return ui;
  }
  /// Fills a f64 variable data
  /// \param ui bits
  /// \return a f64 built from bits of **ui**
  HERMES_DEVICE_CALLABLE static inline f64 bitsToDouble(uint64_t ui) {
    f64 d(0.f);
    std::memcpy(&d, &ui, sizeof(uint64_t));
    return d;
  }
  /// Computes the next greater representable f32ing-point value
  /// \param v f32ing point value
  /// \return the next greater f32ing point value
  HERMES_DEVICE_CALLABLE static inline f32 nextFloatUp(f32 v) {
    if (std::isinf(v) && v > 0.)
      return v;
    if (v == -0.f)
      v = 0.f;
    uint32_t ui = floatToBits(v);
    if (v >= 0)
      ++ui;
    else
      --ui;
    return bitsToFloat(ui);
  }
  /// Computes the next smaller representable f32ing-point value
  /// \param v f32ing point value
  /// \return the next smaller f32ing point value
  HERMES_DEVICE_CALLABLE static inline f32 nextFloatDown(f32 v) {
    if (std::isinf(v) && v > 0.)
      return v;
    if (v == -0.f)
      v = 0.f;
    uint32_t ui = floatToBits(v);
    if (v >= 0)
      --ui;
    else
      ++ui;
    return bitsToFloat(ui);
  }
  /// Computes the next greater representable f32ing-point value
  /// \param v f32ing point value
  /// \return the next greater f32ing point value
  HERMES_DEVICE_CALLABLE static inline f64 nextDoubleUp(f64 v) {
    if (std::isinf(v) && v > 0.)
      return v;
    if (v == -0.f)
      v = 0.f;
    uint64_t ui = floatToBits(v);
    if (v >= 0)
      ++ui;
    else
      --ui;
    return bitsToDouble(ui);
  }
  /// Computes the next smaller representable f32ing-point value
  /// \param v f32ing point value
  /// \return the next smaller f32ing point value
  HERMES_DEVICE_CALLABLE static f64 nextDoubleDown(f64 v) {
    if (std::isinf(v) && v > 0.)
      return v;
    if (v == -0.f)
      v = 0.f;
    uint64_t ui = floatToBits(v);
    if (v >= 0)
      --ui;
    else
      ++ui;
    return bitsToDouble(ui);
  }
};

// *********************************************************************************************************************
//                                                                                                              Check
// *********************************************************************************************************************
struct Check {
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  ///\brief
  ///
  ///\tparam T
  ///\param a **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_zero(T a) {
    return std::fabs(a) < 1e-8;
  }
  ///\brief
  ///
  ///\tparam T
  ///\param a **[in]**
  ///\param b **[in]**
  ///\return constexpr bool
  template<typename T>
  HERMES_DEVICE_CALLABLE static constexpr bool is_equal(T a, T b) {
    return fabs(a - b) < 1e-8;
  }
  ///\brief
  ///
  ///\tparam T
  ///\param a **[in]**
  ///\param b **[in]**
  ///\param e **[in]**
  ///\return constexpr bool
  template<typename T>
  HERMES_DEVICE_CALLABLE static constexpr bool is_equal(T a, T b, T e) {
    return fabs(a - b) < e;
  }
  ///\brief
  ///
  ///\tparam T
  ///\param x **[in]**
  ///\param a **[in]**
  ///\param b **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_between(T x, T a, T b) {
    return x > a && x < b;
  }
  ///\brief
  ///
  ///\tparam T
  ///\param x **[in]**
  ///\param a **[in]**
  ///\param b **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_between_closed(T x, T a, T b) {
    return x >= a && x <= b;
  }
};

//
//template<typename T>
//__device__ __host__ unsigned int mortonCode(const Point3 <T> &v) {
//  return interleaveBits(v[0], v[1], v[2]);
//}
//
//template<typename T>
//__device__ __host__ unsigned int mortonCode(const Point2 <T> &v) {
//  return interleaveBits(v[0], v[1]);
//}

} // namespace hermes

#endif
