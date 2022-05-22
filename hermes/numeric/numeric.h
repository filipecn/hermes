/// Copyright (c) 2019, FilipeCN.
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
///\file numeric.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2019-17-09
///
///\brief Number functions
///
///\ingroup numeric
///\addtogroup numeric
/// @{

#ifndef HERMES_GEOMETRY_CUDA_NUMERIC_H
#define HERMES_GEOMETRY_CUDA_NUMERIC_H

#include <hermes/common/debug.h>
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
/// \brief Numeric constants
struct Constants {
  // *******************************************************************************************************************
  //                                                                                                    STATIC FIELDS
  // *******************************************************************************************************************
  static constexpr real_t pi = 3.14159265358979323846;
  static constexpr real_t two_pi = 6.28318530718;
  static constexpr real_t inv_pi = 0.31830988618379067154;
  static constexpr real_t inv_two_pi = 0.15915494309189533577;
  static constexpr real_t inv_four_pi = 0.07957747154594766788;
  static constexpr real_t pi_over_four = 0.78539816339;
  static constexpr real_t pi_over_two = 1.57079632679;
  static constexpr real_t machine_epsilon = std::numeric_limits<real_t>::epsilon() * .5;
  static constexpr real_t real_infinity = std::numeric_limits<real_t>::max();
  static constexpr f64 f64_one_minus_epsilon = 0x1.fffffffffffffp-1;
  static constexpr f32 f32_one_minus_epsilon = 0x1.fffffep-1;
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
  static constexpr real_t one_minus_epsilon = f64_one_minus_epsilon;
#else
  static constexpr real_t one_minus_epsilon = f32_one_minus_epsilon;
#endif
};

// *********************************************************************************************************************
//                                                                                                            Numbers
// *********************************************************************************************************************
/// \brief Number functions
struct Numbers {
  //                                                                                                           limits
  /// \brief Gets lowest representable 64 bit floating point
  /// \tparam T
  /// \return
  HERMES_DEVICE_CALLABLE static constexpr f64 lowest_f64() {
    return -0x1.fffffffffffffp+1023;
  }
  /// \brief Gets lowest representable 64 bit floating point
  /// \tparam T
  /// \return
  HERMES_DEVICE_CALLABLE static constexpr f32 lowest_f32() {
    return -0x1.fffffep+127;
  }
  /// \brief Gets lowest representable floating point
  /// \tparam T
  /// \return
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T lowest() {
    return T(lowest_f32());
  }
  /// \brief Gets greatest representable 32 bit floating point
  /// \return
  HERMES_DEVICE_CALLABLE static constexpr f64 greatest_f32() {
    return 0x1.fffffep+127;
  }
  /// \brief Gets greatest representable 64 bit floating point
  /// \return
  HERMES_DEVICE_CALLABLE static constexpr f64 greatest_f64() {
    return 0x1.fffffffffffffp+1023;
  }
  /// \brief Gets greatest representable floating point
  /// \tparam T
  /// \return
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T greatest() {
    return T(greatest_f32());
  }
  //                                                                                                          queries
  /// \brief Computes minimum between two numbers
  /// \tparam T
  /// \param a
  /// \param b
  /// \return
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T min(const T &a, const T &b) {
    if (a < b)
      return a;
    return b;
  }
  /// \brief Computes maximum between two numbers
  /// \tparam T
  /// \param a
  /// \param b
  /// \return
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T max(const T &a, const T &b) {
    if (a > b)
      return a;
    return b;
  }
  /// \brief Computes minimum value from input
  /// \tparam T
  /// \param l
  /// \return
  template<typename T>
  HERMES_DEVICE_CALLABLE static inline constexpr T min(std::initializer_list<T> l) {
    T m = *l.begin();
    for (auto n : l)
      if (m > n)
        m = n;
    return m;
  }
  /// \brief Computes maximum value from input
  /// \tparam T
  /// \param l
  /// \return
  template<typename T>
  HERMES_DEVICE_CALLABLE static inline constexpr T max(std::initializer_list<T> l) {
    T m = *l.begin();
    for (auto n : l)
      if (m < n)
        m = n;
    return m;
  }
  /// \brief Counts hexadecimal digits
  /// \tparam T
  /// \param n
  /// \return
  template<typename T>
  HERMES_DEVICE_CALLABLE static u8 countHexDigits(T n) {
    u8 count = 0;
    while (n) {
      count++;
      n >>= 4;
    }
    return count;
  }
  //                                                                                                         rounding
  /// \brief Clamps value to closed interval
  /// \param n **[in]** value
  /// \param l **[in]** low
  /// \param u **[in]** high
  /// \return clamp **b** to be in **[l, h]**
  template<typename T>
  HERMES_DEVICE_CALLABLE static T clamp(const T &n, const T &l, const T &u) {
    return fmaxf(l, fminf(n, u));
  }
  //                                                                                                        functions
  /// \brief Swaps values
  /// \tparam T
  /// \param a
  /// \param b
  template<typename T> HERMES_DEVICE_CALLABLE static void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
  }
  /// \tparam T
  /// \param a
  /// \return
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T sqr(T a) { return a * a; }
  /// \brief Computes square
  /// \tparam T
  /// \param a
  /// \return
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T cube(T a) { return a * a * a; }
  /// \brief Computes sign
  /// \tparam T
  /// \param a
  /// \return
  template<typename T> HERMES_DEVICE_CALLABLE static int sign(T a) {
    return a >= 0 ? 1 : -1;
  }
  /// \brief Computes square root
  /// \tparam T
  /// \param a
  /// \return
  template<typename T> HERMES_DEVICE_CALLABLE static constexpr T sqrt(T a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    return sqrtf(a);
#else
    return std::sqrt(a);
#endif
  }
  /// \brief Computes a * b + c
  /// \tparam T
  /// \param a
  /// \param b
  /// \param c
  /// \return
  template<typename T> HERMES_DEVICE_CALLABLE  static inline T FMA(T a, T b, T c) {
    return a * b + c;
  }
  /// \brief Computes difference of products
  /// \tparam Ta
  /// \tparam Tb
  /// \tparam Tc
  /// \tparam Td
  /// \param a
  /// \param b
  /// \param c
  /// \param d
  /// \return
  template<typename Ta, typename Tb, typename Tc, typename Td>
  HERMES_DEVICE_CALLABLE static inline auto differenceOfProducts(Ta a, Tb b, Tc c, Td d) {
    auto cd = c * d;
    auto difference_of_products = FMA(a, b, -cd);
    auto error = FMA(-c, d, cd);
    return difference_of_products + error;
  }
  /// \brief Solves polynomial
  /// \tparam T
  /// \tparam C
  /// \param t
  /// \param c
  /// \return
  template<typename T, typename C>
  HERMES_DEVICE_CALLABLE static inline constexpr T evaluatePolynomial(T t, C c) {
    HERMES_UNUSED_VARIABLE(t)
    return c;
  }
  /// \brief Solves polynomial
  /// \tparam T
  /// \tparam C
  /// \tparam Args
  /// \param t
  /// \param c
  /// \param cs
  /// \return
  template<typename T, typename C, typename... Args>
  HERMES_DEVICE_CALLABLE static inline constexpr T evaluatePolynomial(T t, C c, Args... cs) {
    return FMA(t, evaluatePolynomial(t, cs...), c);
  }
  /// \brief Bisect range based on predicate
  /// \tparam Predicate
  /// \param sz
  /// \param pred
  /// \return
  template<typename Predicate>
  HERMES_DEVICE_CALLABLE static inline size_t findInterval(size_t sz, const Predicate &pred) {
    using ssize_t = std::make_signed_t<size_t>;
    ssize_t size = (ssize_t) sz - 2, first = 1;
    while (size > 0) {
      size_t half = (size_t) size >> 1, middle = first + half;
      bool predResult = pred(middle);
      first = predResult ? middle + 1 : first;
      size = predResult ? size - (half + 1) : half;
    }
    return (size_t) clamp<ssize_t>((ssize_t) first - 1, 0, sz - 2);
  }
  // *******************************************************************************************************************
  //                                                                                                           BINARY
  // *******************************************************************************************************************
  //                                                                                                          integer
  /// \brief Separate bits by 1 bit-space
  /// \param n
  /// \return
  HERMES_DEVICE_CALLABLE static inline u32 separateBitsBy1(u32 n) {
    n = (n ^ (n << 8)) & 0x00ff00ff;
    n = (n ^ (n << 4)) & 0x0f0f0f0f;
    n = (n ^ (n << 2)) & 0x33333333;
    n = (n ^ (n << 1)) & 0x55555555;
    return n;
  }
  /// \brief Separate bits by 2 bit-spaces
  /// \param n
  /// \return
  HERMES_DEVICE_CALLABLE static inline u32 separateBitsBy2(u32 n) {
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n << 8)) & 0x0300f00f;
    n = (n ^ (n << 4)) & 0x030c30c3;
    n = (n ^ (n << 2)) & 0x09249249;
    return n;
  }
  /// \brief Interleaves bits of three integers
  /// \param x
  /// \param y
  /// \param z
  /// \return
  HERMES_DEVICE_CALLABLE static inline u32 interleaveBits(u32 x, u32 y, u32 z) {
    return (separateBitsBy2(z) << 2) + (separateBitsBy2(y) << 1) +
        separateBitsBy2(x);
  }
  /// \brief Interleaves bits of two integers
  /// \param x
  /// \param y
  /// \return
  HERMES_DEVICE_CALLABLE static inline u32 interleaveBits(u32 x, u32 y) {
    return (separateBitsBy1(y) << 1) + separateBitsBy1(x);
  }
  //                                                                                                            float
  /// \brief Extracts exponent from floating-point number
  /// \param v
  /// \return
  HERMES_DEVICE_CALLABLE static inline int floatExponent(f32 v) {
    return (floatToBits(v) >> 23) - 127;
  }
  /// \brief Extracts significand bits
  /// \param v
  /// \return
  HERMES_DEVICE_CALLABLE static inline int floatSignificand(f32 v) {
    return floatToBits(v) & ((1 << 23) - 1);
  }
  /// \brief Extracts sign bit
  /// \param v
  /// \return
  HERMES_DEVICE_CALLABLE static inline uint32_t floatSignBit(f32 v) {
    return floatToBits(v) & 0x80000000;
  }
  /// \brief Interprets a floating-point value into a integer type
  /// \param f f32 value
  /// \return  a 32 bit unsigned integer containing the bits of **f**
  HERMES_DEVICE_CALLABLE static inline uint32_t floatToBits(f32 f) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    return __float_as_uint(f);
#else
    uint32_t ui(0);
    std::memcpy(&ui, &f, sizeof(f32));
    return ui;
#endif
  }
  /// \brief Fills a f32 variable data
  /// \param ui bits
  /// \return a f32 built from bits of **ui**
  HERMES_DEVICE_CALLABLE static inline f32 bitsToFloat(uint32_t ui) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    return __uint_as_float(ui);
#else
    f32 f(0.f);
    std::memcpy(&f, &ui, sizeof(uint32_t));
    return f;
#endif
  }
  /// \brief Interprets a f64-point value into a integer type
  /// \param d f64 value
  /// \return  a 64 bit unsigned integer containing the bits of **f**
  HERMES_DEVICE_CALLABLE static inline uint64_t floatToBits(f64 d) {
    uint64_t ui(0);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    memcpy(&ui, &d, sizeof(f64));
#else
    std::memcpy(&ui, &d, sizeof(f64));
#endif
    return ui;
  }
  /// \brief Fills a f64 variable data
  /// \param ui bits
  /// \return a f64 built from bits of **ui**
  HERMES_DEVICE_CALLABLE static inline f64 bitsToDouble(uint64_t ui) {
    f64 d(0.f);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    memcpy(&d, &ui, sizeof(uint64_t));
#else
    std::memcpy(&d, &ui, sizeof(uint64_t));
#endif
    return d;
  }
  /// \brief Computes the next greater representable floating-point value
  /// \param v floating point value
  /// \return the next greater floating point value
  HERMES_DEVICE_CALLABLE static inline f32 nextFloatUp(f32 v) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    if (isinf(v) && v > 0.)
#else
    if (std::isinf(v) && v > 0.f)
#endif
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
  /// \brief Computes the next smaller representable floating-point value
  /// \param v floating point value
  /// \return the next smaller floating point value
  HERMES_DEVICE_CALLABLE static inline f32 nextFloatDown(f32 v) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    if (isinf(v) && v > 0.)
#else
    if (std::isinf(v) && v > 0.f)
#endif
      return v;
    if (v == 0.f)
      v = -0.f;
    uint32_t ui = floatToBits(v);
    if (v > 0)
      --ui;
    else
      ++ui;
    return bitsToFloat(ui);
  }
  /// \brief Computes the next greater representable floating-point value
  /// \param v floating point value
  /// \return the next greater floating point value
  HERMES_DEVICE_CALLABLE static inline f64 nextDoubleUp(f64 v) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    if (isinf(v) && v > 0.)
#else
    if (std::isinf(v) && v > 0.)
#endif
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
  /// \brief Computes the next smaller representable floating-point value
  /// \param v floating point value
  /// \return the next smaller floating point value
  HERMES_DEVICE_CALLABLE static f64 nextDoubleDown(f64 v) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    if (isinf(v) && v > 0.)
#else
    if (std::isinf(v) && v > 0.)
#endif
      return v;
    if (v == 0.)
      v = -0.f;
    uint64_t ui = floatToBits(v);
    if (v > 0)
      --ui;
    else
      ++ui;
    return bitsToDouble(ui);
  }
  // *******************************************************************************************************************
  //                                                                                                          INTEGER
  // *******************************************************************************************************************
  //                                                                                                          limits
  /// \brief Gets minimum representable 32 bit signed integer
  /// \return
  HERMES_DEVICE_CALLABLE static constexpr int lowest_int() { return -2147483647; }
  /// \brief Gets maximum representable 32 bit signed integer
  /// \return
  HERMES_DEVICE_CALLABLE static constexpr int greatest_int() { return 2147483647; }
  /// \brief Checks if integer is power of 2
  /// \param v **[in]** value
  /// \return **true** if **v** is power of 2
  HERMES_DEVICE_CALLABLE static constexpr inline bool isPowerOf2(int v) { return (v & (v - 1)) == 0; }
  /// \brief Computes modulus
  /// \param a **[in]**
  /// \param b **[in]**
  /// \return the remainder of a / b
  HERMES_DEVICE_CALLABLE static inline int mod(int a, int b) {
    int n = a / b;
    a -= n * b;
    if (a < 0)
      a += b;
    return a;
  }
  //                                                                                                         rounding
  /// \brief rounds up
  /// \param f **[in]**
  /// \return ceil of **f**
  HERMES_DEVICE_CALLABLE static inline int ceil2Int(float f) { return static_cast<int>(f + 0.5f); }
  /// \brief rounds down
  /// \param f **[in]**
  /// \return floor of **f**
  HERMES_DEVICE_CALLABLE static inline int floor2Int(float f) { return static_cast<int>(f); }
  /// \brief rounds to closest integer
  /// \param f **[in]**
  /// \return next integer greater or equal to **f**
  HERMES_DEVICE_CALLABLE static inline int round2Int(float f) { return f + .5f; }
  //                                                                                                         rounding
  /// \brief Computes number of digits
  /// \param t
  /// \param base
  /// \return
  HERMES_DEVICE_CALLABLE static inline u8 countDigits(u64 t, u8 base = 10) {
    u8 count{0};
    while (t) {
      count++;
      t /= base;
    }
    return count;
  }

  // *******************************************************************************************************************
  //                                                                                                   FLOATING POINT
  // *******************************************************************************************************************
  //                                                                                                         rounding
  /// \brief Extract decimal fraction from x
  /// \param x
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t fract(real_t x) {
    if (x >= 0.)
      return x - floor(x);
    else
      return x - ceil(x);
  }
  /// \brief Multiplies and rounds down to the next smaller float value
  /// \param a
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t mulRoundDown(real_t a, real_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __dmul_rd(a, b);
#else
    return __fmul_rd(a, b);
#endif
#else
    return nextFloatDown(a * b);
#endif
  }
  /// \brief Multiplies and rounds up to the next float value
  /// \param a
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t mulRoundUp(real_t a, real_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __dmul_ru(a, b);
#else
    return __fmul_ru(a, b);
#endif
#else
    return nextFloatUp(a * b);
#endif
  }
  /// \brief Divides and rounds down to the next smaller float value
  /// \param a
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t divRoundDown(real_t a, real_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __ddiv_rd(a, b);
#else
    return __fdiv_rd(a, b);
#endif
#else
    return nextFloatDown(a / b);
#endif
  }
  /// \brief Divides and rounds up to the next float value
  /// \param a
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t divRoundUp(real_t a, real_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __ddiv_ru(a, b);
#else
    return __fdiv_ru(a, b);
#endif
#else
    return nextFloatUp(a / b);
#endif
  }
  /// \brief Adds and rounds down to the next smaller float value
  /// \param a
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t addRoundDown(real_t a, real_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __dadd_rd(a, b);
#else
    return __fadd_rd(a, b);
#endif
#else
    return nextFloatDown(a + b);
#endif
  }
  /// \brief Adds and rounds up to the next float value
  /// \param a
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t addRoundUp(real_t a, real_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __dadd_ru(a, b);
#else
    return __fadd_ru(a, b);
#endif
#else
    return nextFloatUp(a + b);
#endif
  }
  /// \brief Subtracts and rounds down to the next smaller float value
  /// \param a
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t subRoundDown(real_t a, real_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __dsub_rd(a, b);
#else
    return __fsub_rd(a, b);
#endif
#else
    return nextFloatDown(a - b);
#endif
  }
  /// \brief Subtracts and rounds up to the next float value
  /// \param a
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t subRoundUp(real_t a, real_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __dsub_ru(a, b);
#else
    return __fsub_ru(a, b);
#endif
#else
    return nextFloatUp(a - b);
#endif
  }
  /// \brief Computes square root rounded down to the next smaller float value
  /// \param a
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t sqrtRoundDown(real_t a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __dsqrt_rd(a);
#else
    return __fsqrt_rd(a);
#endif
#else
    return max<real_t>(0, nextFloatDown(std::sqrt(a)));
#endif
  }
  /// \brief Computes square root rounded up to the next float value
  /// \param a
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t sqrtRoundUp(real_t a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
#ifdef HERMES_USE_DOUBLE_AS_DEFAULT
    return __dsqrt_ru(a);
#else
    return __fsqrt_ru(a);
#endif
#else
    return nextFloatUp(std::sqrt(a));
#endif
  }
  //                                                                                                        functions
  /// \brief Computes base 2 log
  /// \param x **[in]** value
  /// \return base-2 logarithm of **x**
  HERMES_DEVICE_CALLABLE static inline f32 log2(f32 x) {
#ifndef HERMES_DEVICE_ENABLED
    static f32 invLog2 = 1.f / logf(2.f);
#else
    f32 invLog2 = 1.f / logf(2.f);
#endif
    return logf(x) * invLog2;
  }
  /// \brief Computes square root with clamped input
  /// \param x
  /// \return
  HERMES_DEVICE_CALLABLE static f32 safe_sqrt(f32 x) {
    HERMES_CHECK_EXP(x >= -1e-3f)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    return sqrtf(fmaxf(0.f, x));
#else
    return std::sqrt(std::max(0.f, x));
#endif
  }
  /// \brief Computes b to the power of n
  /// \tparam n
  /// \param b
  /// \return
  template<int n>
  HERMES_DEVICE_CALLABLE static inline constexpr real_t pow(real_t b) {
    if constexpr (n < 0)
      return 1 / pow<-n>(b);
    float n2 = pow<n / 2>(b);
    return n2 * n2 * pow<n & 1>(b);
  }
  /// \brief Computes fast exponential
  /// \param x
  /// \return
  HERMES_DEVICE_CALLABLE static inline real_t fastExp(real_t x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    return __expf(x);
#else
    // Compute $x'$ such that $\roman{e}^x = 2^{x'}$
    float xp = x * 1.442695041f;

    // Find integer and fractional components of $x'$
    float fxp = std::floor(xp), f = xp - fxp;
    int i = (int) fxp;

    // Evaluate polynomial approximation of $2^f$
    float twoToF = evaluatePolynomial(f, 1.f, 0.695556856f, 0.226173572f, 0.0781455737f);

    // Scale $2^f$ by $2^i$ and return final result
    int exponent = floatExponent(twoToF) + i;
    if (exponent < -126)
      return 0;
    if (exponent > 127)
      return Constants::real_infinity;
    uint32_t bits = floatToBits(twoToF);
    bits &= 0b10000000011111111111111111111111u;
    bits |= (exponent + 127) << 23;
    return bitsToFloat(bits);
#endif
  }
  // *******************************************************************************************************************
  //                                                                                                            ERROR
  // *******************************************************************************************************************
  /// \brief Computes conservative bounds in error
  /// \param n
  /// \return
  HERMES_DEVICE_CALLABLE static constexpr real_t gamma(i32 n) {
    return (n * Constants::machine_epsilon) / (1 - n * Constants::machine_epsilon);
  }
};
/// \brief Gets lowest representable 64 bit floating point
/// \return
template<> HERMES_DEVICE_CALLABLE inline constexpr f64 Numbers::lowest() {
  return Numbers::lowest_f64();
}
/// \brief Gets lowest representable 32 bit floating point
/// \return
template<> HERMES_DEVICE_CALLABLE inline constexpr f32 Numbers::lowest() {
  return Numbers::lowest_f32();
}
/// \brief Computes v to the power of 1
/// \param v
/// \return
template<>
HERMES_DEVICE_CALLABLE inline constexpr float Numbers::pow<1>(float v) {
  HERMES_UNUSED_VARIABLE(v);
  return v;
}
/// \brief Computes v to the power of 0
/// \param v
/// \return
template<>
HERMES_DEVICE_CALLABLE inline constexpr float Numbers::pow<0>(float v) {
  HERMES_UNUSED_VARIABLE(v);
  return 1;
}
/// \brief Gets greatest 32 bit floating point number
/// \return
template<> HERMES_DEVICE_CALLABLE inline constexpr f32 Numbers::greatest() {
  return greatest_f32();
}
/// \brief Gets greatest 64 bit floating point number
/// \return
template<> HERMES_DEVICE_CALLABLE inline constexpr f64 Numbers::greatest() {
  return greatest_f64();
}
// *********************************************************************************************************************
//                                                                                                       Trigonometry
// *********************************************************************************************************************
/// \brief Trigonometric functions
struct Trigonometry {
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  /// \brief Converts radians to degrees
  /// \param a
  /// \return
  HERMES_DEVICE_CALLABLE static constexpr real_t radians2degrees(real_t a) {
    return a * 180.f / Constants::pi;
  }
  /// \brief Converts degrees to radians
  /// \param a
  /// \return
  HERMES_DEVICE_CALLABLE static constexpr real_t degrees2radians(real_t a) {
    return a * Constants::pi / 180.f;
  }
  /// \brief Computes acos with clamped input
  /// \param x
  /// \return
  HERMES_DEVICE_CALLABLE static inline f32 safe_acos(f32 x) {
    return std::acos(Numbers::clamp<f32>(x, -1, 1));
  }
  /// \brief Computes asin with clamped input
  /// \param x
  /// \return
  HERMES_DEVICE_CALLABLE static inline f32 safe_asin(f32 x) {
    return std::asin(Numbers::clamp<f32>(x, -1, 1));
  }
};

// *********************************************************************************************************************
//                                                                                                              Check
// *********************************************************************************************************************
/// \brief Number checks
struct Check {
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  ///\brief Checks if number is 0
  ///\tparam T
  ///\param a **[in]**
  ///\return constexpr bool
  template<typename T>
  HERMES_DEVICE_CALLABLE static constexpr bool is_zero(T a) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    return fabs(a) < 1e-8;
#else
    return std::fabs(a) < 1e-8;
#endif
  }
  ///\brief Checks if two numbers are at most 1e-8 apart
  ///\tparam T
  ///\param a **[in]**
  ///\param b **[in]**
  ///\return constexpr bool
  template<typename T>
  HERMES_DEVICE_CALLABLE static constexpr bool is_equal(T a, T b) {
    return fabs(a - b) < 1e-8;
  }
  ///\brief Checks if two numbers are at most to a threshold apart
  ///\tparam T
  ///\param a **[in]**
  ///\param b **[in]**
  ///\param e **[in]**
  ///\return constexpr bool
  template<typename T>
  HERMES_DEVICE_CALLABLE static constexpr bool is_equal(T a, T b, T e) {
    return fabs(a - b) < e;
  }
  ///\brief Checks if a number is in a open interval
  ///\tparam T
  ///\param x **[in]**
  ///\param a **[in]**
  ///\param b **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_between(T x, T a, T b) {
    return x > a && x < b;
  }
  ///\brief Checks if a number is in a closed interval
  ///\tparam T
  ///\param x **[in]**
  ///\param a **[in]**
  ///\param b **[in]**
  ///\return constexpr bool
  template<typename T> static constexpr bool is_between_closed(T x, T a, T b) {
    return x >= a && x <= b;
  }
  /// \brief Checks if number representation is `nan`
  /// \tparam T
  /// \param v
  /// \return
  template<typename T>
  HERMES_DEVICE_CALLABLE static inline typename std::enable_if_t<std::is_floating_point<T>::value, bool>
  is_nan(T v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    return isnan(v);
#else
    return std::isnan(v);
#endif
  }

};

namespace numeric {

/// \brief Solves a linear system of 2 equations
/// \tparam T
/// \param A
/// \param B
/// \param x0
/// \param x1
/// \return
template<typename T>
HERMES_DEVICE_CALLABLE bool soveLinearSystem(const T A[2][2], const T B[2], T *x0, T *x1) {
  T det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  if (abs(det) < 1e-10f)
    return false;
  *x0 = (A[1][1] * B[0] - A[0][1] * B[1]) / det;
  *x1 = (A[0][0] * B[1] - A[1][0] * B[0]) / det;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
  if(isnan(*x0) || isnan(*x1))
    return false;
#else
  if (std::isnan(*x0) || std::isnan(*x1))
    return false;
#endif
  return true;
}

}
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

/// @}
