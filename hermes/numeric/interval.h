/* Copyright (c) 2021, FilipeCN.
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

/// \file   interval.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2021-07-01
///  Numeric interval

#pragma once

#include <hermes/math/math.h>

namespace hermes {

// *****************************************************************************
//                                                                   Interval
// *****************************************************************************

/// Represents a numeric interval that supports interval arithmetic
template <typename T> class Interval {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                    std::is_same<T, float>::value ||
                    std::is_same<T, double>::value ||
                    std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Interval must hold an numeric type!");

public:
  /// Constructs interval from center and radius
  /// \param c center
  /// \param r radius
  /// \return
  HERMES_CPU_GPU static Interval withRadius(real_t c, real_t r) {
    if (r == 0)
      return {c, c};
    return {numbers::subRoundDown(c, r), numbers::addRoundUp(c, r)};
  }

  /// Default constructor
  HERMES_CPU_GPU Interval() : low(0), high(0) {}
  /// Constructs from center value
  /// \param v
  HERMES_CPU_GPU Interval(T v) : low(v), high(v) {}
  /// Construct from interval values
  /// \param l
  /// \param h
  HERMES_CPU_GPU Interval(T l, T h) : low(l), high(h) {}

  /// Gets interval center value
  /// \return
  HERMES_CPU_GPU explicit operator T() const { return center(); }
  /// Negates interval
  /// \return
  HERMES_CPU_GPU Interval operator-() const { return {-high, -low}; }

  /// Uses interval arithmetic addition
  /// \param i
  /// \return
  HERMES_CPU_GPU Interval operator+(const Interval &i) const {
    return Interval(numbers::addRoundDown(low, i.low),
                    numbers::addRoundUp(high, i.high));
  }
  /// Uses interval arithmetic subtraction
  /// \param i
  /// \return
  HERMES_CPU_GPU Interval operator-(const Interval &i) const {
    return Interval(numbers::subRoundDown(low, i.low),
                    numbers::subRoundUp(high, i.high));
  }
  /// Uses interval arithmetic multiplication
  /// \param i
  /// \return
  HERMES_CPU_GPU Interval operator*(const Interval &i) const {
    T lp[4] = {numbers::mulRoundDown(low, i.low),
               numbers::mulRoundDown(high, i.low),
               numbers::mulRoundDown(low, i.high),
               numbers::mulRoundDown(high, i.high)};
    T hp[4] = {
        numbers::mulRoundUp(low, i.low), numbers::mulRoundUp(high, i.low),
        numbers::mulRoundUp(low, i.high), numbers::mulRoundUp(high, i.high)};
    return {numbers::cmp::min({lp[0], lp[1], lp[2], lp[3]}),
            numbers::cmp::max({hp[0], hp[1], hp[2], hp[3]})};
  }
  /// Uses interval arithmetic division
  /// \param i
  /// \return
  HERMES_CPU_GPU Interval operator/(const Interval &i) const {
    Interval r = i;
    if (r.low < 0 && r.high > 0)
      return {numeric::limits::lowest<T>(), numeric::limits::greatest<T>()};
    T lq[4] = {numbers::divRoundDown(low, i.low),
               numbers::divRoundDown(high, i.low),
               numbers::divRoundDown(low, i.high),
               numbers::divRoundDown(high, i.high)};
    T hq[4] = {
        numbers::divRoundUp(low, i.low), numbers::divRoundUp(high, i.low),
        numbers::divRoundUp(low, i.high), numbers::divRoundUp(high, i.high)};
    return {numbers::cmp::min({lq[0], lq[1], lq[2], lq[3]}),
            numbers::cmp::max({hq[0], hq[1], hq[2], hq[3]})};
  }
  //                                                                                                          boolean
  HERMES_CPU_GPU bool operator==(const Interval<T> &b) const {
    return numbers::cmp::is_equal(low, b.low) &&
           numbers::cmp::is_equal(high, b.high);
  }

  /// Checks if this interval contains v
  /// \param v
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU bool contains(T v) const {
    return v >= low && v <= high;
  }
  /// Gets interval center value
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU T center() const { return (low + high) / 2; }
  /// Gets interval radius
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU T radius() const { return (high - low) / 2; }
  /// Gets interval diameter
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU T width() const { return high - low; }
  /// Checks if interval contains a single value
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU bool isExact() const {
    return high - low == 0;
  }
  /// Computes arithmetic interval square
  /// \return
  HERMES_NODISCARD HERMES_CPU_GPU Interval sqr() const {
    real_t alow = std::abs(low), ahigh = std::abs(high);
    if (alow > ahigh)
      std::swap(alow, ahigh);
    if (contains(0))
      return Interval(0, numbers::mulRoundUp(ahigh, ahigh));
    return Interval(numbers::mulRoundDown(alow, alow),
                    numbers::mulRoundUp(ahigh, ahigh));
  }
  /// Computes arithmetic interval square root
  HERMES_NODISCARD HERMES_CPU_GPU Interval sqrt() const {
    return {numbers::sqrtRoundDown(low), numbers::sqrtRoundUp(high)};
  }

  T low{0};  //!< lowest interval value
  T high{0}; //!< greatest interval value
};

#define ARITHMETIC_OP(OP)                                                      \
  template <typename T>                                                        \
  HERMES_CPU_GPU Interval<T> operator OP(T f, const Interval<T> &i) {          \
    return Interval<T>(f) OP i;                                                \
  }                                                                            \
  template <typename T>                                                        \
  HERMES_CPU_GPU Interval<T> operator OP(const Interval<T> &i, T f) {          \
    return i OP Interval<T>(f);                                                \
  }
ARITHMETIC_OP(+)
ARITHMETIC_OP(-)
ARITHMETIC_OP(*)
ARITHMETIC_OP(/)
#undef ARITHMETIC_OP

template <typename T> struct DebugTraits<Interval<T>> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static DebugMessage message(const Interval<T> &data) {
    return DebugMessage("I[{}, {}]", data.low, data.high);
  }
};

} // namespace hermes
