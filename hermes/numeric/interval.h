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

#include <hermes/numeric/math.h>

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
  HERMES_DEVICE_CALLABLE static Interval withRadius(real_t c, real_t r) {
    if (r == 0)
      return {c, c};
    return {math::numbers::subRoundDown(c, r), math::numbers::addRoundUp(c, r)};
  }

  /// Default constructor
  HERMES_DEVICE_CALLABLE Interval() : low(0), high(0) {}
  /// Constructs from center value
  /// \param v
  HERMES_DEVICE_CALLABLE Interval(T v) : low(v), high(v) {}
  /// Construct from interval values
  /// \param l
  /// \param h
  HERMES_DEVICE_CALLABLE Interval(T l, T h) : low(l), high(h) {}

  /// Gets interval center value
  /// \return
  HERMES_DEVICE_CALLABLE explicit operator T() const { return center(); }
  /// Negates interval
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator-() const { return {-high, -low}; }

  /// Uses interval arithmetic addition
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator+(const Interval &i) const {
    return Interval(math::numbers::addRoundDown(low, i.low),
                    math::numbers::addRoundUp(high, i.high));
  }
  /// Uses interval arithmetic subtraction
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator-(const Interval &i) const {
    return Interval(math::numbers::subRoundDown(low, i.low),
                    math::numbers::subRoundUp(high, i.high));
  }
  /// Uses interval arithmetic multiplication
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator*(const Interval &i) const {
    T lp[4] = {math::numbers::mulRoundDown(low, i.low),
               math::numbers::mulRoundDown(high, i.low),
               math::numbers::mulRoundDown(low, i.high),
               math::numbers::mulRoundDown(high, i.high)};
    T hp[4] = {math::numbers::mulRoundUp(low, i.low),
               math::numbers::mulRoundUp(high, i.low),
               math::numbers::mulRoundUp(low, i.high),
               math::numbers::mulRoundUp(high, i.high)};
    return {math::numbers::min({lp[0], lp[1], lp[2], lp[3]}),
            math::numbers::max({hp[0], hp[1], hp[2], hp[3]})};
  }
  /// Uses interval arithmetic division
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator/(const Interval &i) const {
    Interval r = i;
    if (r.low < 0 && r.high > 0)
      return {math::numbers::lowest<T>(), math::numbers::greatest<T>()};
    T lq[4] = {math::numbers::divRoundDown(low, i.low),
               math::numbers::divRoundDown(high, i.low),
               math::numbers::divRoundDown(low, i.high),
               math::numbers::divRoundDown(high, i.high)};
    T hq[4] = {math::numbers::divRoundUp(low, i.low),
               math::numbers::divRoundUp(high, i.low),
               math::numbers::divRoundUp(low, i.high),
               math::numbers::divRoundUp(high, i.high)};
    return {math::numbers::min({lq[0], lq[1], lq[2], lq[3]}),
            math::numbers::max({hq[0], hq[1], hq[2], hq[3]})};
  }
  //                                                                                                          boolean
  HERMES_DEVICE_CALLABLE bool operator==(const Interval<T> &b) const {
    return math::check::is_equal(low, b.low) &&
           math::check::is_equal(high, b.high);
  }

  /// Checks if this interval contains v
  /// \param v
  /// \return
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE bool contains(T v) const {
    return v >= low && v <= high;
  }
  /// Gets interval center value
  /// \return
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE T center() const {
    return (low + high) / 2;
  }
  /// Gets interval radius
  /// \return
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE T radius() const {
    return (high - low) / 2;
  }
  /// Gets interval diameter
  /// \return
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE T width() const { return high - low; }
  /// Checks if interval contains a single value
  /// \return
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE bool isExact() const {
    return high - low == 0;
  }
  /// Computes arithmetic interval square
  /// \return
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Interval sqr() const {
    real_t alow = std::abs(low), ahigh = std::abs(high);
    if (alow > ahigh)
      math::numbers::swap(alow, ahigh);
    if (contains(0))
      return Interval(0, math::numbers::mulRoundUp(ahigh, ahigh));
    return Interval(math::numbers::mulRoundDown(alow, alow),
                    math::numbers::mulRoundUp(ahigh, ahigh));
  }
  /// Computes arithmetic interval square root
  HERMES_NODISCARD HERMES_DEVICE_CALLABLE Interval sqrt() const {
    return {math::numbers::sqrtRoundDown(low),
            math::numbers::sqrtRoundUp(high)};
  }

  T low{0};  //!< lowest interval value
  T high{0}; //!< greatest interval value
};

#define ARITHMETIC_OP(OP)                                                      \
  template <typename T>                                                        \
  HERMES_DEVICE_CALLABLE Interval<T> operator OP(T f, const Interval<T> &i) {  \
    return Interval<T>(f) OP i;                                                \
  }                                                                            \
  template <typename T>                                                        \
  HERMES_DEVICE_CALLABLE Interval<T> operator OP(const Interval<T> &i, T f) {  \
    return i OP Interval<T>(f);                                                \
  }
ARITHMETIC_OP(+)
ARITHMETIC_OP(-)
ARITHMETIC_OP(*)
ARITHMETIC_OP(/)
#undef ARITHMETIC_OP

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(Interval<T>, typename T)
HERMES_PUSH_DEBUG_LINE("I[{}, {}]", object.low, object.high);
HERMES_TO_STRING_DEBUG_METHOD_END

} // namespace hermes
