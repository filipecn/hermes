/// Copyright (c) 2021, FilipeCN.
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
///\file interval.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-07-01
///
///\brief Numeric interval
///
///\ingroup numeric
///\addtogroup numeric
/// @{

#ifndef HERMES_HERMES_NUMERIC_INTERVAL_H
#define HERMES_HERMES_NUMERIC_INTERVAL_H

#include <hermes/numeric/math_element.h>
#include <hermes/numeric/numeric.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                           Interval
// *********************************************************************************************************************
/// \brief Represents a numeric interval that supports interval arithmetic
template<typename T>
class Interval : public MathElement<T, 2u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value || std::is_same<T, float>::value
                    || std::is_same<T, double>::value || std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Interval must hold an numeric type!");
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  /// \brief Constructs interval from center and radius
  /// \param c
  /// \param r
  /// \return
  HERMES_DEVICE_CALLABLE static Interval withRadius(real_t c, real_t r) {
    if (r == 0)
      return {c, c};
    return {Numbers::subRoundDown(c, r), Numbers::addRoundUp(c, r)};
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Default constructor
  HERMES_DEVICE_CALLABLE Interval() : low(0), high(0) {}
  /// \brief Constructs from center value
  /// \param v
  HERMES_DEVICE_CALLABLE Interval(T v) : low(v), high(v) {}
  /// \brief Construct from interval values
  /// \param l
  /// \param h
  HERMES_DEVICE_CALLABLE Interval(T l, T h) : low(l), high(h) {}
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                            unary
  /// \brief Gets interval center value
  /// \return
  HERMES_DEVICE_CALLABLE explicit operator T() const { return center(); }
  /// \brief Negates interval
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator-() const { return {-high, -low}; }
  //                                                                                                       arithmetic
  /// \brief Uses interval arithmetic addition
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator+(const Interval &i) const {
    return Interval(Numbers::addRoundDown(low, i.low), Numbers::addRoundUp(high, i.high));
  }
  /// \brief Uses interval arithmetic subtraction
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator-(const Interval &i) const {
    return Interval(Numbers::subRoundDown(low, i.low), Numbers::subRoundUp(high, i.high));
  }
  /// \brief Uses interval arithmetic multiplication
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator*(const Interval &i) const {
    T lp[4] = {Numbers::mulRoundDown(low, i.low), Numbers::mulRoundDown(high, i.low),
               Numbers::mulRoundDown(low, i.high), Numbers::mulRoundDown(high, i.high)};
    T hp[4] = {Numbers::mulRoundUp(low, i.low), Numbers::mulRoundUp(high, i.low),
               Numbers::mulRoundUp(low, i.high), Numbers::mulRoundUp(high, i.high)};
    return {Numbers::min({lp[0], lp[1], lp[2], lp[3]}),
            Numbers::max({hp[0], hp[1], hp[2], hp[3]})};
  }
  /// \brief Uses interval arithmetic division
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE Interval operator/(const Interval &i) const {
    Interval r = i;
    if (r.low < 0 && r.high > 0)
      return {Numbers::lowest<T>(), Numbers::greatest<T>()};
    T lq[4] = {Numbers::divRoundDown(low, i.low), Numbers::divRoundDown(high, i.low),
               Numbers::divRoundDown(low, i.high), Numbers::divRoundDown(high, i.high)};
    T hq[4] = {Numbers::divRoundUp(low, i.low), Numbers::divRoundUp(high, i.low),
               Numbers::divRoundUp(low, i.high), Numbers::divRoundUp(high, i.high)};
    return {Numbers::min({lq[0], lq[1], lq[2], lq[3]}), Numbers::max({hq[0], hq[1], hq[2], hq[3]})};
  }
  //                                                                                                          boolean
  HERMES_DEVICE_CALLABLE bool operator==(const Interval<T> &b) const {
    return Check::is_equal(low, b.low) && Check::is_equal(high, b.high);
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Checks if this interval contains v
  /// \param v
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool contains(T v) const { return v >= low && v <= high; }
  /// \brief Gets interval center value
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE T center() const { return (low + high) / 2; }
  /// \brief Gets interval radius
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE T radius() const { return (high - low) / 2; }
  /// \brief Gets interval diameter
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE T width() const { return high - low; }
  /// \brief Checks if interval contains a single value
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool isExact() const { return high - low == 0; }
  /// \brief Computes arithmetic interval square
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE Interval sqr() const {
    real_t alow = std::abs(low), ahigh = std::abs(high);
    if (alow > ahigh)
      Numbers::swap(alow, ahigh);
    if (contains(0))
      return Interval(0, Numbers::mulRoundUp(ahigh, ahigh));
    return Interval(Numbers::mulRoundDown(alow, alow), Numbers::mulRoundUp(ahigh, ahigh));
  }
  /// \brief Computes arithmetic interval square root
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE Interval sqrt() const {
    return {Numbers::sqrtRoundDown(low), Numbers::sqrtRoundUp(high)};
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T low{0};      //!< lowest interval value
  T high{0};     //!< greatest interval value
};

// *********************************************************************************************************************
//                                                                                                         ARITHMETIC
// *********************************************************************************************************************
#define ARITHMETIC_OP(OP)                                                                                           \
template<typename T>                                                                                                \
HERMES_DEVICE_CALLABLE Interval<T> operator OP (T f, const Interval<T>& i) {                                        \
  return Interval<T>(f) OP i; }                                                                                     \
template<typename T>                                                                                                \
HERMES_DEVICE_CALLABLE Interval<T> operator OP (const Interval<T>& i, T f) {                                        \
  return i OP Interval<T>(f); }
ARITHMETIC_OP(+)
ARITHMETIC_OP(-)
ARITHMETIC_OP(*)
ARITHMETIC_OP(/)
#undef ARITHMETIC_OP
// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
/// \brief Interval support for `std::ostream::operator <<`
/// \tparam T
/// \param os
/// \param i
/// \return
template<typename T>
std::ostream &operator<<(std::ostream &os, const Interval<T> &i) {
  os << "[" << i.low << " " << i.high << "]";
  return os;
}

}

#endif //HERMES_HERMES_NUMERIC_INTERVAL_H

/// @}
