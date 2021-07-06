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
///\brief

#ifndef HERMES_HERMES_NUMERIC_INTERVAL_H
#define HERMES_HERMES_NUMERIC_INTERVAL_H

#include <hermes/numeric/math_element.h>
#include <hermes/numeric/numeric.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                           Interval
// *********************************************************************************************************************
template<typename T>
class Interval : public MathElement<T, 2u> {
  static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value || std::is_same<T, float>::value
                    || std::is_same<T, double>::value || std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Interval must hold an numeric type!");
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  template<typename U>
  HERMES_DEVICE_CALLABLE friend Interval operator*(U f, Interval fe) {
    return Interval(f) * fe;
  }
  template<typename U>
  HERMES_DEVICE_CALLABLE friend Interval operator/(U f, Interval fe) {
    return Interval(f) / fe;
  }
  template<typename U>
  HERMES_DEVICE_CALLABLE friend Interval operator+(U f, Interval fe) {
    return Interval(f) + fe;
  }
  template<typename U>
  HERMES_DEVICE_CALLABLE friend Interval operator-(U f, Interval fe) {
    return Interval(f) - fe;
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Interval() : low(0), high(0) {}
  HERMES_DEVICE_CALLABLE Interval(T l, T h) : low(l), high(h) {}
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE Interval operator+(const Interval &i) const {
    return Interval(low + i.low, high + i.high);
  }
  HERMES_DEVICE_CALLABLE Interval operator-(const Interval &i) const {
    return Interval(low - i.high, high - i.low);
  }
  HERMES_DEVICE_CALLABLE Interval operator*(const Interval &i) const {
    Interval r;
    T prod[4] = {low * i.low, high * i.low, low * i.high, high * i.high};
#ifdef HERMES_DEVICE_CODE
    r.low = fminf(fminf(prod[0], prod[1]), fminf(prod[2], prod[3]));
    r.high = fmaxf(fmaxf(prod[0], prod[1]), fmaxf(prod[2], prod[3]));
#else
    r.low = std::min(std::min(prod[0], prod[1]), std::min(prod[2], prod[3]));
    r.high = std::max(std::max(prod[0], prod[1]), std::max(prod[2], prod[3]));
#endif
    return r;
  }
  HERMES_DEVICE_CALLABLE Interval operator/(const Interval &i) const {
    Interval r = i;
    if (r.low < 0 && r.high > 0) {
      r.low = -Numbers::lowest<T>();
      r.high = Numbers::greatest<T>();
    } else {
      T div[4] = {low / r.low, high / r.low, low / r.high, high / r.high};
#ifdef HERMES_DEVICE_CODE
      r.low = fminf(fminf(div[0], div[1]), fminf(div[2], div[3]));
      r.high = fmaxf(fmaxf(div[0], div[1]), fmaxf(div[2], div[3]));
#else
      r.low = std::min(std::min(div[0], div[1]), std::min(div[2], div[3]));
      r.high = std::max(std::max(div[0], div[1]), std::max(div[2], div[3]));
#endif
    }
    return r;
  }
  //                                                                                                          boolean
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T low{0}, high{0};
};

}

#endif //HERMES_HERMES_NUMERIC_INTERVAL_H
