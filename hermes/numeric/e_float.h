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
///\file e_float.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-07-01
///
///\brief

#ifndef HERMES_HERMES_NUMERIC_E_FLOAT_H
#define HERMES_HERMES_NUMERIC_E_FLOAT_H

#include <hermes/numeric/interval.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                             EFloat
// *********************************************************************************************************************
/// Implements the running error analysis by carrying the error bounds
/// accumulated by a floating point value. It keeps track of the interval of
/// uncertainty of the computed value.
class EFloat {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE inline friend EFloat operator*(f32 f, EFloat fe) { return EFloat(f) * fe; }
  HERMES_DEVICE_CALLABLE inline friend EFloat operator/(f32 f, EFloat fe) { return EFloat(f) / fe; }
  HERMES_DEVICE_CALLABLE inline friend EFloat operator+(f32 f, EFloat fe) { return EFloat(f) + fe; }
  HERMES_DEVICE_CALLABLE inline friend EFloat operator-(f32 f, EFloat fe) { return EFloat(f) - fe; }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE EFloat();
  /// \param v floating point value
  /// \param e absolute error bound
  HERMES_DEVICE_CALLABLE explicit EFloat(f32 v, f32 e = 0.f);
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
  explicit operator float() const;
  //                                                                                                       assignment
  //                                                                                                       arithmetic
  EFloat operator+(EFloat f) const;
  EFloat operator-(EFloat f) const;
  EFloat operator*(EFloat f) const;
  EFloat operator/(EFloat f) const;
  //                                                                                                          boolean
  bool operator==(EFloat f) const;
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \return a bound for the absolute error
  [[nodiscard]] f32 absoluteError() const;
  /// \return lower error interval bound
  [[nodiscard]] f32 upperBound() const;
  /// \return upper error interval bound
  [[nodiscard]] f32 lowerBound() const;
#ifdef NDEBUG
  /// \return relative error
  f64 relativeError() const;
  /// \return highly precise value
  long double preciseValue() const;
#endif
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
private:
  float v;                    //!< computed value
  Interval<f32> err; //!< absolute error bound
#ifdef NDEBUG
  long double ld; //!< highly precise version of v
#endif
};

}

#endif //HERMES_HERMES_NUMERIC_E_FLOAT_H
