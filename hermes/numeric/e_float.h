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
///\brief Floating-point with attached error
///
///\ingroup numeric
///\addtogroup numeric
/// @{

#ifndef HERMES_HERMES_NUMERIC_E_FLOAT_H
#define HERMES_HERMES_NUMERIC_E_FLOAT_H

#include <hermes/numeric/interval.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                             EFloat
// *********************************************************************************************************************
/// \brief Represents a value with error bounds
///
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
  /// \brief Default constructor
  HERMES_DEVICE_CALLABLE EFloat() {}
  /// \brief Constructs from value and error size
  /// \param v floating point value
  /// \param e absolute error bound
  HERMES_DEVICE_CALLABLE explicit EFloat(f32 v, f32 e = 0.f) : v(v) {
#ifdef NDEBUG
    ld = v;
#endif
    if (e == 0.)
      err.low = err.high = v;
    else {
      err.low = Numbers::nextFloatDown(v);
      err.high = Numbers::nextFloatUp(v);
    }
  }
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                          casting
  /// \brief Casts to 32 bit floating point
  /// \return
  HERMES_DEVICE_CALLABLE explicit operator float() const { return v; }
  /// \brief Casts to 64 bit floating point
  /// \return
  HERMES_DEVICE_CALLABLE explicit operator double() const { return v; }
  //                                                                                                       assignment
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE EFloat operator+(EFloat f) const {
    EFloat r;
    r.v = v + f.v;
#ifdef NDEBUG
    r.ld = ld + f.ld;
#endif
    r.err = err + f.err;
    r.err.low = Numbers::nextFloatDown(r.err.low);
    r.err.high = Numbers::nextFloatUp(r.err.high);
    return r;
  }
  HERMES_DEVICE_CALLABLE EFloat operator-(EFloat f) const {
    EFloat r;
    r.v = v - f.v;
#ifdef NDEBUG
    r.ld = ld - f.ld;
#endif
    r.err = err - f.err;
    r.err.low = Numbers::nextFloatDown(r.err.low);
    r.err.high = Numbers::nextFloatUp(r.err.high);
    return r;
  }
  HERMES_DEVICE_CALLABLE EFloat operator*(EFloat f) const {
    EFloat r;
    r.v = v * f.v;
#ifdef NDEBUG
    r.ld = ld * f.ld;
#endif
    r.err = err * f.err;
    r.err.low = Numbers::nextFloatDown(r.err.low);
    r.err.high = Numbers::nextFloatUp(r.err.high);
    return r;
  }
  HERMES_DEVICE_CALLABLE EFloat operator/(EFloat f) const {
    EFloat r;
    r.v = v / f.v;
#ifdef NDEBUG
    r.ld = ld / f.ld;
#endif
    r.err = err / f.err;
    r.err.low = Numbers::nextFloatDown(r.err.low);
    r.err.high = Numbers::nextFloatUp(r.err.high);
    return r;
  }
  //                                                                                                          boolean
  HERMES_DEVICE_CALLABLE bool operator==(EFloat f) const { return v == f.v; }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Computes the absolute error carried by this number
  /// \return a bound for the absolute error
  [[nodiscard]] HERMES_DEVICE_CALLABLE f32 absoluteError() const { return err.high - err.low; }
  /// \brief Computes the upper error bound carried by this number
  /// \return lower error interval bound
  [[nodiscard]] HERMES_DEVICE_CALLABLE f32 upperBound() const { return err.high; }
  /// \brief Computes the lower error bound carried by this number
  /// \return upper error interval bound
  [[nodiscard]] HERMES_DEVICE_CALLABLE f32 lowerBound() const { return err.low; }
#ifdef NDEBUG
  /// \brief Gets relative error
  /// \return relative error
  f64 relativeError() const;
  /// \brief Gets highly precise value
  /// \return highly precise value
  long double preciseValue() const;
#endif
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
private:
  float v;                //!< computed value
  Interval<f32> err;      //!< absolute error bound
#ifdef NDEBUG
  long double ld;         //!< highly precise version of v
#endif
};

}

#endif //HERMES_HERMES_NUMERIC_E_FLOAT_H

/// @}
