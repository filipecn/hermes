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
///\file e_float.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-07-01
///
///\brief

#include <hermes/numeric/e_float.h>

namespace hermes {

HERMES_DEVICE_CALLABLE EFloat::EFloat() {}

HERMES_DEVICE_CALLABLE EFloat::EFloat(f32 v, f32 e) : v(v) {
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

HERMES_DEVICE_CALLABLE EFloat::operator float() const { return v; }

HERMES_DEVICE_CALLABLE EFloat::operator double() const { return v; }

HERMES_DEVICE_CALLABLE float EFloat::absoluteError() const { return err.high - err.low; }

HERMES_DEVICE_CALLABLE float EFloat::upperBound() const { return err.high; }

HERMES_DEVICE_CALLABLE float EFloat::lowerBound() const { return err.low; }

HERMES_DEVICE_CALLABLE EFloat EFloat::operator+(EFloat f) const {
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

HERMES_DEVICE_CALLABLE EFloat EFloat::operator*(EFloat f) const {
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

HERMES_DEVICE_CALLABLE EFloat EFloat::operator/(EFloat f) const {
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

HERMES_DEVICE_CALLABLE EFloat EFloat::operator-(EFloat f) const {
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

HERMES_DEVICE_CALLABLE bool EFloat::operator==(EFloat f) const { return v == f.v; }

}
