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
///\file ray.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-27
///
///\brief

#ifndef HERMES_HERMES_GEOMETRY_RAY_H
#define HERMES_HERMES_GEOMETRY_RAY_H

#include <hermes/geometry/point.h>
#include <hermes/geometry/vector.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                               Ray2
// *********************************************************************************************************************
class Ray2 {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  Ray2();
  Ray2(const point2 &origin, const vec2 &direction);
  virtual ~Ray2() = default;
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  point2 operator()(float t) const { return o + d * t; }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  point2 o;
  vec2 d;
};

// *********************************************************************************************************************
//                                                                                                               Ray3
// *********************************************************************************************************************
class Ray3 {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  Ray3();
  Ray3(const point3 &origin, const vec3 &direction);
  virtual ~Ray3() = default;
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  point3 operator()(float t) const { return o + d * t; }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  point3 o;
  vec3 d;
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
inline std::ostream &operator<<(std::ostream &os, const Ray2 &r) {
  os << "[Ray]\n";
  os << r.o << r.d;
  return os;
}
inline std::ostream &operator<<(std::ostream &os, const Ray3 &r) {
  os << "[Ray]\n";
  os << r.o << r.d;
  return os;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
typedef Ray2 ray2;
typedef Ray3 ray3;

} // namespace hermes

#endif //HERMES_HERMES_GEOMETRY_RAY_H
