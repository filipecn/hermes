/// Copyright (c) 2020, FilipeCN.
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
///\file size.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-28
///
///\brief

#ifndef HERMES_COMMON_SIZE_H
#define HERMES_COMMON_SIZE_H

#include <hermes/common/defs.h>
#include <type_traits>
#include <iostream>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                              Size2
// *********************************************************************************************************************
/// Holds 2-dimensional size
///\tparam T must be an unsigned integer type
template<typename T> class Size2 {
  static_assert(std::is_same<T, u8>::value
                    || std::is_same<T, u16>::value ||
                    std::is_same<T, u32>::value || std::is_same<T, u64>::value,
                "Size2 must hold an unsigned integer type!");

public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Size2() : width{0}, height{0} {};
  HERMES_DEVICE_CALLABLE explicit Size2(T size) : width(size), height(size) {}
  HERMES_DEVICE_CALLABLE Size2(T width, T height)
      : width(width), height(height) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE T operator[](int i) const { return (&width)[i]; }
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&width)[i]; }
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE Size2<T> operator+(const Size2<T> &b) const {
    return Size2<T>(width + b.width, height + b.height);
  }
  HERMES_DEVICE_CALLABLE Size2<T> operator*(T s) const {
    return Size2<T>(width * s, height * s);
  }
  //                                                                                                          boolean
  HERMES_DEVICE_CALLABLE bool operator==(const Size2<T> &b) const {
    return width == b.width && height == b.height;
  }
  HERMES_DEVICE_CALLABLE bool operator!=(const Size2<T> &b) const {
    return width != b.width || height != b.height;
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE T total() const { return width * height; }
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool contains(int i, int j) const {
    return i >= 0 && j >= 0 && i < static_cast<i64>(width) &&
        j < static_cast<i64>(height);
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T width{0};
  T height{0};
};

// *********************************************************************************************************************
//                                                                                                              Size3
// *********************************************************************************************************************
/// Holds 2-dimensional size
///\tparam T must be an unsigned integer type
template<typename T> class Size3 {
  static_assert(std::is_same<T, u8>::value
                    || std::is_same<T, u16>::value ||
                    std::is_same<T, u32>::value || std::is_same<T, u64>::value,
                "Size3 must hold an unsigned integer type!");

public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Size3() : width{0}, height{0}, depth{0} {};
  explicit Size3(T size) : width(size), height(size), depth(size) {}
  HERMES_DEVICE_CALLABLE Size3(T _width, T _height, T _depth)
      : width(_width), height(_height), depth(_depth) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE T operator[](int i) const { return (&width)[i]; }
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&width)[i]; }
  //                                                                                                       arithmetic
  HERMES_DEVICE_CALLABLE Size3<T> operator+(const Size3<T> &b) const {
    return {width + b.width, height + b.height, depth + b.depth};
  }
  HERMES_DEVICE_CALLABLE Size3<T> operator-(const Size3<T> &b) const {
    return {width - b.width, height - b.height, depth - b.depth};
  }
  //                                                                                                          boolean
  HERMES_DEVICE_CALLABLE bool operator==(const Size3<T> &b) const {
    return width == b.width && height == b.height && depth == b.depth;
  }
  HERMES_DEVICE_CALLABLE bool operator!=(const Size3<T> &b) const {
    return width != b.width || height != b.height || depth != b.depth;
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE T total() const { return width * height * depth; }
  HERMES_DEVICE_CALLABLE Size2<T> slice(int d1 = 0, int d2 = 1) const {
    return Size2<T>((&width)[d1], (&width)[d2]);
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T width{0};
  T height{0};
  T depth{0};
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
std::ostream &operator<<(std::ostream &o, const Size2<T> &s) {
  o << "Size[" << s.width << ", " << s.height << "]";
  return o;
}
template<typename T>
std::ostream &operator<<(std::ostream &o, const Size3<T> &s) {
  o << "Size[" << s.width << ", " << s.height << ", " << s.depth << "]";
  return o;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using size2 = Size2<u32>;
using size2_8 = Size2<u8>;
using size2_16 = Size2<u16>;
using size2_32 = Size2<u32>;
using size2_64 = Size2<u64>;
using size3 = Size3<u32>;
using size3_8 = Size3<u8>;
using size3_16 = Size3<u16>;
using size3_32 = Size3<u32>;
using size3_64 = Size3<u64>;

} // namespace hermes

#endif