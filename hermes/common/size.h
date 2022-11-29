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
///\brief Set of multi-dimensional size representations
///
///\ingroup common
///\addtogroup common
/// @{

#ifndef HERMES_COMMON_SIZE_H
#define HERMES_COMMON_SIZE_H

#include <hermes/common/defs.h>
#include <type_traits>
#include <iostream>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                              Size2
// *********************************************************************************************************************
/// \brief Holds 2-dimensional size
/// \pre T must be an unsigned integer type
///\tparam T
template<typename T> class Size2 {
  static_assert(std::is_same<T, u8>::value
                    || std::is_same<T, u16>::value ||
                    std::is_same<T, u32>::value || std::is_same<T, u64>::value,
                "Size2 must hold an unsigned integer type!");

public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Default constructor
  HERMES_DEVICE_CALLABLE Size2() : width{0}, height{0} {};
  /// \brief Single value constructor
  /// \param size
  HERMES_DEVICE_CALLABLE explicit Size2(T size) : width(size), height(size) {}
  /// \brief Constructor
  /// \param width
  /// \param height
  HERMES_DEVICE_CALLABLE Size2(T width, T height)
      : width(width), height(height) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                           access
  /// \brief Dimension size const access
  /// \pre i must be inside interval [0,1]
  /// \warning This method does not check input value
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE T operator[](int i) const { return (&width)[i]; }
  /// \brief Dimension size access
  /// \pre i must be inside interval [0,1]
  /// \warning This method does not check input value
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&width)[i]; }
  //                                                                                                       arithmetic
  /// \brief Addition
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE Size2<T> operator+(const Size2<T> &b) const {
    return Size2<T>(width + b.width, height + b.height);
  }
  ///
  /// \param n
  /// \return
  HERMES_DEVICE_CALLABLE Size2<T> operator/(T n) const {
    return Size2<T>(width / n, height / n);
  }
  /// \brief Scalar multiplication
  /// \param s
  /// \return
  HERMES_DEVICE_CALLABLE Size2<T> operator*(T s) const {
    return Size2<T>(width * s, height * s);
  }
  //                                                                                                          boolean
  /// \brief Comparison
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE bool operator==(const Size2<T> &b) const {
    return width == b.width && height == b.height;
  }
  /// \brief Comparison
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE bool operator!=(const Size2<T> &b) const {
    return width != b.width || height != b.height;
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Computes total size area
  /// \return width * height
  HERMES_DEVICE_CALLABLE T total() const { return width * height; }
  /// \brief Checks if coordinate is inside half-open range [0, size)
  /// \param i
  /// \param j
  /// \return
  [[nodiscard]] HERMES_DEVICE_CALLABLE bool contains(int i, int j) const {
    return i >= 0 && j >= 0 && i < static_cast<i64>(width) &&
        j < static_cast<i64>(height);
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T width{0};  //!< 0-th dimension size
  T height{0}; //!< 1-th dimension size
};

// *********************************************************************************************************************
//                                                                                                              Size3
// *********************************************************************************************************************
/// \brief Holds 2-dimensional size
/// \pre T must be an unsigned integer type
/// \tparam T
template<typename T> class Size3 {
  static_assert(std::is_same<T, u8>::value
                    || std::is_same<T, u16>::value ||
                    std::is_same<T, u32>::value || std::is_same<T, u64>::value,
                "Size3 must hold an unsigned integer type!");

public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Default constructor
  HERMES_DEVICE_CALLABLE Size3() : width{0}, height{0}, depth{0} {};
  /// \brief Single value constructor
  /// \param size
  HERMES_DEVICE_CALLABLE explicit Size3(T size) : width(size), height(size), depth(size) {}
  /// \brief Constructor
  /// \param _width
  /// \param _height
  /// \param _depth
  HERMES_DEVICE_CALLABLE Size3(T _width, T _height, T _depth)
      : width(_width), height(_height), depth(_depth) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                           access
  /// \brief Dimension size const access
  /// \pre i must be inside interval [0,2]
  /// \warning This method does not check input value
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE T operator[](int i) const { return (&width)[i]; }
  /// \brief Dimension size access
  /// \pre i must be inside interval [0,2]
  /// \warning This method does not check input value
  /// \param i
  /// \return
  HERMES_DEVICE_CALLABLE T &operator[](int i) { return (&width)[i]; }
  //                                                                                                       arithmetic
  /// \brief Addition
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE Size3<T> operator+(const Size3<T> &b) const {
    return {width + b.width, height + b.height, depth + b.depth};
  }
  /// \brief Subtraction
  /// \warning This method does not check underflow condition
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE Size3<T> operator-(const Size3<T> &b) const {
    return {width - b.width, height - b.height, depth - b.depth};
  }
  //                                                                                                          boolean
  /// \brief Dimension-wise comparison
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE bool operator==(const Size3<T> &b) const {
    return width == b.width && height == b.height && depth == b.depth;
  }
  /// \brief Dimension-wise comparison
  /// \param b
  /// \return
  HERMES_DEVICE_CALLABLE bool operator!=(const Size3<T> &b) const {
    return width != b.width || height != b.height || depth != b.depth;
  }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Computes total size area
  /// \return width * height * depth
  HERMES_DEVICE_CALLABLE T total() const { return width * height * depth; }
  /// \brief Gets 2-dimensional slice
  /// \pre `d1` and `d2` must be inside interval [0,2]
  /// \warning This method does not check input values
  /// \param d1
  /// \param d2
  /// \return
  HERMES_DEVICE_CALLABLE Size2<T> slice(int d1 = 0, int d2 = 1) const {
    return Size2<T>((&width)[d1], (&width)[d2]);
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T width{0};  //!< 0-th dimension size
  T height{0}; //!< 1-th dimension size
  T depth{0};  //!< 2-th dimension size
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
///
/// \tparam T
/// \param o
/// \param s
/// \return
template<typename T>
std::ostream &operator<<(std::ostream &o, const Size2<T> &s) {
  o << "Size[" << s.width << ", " << s.height << "]";
  return o;
}
///
/// \tparam T
/// \param o
/// \param s
/// \return
template<typename T>
std::ostream &operator<<(std::ostream &o, const Size3<T> &s) {
  o << "Size[" << s.width << ", " << s.height << ", " << s.depth << "]";
  return o;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using size2 = Size2<u32>;    //!< u32
using size2_8 = Size2<u8>;   //!< u8
using size2_16 = Size2<u16>; //!< u16
using size2_32 = Size2<u32>; //!< u32
using size2_64 = Size2<u64>; //!< u64
using size3 = Size3<u32>;    //!< u32
using size3_8 = Size3<u8>;   //!< u8
using size3_16 = Size3<u16>; //!< u16
using size3_32 = Size3<u32>; //!< u32
using size3_64 = Size3<u64>; //!< u64

} // namespace hermes

#endif

/// @}
