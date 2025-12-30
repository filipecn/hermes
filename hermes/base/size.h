/* Copyright (c) 2020, FilipeCN.
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

/// \file   size.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2020-01-28
/// \brief  Set of multi-dimensional size representations

#pragma once

#include <hermes/core/debug.h>

#include <type_traits>

namespace hermes {

// *****************************************************************************
//                                                                        Size2
// *****************************************************************************

/// Holds 2-dimensional size
/// \note T must be an unsigned integer type.
/// \tparam T underlying data type.
template <typename T> struct Size2 {
  static_assert(std::is_same<T, u8>::value || std::is_same<T, u16>::value ||
                    std::is_same<T, u32>::value || std::is_same<T, u64>::value,
                "Size2 must hold an unsigned integer type!");

  HERMES_CPU_GPU Size2() : width{0}, height{0} {};
  HERMES_CPU_GPU explicit Size2(T size) : width(size), height(size) {}
  HERMES_CPU_GPU Size2(T width, T height) : width(width), height(height) {}

  /// \param i Dimension index.
  /// \return Size value in dimension i % 2.
  HERMES_CPU_GPU T operator[](int i) const { return (&width)[i % 2]; }
  /// \param i Dimension index.
  /// \return Reference to size value in dimension i % 2.
  HERMES_CPU_GPU T &operator[](int i) { return (&width)[i % 2]; }

  /// \return [ this[i] + b[i] ]
  HERMES_CPU_GPU Size2<T> operator+(const Size2<T> &b) const {
    return Size2<T>(width + b.width, height + b.height);
  }
  /// \return [ this[i] / n ]
  HERMES_CPU_GPU Size2<T> operator/(T n) const {
    return Size2<T>(width / n, height / n);
  }
  /// \return [ this[i] * s ]
  HERMES_CPU_GPU Size2<T> operator*(T s) const {
    return Size2<T>(width * s, height * s);
  }

  /// \return this[i] == b[i] for all i.
  HERMES_CPU_GPU bool operator==(const Size2<T> &b) const {
    return width == b.width && height == b.height;
  }
  /// \return this[i] != b[i] for any i.
  HERMES_CPU_GPU bool operator!=(const Size2<T> &b) const {
    return width != b.width || height != b.height;
  }

  /// Computes total size area
  /// \return width * height
  HERMES_CPU_GPU T total() const { return width * height; }
  /// \param i Coordinate in width dimension.
  /// \param j Coordinate in height dimension.
  /// \return True if coordinate is inside half-open range [0, size).
  [[nodiscard]] HERMES_CPU_GPU bool contains(int i, int j) const {
    return i >= 0 && j >= 0 && i < static_cast<i64>(width) &&
           j < static_cast<i64>(height);
  }

  T width{0};  //!< 0-th dimension size
  T height{0}; //!< 1-th dimension size
};

// *****************************************************************************
//                                                                        Size3
// *****************************************************************************

/// Holds 3-dimensional size
/// \note T must be an unsigned integer type
/// \tparam T underlying data type.
template <typename T> struct Size3 {
  static_assert(std::is_same<T, u8>::value || std::is_same<T, u16>::value ||
                    std::is_same<T, u32>::value || std::is_same<T, u64>::value,
                "Size3 must hold an unsigned integer type!");

  HERMES_CPU_GPU Size3() : width{0}, height{0}, depth{0} {};
  HERMES_CPU_GPU explicit Size3(T size)
      : width(size), height(size), depth(size) {}
  HERMES_CPU_GPU Size3(T _width, T _height, T _depth)
      : width(_width), height(_height), depth(_depth) {}

  /// \param i Dimension index.
  /// \return Size value in dimension i % 3.
  HERMES_CPU_GPU T operator[](int i) const { return (&width)[i % 3]; }
  /// \param i Dimension index.
  /// \return Reference to size value in dimension i % 3.
  HERMES_CPU_GPU T &operator[](int i) { return (&width)[i % 3]; }

  /// \note This does not check for over-flows.
  /// \return [ this[i] + b[i] ]
  HERMES_CPU_GPU Size3<T> operator+(const Size3<T> &b) const {
    return {width + b.width, height + b.height, depth + b.depth};
  }
  /// \note This does not check for under-flows.
  /// \return [ this[i] - b[i] ]
  HERMES_CPU_GPU Size3<T> operator-(const Size3<T> &b) const {
    return {width - b.width, height - b.height, depth - b.depth};
  }

  /// \return this[i] == b[i] for all i.
  HERMES_CPU_GPU bool operator==(const Size3<T> &b) const {
    return width == b.width && height == b.height && depth == b.depth;
  }
  /// \return this[i] != b[i] for any i.
  HERMES_CPU_GPU bool operator!=(const Size3<T> &b) const {
    return width != b.width || height != b.height || depth != b.depth;
  }

  /// Computes total size area.
  /// \return width * height * depth
  HERMES_CPU_GPU T total() const { return width * height * depth; }
  /// Gets 2-dimensional slice.
  /// \param d1 Dimension index associated to the first 2D dimension index.
  /// \param d2 Dimension index associated to the first 2D dimension index.
  /// \return Slice2(this[d1],this[d2]).
  HERMES_CPU_GPU Size2<T> slice(int d1 = 0, int d2 = 1) const {
    return Size2<T>((&width)[d1], (&width)[d2]);
  }

  T width{0};  //!< 0-th dimension size
  T height{0}; //!< 1-th dimension size
  T depth{0};  //!< 2-th dimension size
};

HERMES_TO_STRING_TEMPLATED_METHOD_BEGIN(Size2<T>, typename T)
HERMES_TO_STRING_METHOD_LINE("Size[{}, {}]", object.width, object.height);
HERMES_TO_STRING_METHOD_END

HERMES_TO_STRING_TEMPLATED_METHOD_BEGIN(Size3<T>, typename T)
HERMES_TO_STRING_METHOD_LINE("Size[{}, {}, {}]", object.width, object.height,
                             object.depth);
HERMES_TO_STRING_METHOD_END

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
