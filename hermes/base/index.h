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

/// \file   index.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2020-01-28
///  Set of multi-dimensional integer iterators

#pragma once

#include <hermes/base/size.h>

namespace hermes {

// *****************************************************************************
//                                                                     Index2
// *****************************************************************************

///  Holds 2-dimensional integer index coordinates
/// \note   Usually the field ``i`` is related to the **x** axis in cartesian
///         coordinates, and the field ``j`` is related to the **y** axis.
/// \note   Index type must be a signed integer type.
/// \tparam T index type
template <typename T> struct Index2 {
  static_assert(std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
                    std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Index2 must hold an integer type!");

  /// \brief Represents a closed-open range of indices ``[lower, upper)``
  /// Can be used in a for each loop that iterates over all indices in the
  /// range:
  /// \code{.cpp}
  ///       hermes::size2 size(10,10);
  ///       for(auto ij : hermes::Index2Range<int>(size)) {
  ///         *ij; // index coordinates
  ///       }
  /// \endcode
  /// \tparam T must be an integer type
  class Range {
  public:
    class iterator {
    public:
      /// Default constructor
      HERMES_CPU_GPU iterator() {}
      /// Constructor
      /// \param lower
      /// \param upper
      HERMES_CPU_GPU iterator(Index2<T> lower, Index2<T> upper)
          : index_(lower), lower_(lower), upper_(upper) {}
      /// Construct a new iterator object
      /// \param lower  -  lower bound
      /// \param upper  -  upper bound
      /// \param start  -  starting coordinate
      HERMES_CPU_GPU iterator(Index2<T> lower, Index2<T> upper, Index2<T> start)
          : index_(start), lower_(lower), upper_(upper) {}

      /// \return iterator&
      HERMES_CPU_GPU iterator &operator++() {
        index_.i++;
        if (index_.i >= upper_.i) {
          index_.i = lower_.i;
          index_.j++;
          if (index_.j >= upper_.j)
            index_ = upper_;
        }
        return *this;
      }
      /// \return const Index2<T>& current index coordinate
      HERMES_CPU_GPU const Index2<T> &operator*() const { return index_; }
      /// Computes a flat index based on size
      ///
      /// \f(j * (upper - lower)_i + i\f)
      /// \return
      HERMES_NODISCARD HERMES_CPU_GPU size_t flatIndex() const {
        auto size = upper_ - lower_;
        return index_.j * size.i + index_.i;
      }
      HERMES_NODISCARD HERMES_CPU_GPU bool isBoundary() const {
        return index_.i <= 0 || index_.i >= upper_.i - 1 || index_.j <= 0 ||
               index_.j >= upper_.j - 1;
      }
      /// are equal? operator
      ///\param other  -
      ///\return bool true if current indices are equal
      HERMES_CPU_GPU bool operator==(const iterator &rhs) const {
        return index_ == rhs.index_;
      }
      /// are different? operator
      ///\param rhs  -
      ///\return bool true if current indices are different
      HERMES_CPU_GPU bool operator!=(const iterator &rhs) const {
        return index_ != rhs.index_;
      }

    private:
      Index2<T> index_, lower_, upper_;
    };

    /// \param a
    /// \param b
    /// \return
    HERMES_CPU_GPU friend Index2<T>::Range
    intersect(const Index2<T>::Range &a, const Index2<T>::Range &b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
      return Index2<T>::Range(
          Index2<T>(max(a.lower_.i, b.lower_.i), max(a.lower_.j, b.lower_.j)),
          Index2<T>(min(a.upper_.i, b.upper_.i), min(a.upper_.j, b.upper_.j)));
#else
      return {
          {std::max(a.lower_.i, b.lower_.i), std::max(a.lower_.j, b.lower_.j)},
          {std::min(a.upper_.i, b.upper_.i), std::min(a.upper_.j, b.upper_.j)}};
#endif
    }

    HERMES_CPU_GPU Range() {}
    /// Constructs an index range ``[0, {upper_i,upper_j})``
    ///\param upper_i  -  upper bound i
    ///\param upper_j  -  upper bound j
    HERMES_CPU_GPU Range(T upper_i, T upper_j)
        : lower_(Index2<T>()), upper_(Index2<T>(upper_i, upper_j)) {}
    /// Constructs an index range ``[lower, upper)``
    ///\param lower  -  lower bound
    ///\param upper **[in | default = Index2<T>()]** upper bound
    HERMES_CPU_GPU Range(Index2<T> lower, Index2<T> upper)
        : lower_(lower), upper_(upper) {}
    /// Constructs an index range ``[0, upper)``
    /// \param upper  -  upper bound
    HERMES_CPU_GPU explicit Range(size2 upper)
        : lower_(Index2<T>()), upper_(Index2<T>(upper.width, upper.height)) {}
    /// \param ij
    /// \return
    HERMES_CPU_GPU bool contains(const Index2<T> &ij) const {
      return ij >= lower_ && ij < upper_;
    }

    /// \param r
    /// \return
    HERMES_CPU_GPU bool operator==(const Index2<T>::Range &r) const {
      return lower_ == r.lower_ && upper_ == r.upper_;
    }

    /// \return
    HERMES_CPU_GPU iterator begin() const {
      return iterator(lower_, upper_, lower_);
    }
    /// \return
    HERMES_CPU_GPU iterator end() const {
      return iterator(lower_, upper_, upper_);
    }
    /// \return
    HERMES_NODISCARD HERMES_CPU_GPU const Index2<T> &lower() const {
      return lower_;
    }
    /// \return
    HERMES_NODISCARD HERMES_CPU_GPU const Index2<T> &upper() const {
      return upper_;
    }
    /// \return
    HERMES_CPU_GPU u32 area() const {
      auto d = upper_ - lower_;
      return d.i * d.j;
    }
    HERMES_CPU_GPU size2 size() const {
      return {static_cast<u32>(upper_.i - lower_.i),
              static_cast<u32>(upper_.j - lower_.j)};
    }
    /// Computes a flat index based on size
    ///
    /// \f(j * (upper - lower)_i + i\f)
    /// \return
    HERMES_NODISCARD HERMES_CPU_GPU size_t
    flatIndex(const Index2<T> &ij) const {
      return ij.j * (upper_.i - lower_.i) + ij.i;
    }
    HERMES_NODISCARD HERMES_CPU_GPU bool isBoundary(const Index2<T> &ij) const {
      return ij.i <= 0 || ij.i >= upper_.i - 1 || ij.j <= 0 ||
             ij.j >= upper_.j - 1;
    }

  private:
    Index2<T> lower_, upper_;
  };

#define ARITHMETIC_OP(OP)                                                      \
  template <typename U>                                                        \
  HERMES_CPU_GPU friend Index2<T> operator OP(const Size2<U> &b,               \
                                              const Index2<T> &a) {            \
    return Index2<T>(b.width OP a.i, b.height OP a.j);                         \
  }                                                                            \
  HERMES_CPU_GPU friend Index2<T> operator OP(const Index2<T> &b,              \
                                              const T &a) {                    \
    return Index2<T>(b.i OP a, b.j OP a);                                      \
  }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP
  HERMES_CPU_GPU Index2<T> operator-() const { return {-i, -j}; }

  /// Computes the Manhattan distance between two indices
  ///
  /// \f$\sum_i |a_i - b_i|  \f$
  ///
  /// \tparam T
  /// \param a  -
  /// \param b  -
  /// \return T
  HERMES_CPU_GPU friend T distance(const Index2<T> &a, const Index2<T> &b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    HERMES_NOT_IMPLEMENTED
#else
    return std::abs(a.i - b.i) + std::abs(a.j - b.j);
#endif
  }

#define MATH_OP(NAME, OP)                                                      \
  HERMES_CPU_GPU friend Index2<T> NAME(const Index2<T> &a,                     \
                                       const Index2<T> &b) {                   \
    return Index2<T>(OP(a.i, b.i), OP(a.j, b.j));                              \
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
  MATH_OP(max, max)
  MATH_OP(min, min)
#else
  MATH_OP(max, std::max)
  MATH_OP(min, std::min)
#endif
#undef MATH_OP

  /// Default constructor
  HERMES_CPU_GPU Index2() : i{0}, j{0} {};
  /// Constructor
  /// \param v  -  value assigned to both ``i`` and ``j``
  HERMES_CPU_GPU explicit Index2(T v) : i(v), j(v) {}
  /// Constructor
  /// \param i  -  coordinate value for ``i``
  /// \param j  -  coordinate value for ``j``
  HERMES_CPU_GPU Index2(T i, T j) : i(i), j(j) {}
  /// Constructor from a Size2 object
  /// - ``i`` receives ``size.with`` and ``j`` receives ``size.height``
  /// \tparam S size type
  /// \param size  -
  template <typename S>
  HERMES_CPU_GPU explicit Index2(const Size2<S> &size)
      : i(size.width), j(size.height) {}

  /// \param d coordinate index
  /// \pre d must be in [0,1]
  /// \warning the value of d is not checked
  /// \return
  HERMES_CPU_GPU T operator[](int d) const { return (&i)[d]; }
  /// \param d coordinate index
  /// \pre d must be in [0,1]
  /// \warning the value of d is not checked
  /// \return
  HERMES_CPU_GPU T &operator[](int d) { return (&i)[d]; }

#define ARITHMETIC_OP(OP)                                                      \
  HERMES_CPU_GPU Index2<T> &operator OP## = (const Index2<T> &b) {             \
    i OP## = b.i;                                                              \
    j OP## = b.j;                                                              \
    return *this;                                                              \
  }                                                                            \
  HERMES_CPU_GPU Index2<T> operator OP(const Index2<T> &b) const {             \
    return {i OP b.i, j OP b.j};                                               \
  }                                                                            \
  template <typename U>                                                        \
  HERMES_CPU_GPU Index2<T> &operator OP## = (const Size2<U> &b) {              \
    i OP## = static_cast<T>(b.width);                                          \
    j OP## = static_cast<T>(b.height);                                         \
    return *this;                                                              \
  }                                                                            \
  template <typename U>                                                        \
  HERMES_CPU_GPU Index2<T> operator OP(const Size2<U> &b) const {              \
    return {i OP static_cast<T>(b.width), j OP static_cast<T>(b.height)};      \
  }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP

#define RELATIONAL_OP(OP, CO)                                                  \
  HERMES_CPU_GPU bool operator OP(const Index2<T> &b) const {                  \
    return i OP b.i CO j OP b.j;                                               \
  }                                                                            \
  template <typename U>                                                        \
  HERMES_CPU_GPU bool operator OP(const Size2<U> &b) const {                   \
    return i OP static_cast<T>(b.width) CO j OP static_cast<T>(b.height);      \
  }
  RELATIONAL_OP(==, &&)
  RELATIONAL_OP(!=, ||)
  RELATIONAL_OP(>=, &&)
  RELATIONAL_OP(<=, &&)
  RELATIONAL_OP(<, &&)
  RELATIONAL_OP(>, &&)
#undef RELATIONAL_OP

  /// Generates an index with incremented values
  /// \param _i  -  value incremented to ``i``
  /// \param _j  -  value incremented to ``j``
  /// \return Index2<T> resulting index coordinates
  HERMES_CPU_GPU Index2<T> plus(T _i, T _j) const {
    return Index2<T>(i + _i, j + _j);
  }
  /// Generates a copy with ``i`` decremented by ``d``
  /// \param d **[in | default = 1]** decrement value
  /// \return Index2<T> resulting index coordinates (``i-d``, ``j``)
  HERMES_CPU_GPU Index2<T> left(T d = T(1)) const {
    return Index2<T>(i - d, j);
  }
  /// Generates a copy with ``i`` incremented by ``d``
  /// \param d **[in | default = 1]** increment value
  /// \return Index2<T> resulting index coordinates (``i+d``, ``j``)
  HERMES_CPU_GPU Index2<T> right(T d = T(1)) const {
    return Index2<T>(i + d, j);
  }
  /// Generates a copy with ``j`` decremented by ``d``
  /// \param d **[in | default = 1]** decrement value
  /// \return Index2<T> resulting index coordinates (``i``, ``j-d``)
  HERMES_CPU_GPU Index2<T> down(T d = T(1)) const {
    return Index2<T>(i, j - d);
  }
  /// Generates a copy with ``j`` incremented by ``d``
  /// \param d **[in | default = 1]** increment value
  /// \return Index2<T> resulting index coordinates (``i``, ``j+d``)
  HERMES_CPU_GPU Index2<T> up(T d = T(1)) const { return Index2<T>(i, j + d); }
  /// Clamps to the inclusive range ``[0, size]``
  /// \param s  -  upper bound
  HERMES_CPU_GPU void clampTo(const size2 &s) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    HERMES_UNUSED_VARIABLE(s);
#else
    i = std::max(0, std::min(i, static_cast<T>(s.width)));
    j = std::max(0, std::min(j, static_cast<T>(s.height)));
#endif
  }

  /// 0-th coordinate value
  T i = T(0);
  /// 1-th coordinate value
  T j = T(0);
};

// *****************************************************************************
//                                                                     Index3
// *****************************************************************************

/// Holds 3-dimensional index coordinates
/// \tparam T must be an integer type
template <typename T> struct Index3 {
  static_assert(std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
                    std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Index3 must hold an integer type!");

  /// Represents a closed-open range of indices [lower, upper),
  ///
  /// Can be used in a for each loop that iterates over all indices in the
  /// range:
  /// \code{.cpp}
  ///       hermes::size3 size(10);
  ///       for(auto ij : hermes::Index3Range<int>(size)) {
  ///         *ij; // index coordinates
  ///       }
  /// \endcode
  ///\tparam T must be an integer type
  class Range {
  public:
    class iterator {
    public:
      HERMES_CPU_GPU iterator() {}
      /// Construct a new iterator object
      ///\param lower  -  lower bound
      ///\param upper  -  upper bound
      ///\param start  -  starting coordinate
      HERMES_CPU_GPU iterator(Index3<T> lower, Index3<T> upper, Index3<T> start)
          : index_(start), lower_(lower), upper_(upper) {}
      /// \param upper
      HERMES_CPU_GPU explicit iterator(Index3<T> upper) : upper_(upper) {}

      /// \return const Index3<T>& current index coordinate
      HERMES_CPU_GPU const Index3<T> &operator*() const { return index_; }
      /// Computes a flat index based on size
      ///
      /// \f(k * (d_i * d_j) + j * d_i + i\f)
      ///
      /// where \f(d = upper - lower\f)
      /// \return
      HERMES_NODISCARD HERMES_CPU_GPU size_t flatIndex() const {
        auto size = upper_ - lower_;
        return index_.k * (size.i * size.j) + index_.j * size.i + index_.i;
      }
      ///\return iterator&
      HERMES_CPU_GPU iterator &operator++() {
        index_.i++;
        if (index_.i >= upper_.i) {
          index_.i = lower_.i;
          index_.j++;
          if (index_.j >= upper_.j) {
            index_.j = lower_.j;
            index_.k++;
            if (index_.k >= upper_.k)
              index_ = upper_;
          }
        }
        return *this;
      }
      //                                                                                                          boolean
      /// are equal? operator
      ///\param other  -
      ///\return bool true if current indices are equal
      HERMES_CPU_GPU bool operator==(const iterator &other) const {
        return index_ == other.index_;
      }
      /// are different? operator
      ///\param other  -
      ///\return bool true if current indices are different
      HERMES_CPU_GPU bool operator!=(const iterator &other) const {
        return index_ != other.index_;
      }

    private:
      Index3<T> index_, lower_, upper_;
    };

    /// \param a
    /// \param b
    /// \return
    HERMES_CPU_GPU friend Index3<T>::Range
    intersect(const Index3<T>::Range &a, const Index3<T>::Range &b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
      return {
          Index3<T>(max(a.lower_.i, b.lower_.i), max(a.lower_.j, b.lower_.j),
                    max(a.lower_.k, b.lower_.k)),
          Index3<T>(min(a.upper_.i, b.upper_.i), min(a.upper_.j, b.upper_.j),
                    min(a.upper_.k, b.upper_.k))};
#else
      return {
          {std::max(a.lower_.i, b.lower_.i), std::max(a.lower_.i, b.lower_.j),
           std::max(a.lower_.k, b.lower_.k)},
          {std::min(a.upper_.i, b.upper_.i), std::min(a.upper_.i, b.upper_.j),
           std::min(a.upper_.k, b.upper_.k)}};
#endif
    }
    /// Construct a new Index3Range object
    ///\param upper_i - upper bound i
    ///\param upper_j - upper bound j
    ///\param upper_k - upper bound k
    HERMES_CPU_GPU Range(T upper_i, T upper_j, T upper_k)
        : lower_(Index3<T>()), upper_(Index3<T>(upper_i, upper_j, upper_k)) {}
    /// Construct a new Index3Range object
    ///\param upper - upper bound
    HERMES_CPU_GPU explicit Range(Index3<T> upper) : upper_(upper) {}
    /// \param upper
    HERMES_CPU_GPU explicit Range(size3 upper)
        : lower_(Index3<T>()),
          upper_(Index3<T>(upper.width, upper.height, upper.depth)) {}
    /// \param lower
    /// \param upper
    HERMES_CPU_GPU Range(Index3<T> lower, Index3<T> upper)
        : lower_(lower), upper_(upper) {}
    /// Computes a flat index based on size
    ///
    /// \f(k * (d_i * d_j) + j * d_i + i\f)
    ///
    /// where \f(d = upper - lower\f)
    /// \param ijk
    /// \return
    HERMES_CPU_GPU size_t flatIndex(const Index3<T> &ijk) const {
      auto size = upper_ - lower_;
      return ijk.k * (size.i * size.j) + ijk.j * size.i + ijk.i;
    }
    ///\return iterator<T>
    HERMES_CPU_GPU iterator begin() const {
      return iterator(lower_, upper_, lower_);
    }
    ///\return iterator<T>
    HERMES_CPU_GPU iterator end() const {
      return iterator(lower_, upper_, upper_);
    }
    /// \f(|upper - lower|_i\f)
    /// \return
    HERMES_NODISCARD HERMES_CPU_GPU size3 size() const {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
      return size3(std::abs(upper_[0] - lower_[0]),
                   std::abs(upper_[1] - lower_[1]),
                   std::abs(upper_[2] - lower_[2]));
#else
      return size3(std::abs(upper_[0] - lower_[0]),
                   std::abs(upper_[1] - lower_[1]),
                   std::abs(upper_[2] - lower_[2]));
#endif
    }

  private:
    Index3<T> lower_, upper_;
  };

#define ARITHMETIC_OP(OP)                                                      \
  template <typename U>                                                        \
  HERMES_CPU_GPU friend Index3<T> operator OP(const Size3<U> &b,               \
                                              const Index3<T> &a) {            \
    return Index3<T>(b.width OP a.i, b.height OP a.j, b.depth OP a.k);         \
  }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP

  HERMES_CPU_GPU Index3() : i(0), j(0), k(0) {}
  HERMES_CPU_GPU explicit Index3(T v) : i(v), j(v), k(v) {}
  /// Construct a new Index2 object
  ///\param i  -  i coordinate value
  ///\param j  -  j coordinate value
  ///\param k  -  k coordinate value
  HERMES_CPU_GPU Index3(T i, T j, T k) : i(i), j(j), k(k) {}

  /// \param _i
  /// \pre _i must be in [0, 2]
  /// \warning the value of _i is not checked
  HERMES_CPU_GPU T operator[](int _i) const { return (&i)[_i]; }
  /// \param _i
  /// \pre _i must be in [0, 2]
  /// \warning the value of _i is not checked
  HERMES_CPU_GPU T &operator[](int _i) { return (&i)[_i]; }

#define ARITHMETIC_OP(OP)                                                      \
  HERMES_CPU_GPU Index3<T> &operator OP## = (const Index3<T> &b) {             \
    i OP## = b.i;                                                              \
    j OP## = b.j;                                                              \
    k OP## = b.k;                                                              \
    return *this;                                                              \
  }                                                                            \
  HERMES_CPU_GPU Index3<T> operator OP(const Index3<T> &b) const {             \
    return {i OP b.i, j OP b.j, k OP b.k};                                     \
  }                                                                            \
  template <typename U>                                                        \
  HERMES_CPU_GPU Index3<T> &operator OP## = (const Size3<U> &b) {              \
    i OP## = b.width;                                                          \
    j OP## = b.height;                                                         \
    k OP## = b.depth;                                                          \
    return *this;                                                              \
  }                                                                            \
  template <typename U>                                                        \
  HERMES_CPU_GPU Index3<T> operator OP(const Size3<U> &b) const {              \
    return {i OP static_cast<T>(b.width), j OP static_cast<T>(b.height),       \
            k OP static_cast<T>(b.depth)};                                     \
  }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP

#define RELATIONAL_OP(OP, CO)                                                  \
  HERMES_CPU_GPU bool operator OP(const Index3<T> &b) const {                  \
    return i OP b.i CO j OP b.j CO k OP b.k;                                   \
  }                                                                            \
  template <typename U>                                                        \
  HERMES_CPU_GPU bool operator OP(const Size3<U> &b) const {                   \
    return i OP static_cast<T>(b.width) CO j OP static_cast<T>(b.height)       \
        CO k OP static_cast<T>(b.depth);                                       \
  }
  RELATIONAL_OP(==, &&)
  RELATIONAL_OP(!=, ||)
  RELATIONAL_OP(>=, &&)
  RELATIONAL_OP(<=, &&)
  RELATIONAL_OP(<, &&)
  RELATIONAL_OP(>, &&)
#undef RELATIONAL_OP

  /// 0-th coordinate value
  T i{0};
  /// 1-th coordinate value
  T j{0};
  /// 2-th coordinate value
  T k{0};
};

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(Index2<T>, typename T)
HERMES_PUSH_DEBUG_LINE("Index[{}, {}]", object.i, object.j);
HERMES_TO_STRING_DEBUG_METHOD_END

HERMES_TO_STRING_DEBUG_TEMPLATED_METHOD_BEGIN(Index3<T>, typename T)
HERMES_PUSH_DEBUG_LINE("Size[{}, {}, {}]", object.i, object.j, object.j);
HERMES_TO_STRING_DEBUG_METHOD_END

using range2 = Index2<i32>::Range;    //!< i32
using range2_64 = Index2<i64>::Range; //!< i64
using range3 = Index3<i32>::Range;    //!< i32
using index2 = Index2<i32>;           //!< i32
using index2_8 = Index2<i8>;          //!< i8
using index2_16 = Index2<i16>;        //!< i16
using index2_32 = Index2<i32>;        //!< i32
using index2_64 = Index2<i64>;        //!< i64
using index3 = Index3<i32>;           //!< i32
using index3_8 = Index3<i8>;          //!< i8
using index3_16 = Index3<i16>;        //!< i16
using index3_32 = Index3<i32>;        //!< i32
using index3_64 = Index3<i64>;        //!< i64

} // namespace hermes
