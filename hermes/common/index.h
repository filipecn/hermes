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
///\file index.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-01-28
///
///\brief

#ifndef HERMES_COMMON_INDEX_H
#define HERMES_COMMON_INDEX_H

#include <hermes/common/size.h>
#include <hermes/common/debug.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                             Index2
// *********************************************************************************************************************
/// Holds 2-dimensional index coordinates
///
/// - Usually the field ``i`` is related to the **x** axis in cartesian
/// coordinates, and the field ``j`` is related to the **y** axis.
///\tparam T index type
/// \verbatim embed:rst:leading-slashes
///    .. warning::
///       Index type must be a signed integer type.
/// \endverbatim
template<typename T> struct Index2 {
  static_assert(std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
                    std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Index2 must hold an integer type!");
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP)                                                                                           \
  template<typename U>                                                                                              \
  HERMES_DEVICE_CALLABLE friend Index2<T> operator OP (const Size2<U> &b, const Index2<T> &a) {                     \
    return Index2<T>(b.width OP a.i, b.height OP a.j);  }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP
  //                                                                                                         geometry
  /// \brief Computes the manhattan distance between two indices
  /// \tparam T
  /// \param a **[in]**
  /// \param b **[in]**
  /// \return T
  HERMES_DEVICE_CALLABLE friend T distance(const Index2<T> &a, const Index2<T> &b) {
#ifdef HERMES_DEVICE_CODE
    HERMES_NOT_IMPLEMENTED
#else
    return std::abs(a.i - b.i) + std::abs(a.j - b.j);
#endif
  }
#define MATH_OP(NAME, OP)                                                                                           \
  HERMES_DEVICE_CALLABLE friend Index2<T> NAME(const Index2<T>& a, const Index2<T>& b) {                            \
    return Index2<T>(OP(a.i, b.i), OP(a.j, b.j));  }
#ifdef HERMES_DEVICE_CODE
  MATH_OP(max, max)
  MATH_OP(min, min)
#else
  MATH_OP(max, std::max)
  MATH_OP(min, std::min)
#endif
#undef MATH_OP
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Index2() : i{0}, j{0} {};
  /// Constructor
  /// \param v **[in]** value assigned to both ``i`` and ``j``
  HERMES_DEVICE_CALLABLE explicit Index2(T v) : i(v), j(v) {}
  ///\brief Constructor
  ///\param i **[in]** coordinate value for ``i``
  ///\param j **[in]** coordinate value for ``j``
  HERMES_DEVICE_CALLABLE Index2(T i, T j) : i(i), j(j) {}
  /// \brief Constructor from a Size2 object
  /// - ``i`` receives ``size.with`` and ``j`` receives ``size.height``
  /// \tparam S size type
  /// \param size **[in]**
  template<typename S>
  HERMES_DEVICE_CALLABLE explicit Index2(const Size2<S> &size) : i(size.width), j(size.height) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE T operator[](int d) const { return (&i)[d]; }
  HERMES_DEVICE_CALLABLE T &operator[](int d) { return (&i)[d]; }
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP)                                                                                           \
  HERMES_DEVICE_CALLABLE Index2<T>& operator OP##= (const Index2<T> &b) {                                           \
    i OP##= b.i; j OP##= b.j; return *this; }                                                                      \
  HERMES_DEVICE_CALLABLE Index2<T> operator OP (const Index2<T>& b) const {                                         \
    return {i OP b.i, j OP b.j}; }                                                                                  \
  template<typename U>                                                                                              \
  HERMES_DEVICE_CALLABLE Index2<T>& operator OP##= (const Size2<U> &b) {                                            \
    i OP##= static_cast<T>(b.width); j OP##= static_cast<T>(b.height); return *this; }                                                             \
  template<typename U>                                                                                              \
  HERMES_DEVICE_CALLABLE Index2<T> operator OP (const Size2<U>& b) const {                                          \
    return {i OP static_cast<T>(b.width), j OP static_cast<T>(b.height)}; }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP
  //                                                                                                       relational
#define RELATIONAL_OP(OP, CO)                                                                                       \
  HERMES_DEVICE_CALLABLE bool operator OP (const Index2<T> &b) const {                                              \
    return i OP b.i CO j OP b.j; }                                                                                  \
  template<typename U>                                                                                              \
  HERMES_DEVICE_CALLABLE bool operator OP (const Size2<U>& b)  const {                                              \
    return i OP static_cast<T>(b.width) CO j OP static_cast<T>(b.height); }
  RELATIONAL_OP(==, &&)
  RELATIONAL_OP(!=, ||)
  RELATIONAL_OP(>=, &&)
  RELATIONAL_OP(<=, &&)
  RELATIONAL_OP(<, &&)
  RELATIONAL_OP(>, &&)
#undef RELATIONAL_OP
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// Generates an index with incremented values
  /// \param _i **[in]** value incremented to ``i``
  /// \param _j **[in]** value incremented to ``j``
  /// \return Index2<T> resulting index coordinates
  HERMES_DEVICE_CALLABLE Index2<T> plus(T _i, T _j) const { return Index2<T>(i + _i, j + _j); }
  /// Generates a copy with ``i`` decremented by ``d``
  /// \param d **[in | default = 1]** decrement value
  /// \return Index2<T> resulting index coordinates (``i-d``, ``j``)
  HERMES_DEVICE_CALLABLE Index2<T> left(T d = T(1)) const { return Index2<T>(i - d, j); }
  /// Generates a copy with ``i`` incremented by ``d``
  /// \param d **[in | default = 1]** increment value
  /// \return Index2<T> resulting index coordinates (``i+d``, ``j``)
  HERMES_DEVICE_CALLABLE Index2<T> right(T d = T(1)) const { return Index2<T>(i + d, j); }
  /// Generates a copy with ``j`` decremented by ``d``
  /// \param d **[in | default = 1]** decrement value
  /// \return Index2<T> resulting index coordinates (``i``, ``j-d``)
  HERMES_DEVICE_CALLABLE Index2<T> down(T d = T(1)) const { return Index2<T>(i, j - d); }
  /// Generates a copy with ``j`` incremented by ``d``
  /// \param d **[in | default = 1]** increment value
  /// \return Index2<T> resulting index coordinates (``i``, ``j+d``)
  HERMES_DEVICE_CALLABLE Index2<T> up(T d = T(1)) const { return Index2<T>(i, j + d); }
  /// Clamps to the inclusive range ``[0, size]``
  /// \param s **[in]** upper bound
  HERMES_DEVICE_CALLABLE void clampTo(const size2 &s) {
#ifdef HERMES_DEVICE_CODE
    HERMES_UNUSED_VARIABLE(s);
#else
    i = std::max(0, std::min(i, static_cast<T>(s.width)));
    j = std::max(0, std::min(j, static_cast<T>(s.height)));
#endif
  }
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  /// 0-th coordinate value
  T i = T(0);
  /// 1-th coordinate value
  T j = T(0);
};

// *********************************************************************************************************************
//                                                                                                     Index2Iterator
// *********************************************************************************************************************
/// \tparam T
template<typename T> class Index2Iterator {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Index2Iterator() {}
  HERMES_DEVICE_CALLABLE Index2Iterator(Index2<T> lower, Index2<T> upper)
      : index_(lower), lower_(lower), upper_(upper) {}
  ///\brief Construct a new Index2Iterator object
  ///\param lower **[in]** lower bound
  ///\param upper **[in]** upper bound
  ///\param start **[in]** starting coordinate
  HERMES_DEVICE_CALLABLE Index2Iterator(Index2<T> lower, Index2<T> upper, Index2<T> start)
      : index_(start), lower_(lower), upper_(upper) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
  ///\return Index2Iterator&
  HERMES_DEVICE_CALLABLE Index2Iterator &operator++() {
    index_.i++;
    if (index_.i >= upper_.i) {
      index_.i = lower_.i;
      index_.j++;
      if (index_.j >= upper_.j)
        index_ = upper_;
    }
    return *this;
  }
  //                                                                                                           access
  ///\return const Index2<T>& current index coordinate
  HERMES_DEVICE_CALLABLE const Index2<T> &operator*() const { return index_; }
  //                                                                                                          boolean
  ///\brief are equal? operator
  ///\param other **[in]**
  ///\return bool true if current indices are equal
  HERMES_DEVICE_CALLABLE bool operator==(const Index2Iterator<T> &other) const {
    return index_ == other.index_;
  }
  ///\brief are different? operator
  ///\param other **[in]**
  ///\return bool true if current indices are different
  HERMES_DEVICE_CALLABLE bool operator!=(const Index2Iterator<T> &other) const {
    return index_ != other.index_;
  }

private:
  Index2<T> index_, lower_, upper_;
};

// *********************************************************************************************************************
//                                                                                                        Index2Range
// *********************************************************************************************************************
/// Represents a closed-open range of indices ``[lower, upper)``
///
/// Can be used in a for each loop that iterates over all indices in the range:
/// \verbatim embed:rst:leading-slashes
///    .. code-block:: cpp
///
///       ponos::size2 size(10,10);
///       for(auto ij : ponos::Index2Range<int>(size)) {
///         *ij; // index coordinates
///       }
/// \endverbatim
///\tparam T must be an integer type
template<typename T> class Index2Range {
public:
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE friend Index2Range<T> intersect(const Index2Range<T> &a, const Index2Range<T> &b) {
#ifdef HERMES_DEVICE_CODE
    return Index2Range<T>(Index2<T>(max(a.lower_.i, b.lower_.i), max(a.lower_.i, b.lower_.j)),
                          Index2<T>(min(a.upper_.i, b.upper_.i), min(a.upper_.i, b.upper_.j)));
#else
    return {{std::max(a.lower_.i, b.lower_.i), std::max(a.lower_.i, b.lower_.j)},
            {std::min(a.upper_.i, b.upper_.i), std::min(a.upper_.i, b.upper_.j)}};
#endif
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Index2Range() {}
///\brief Constructs an index range ``[0, {upper_i,upper_j})``
///\param upper_i **[in]** upper bound i
///\param upper_j **[in]** upper bound j
  HERMES_DEVICE_CALLABLE Index2Range(T upper_i, T upper_j) :
      lower_(Index2<T>()), upper_(Index2<T>(upper_i, upper_j)) {}
///\brief Constructs an index range ``[lower, upper)``
///\param lower **[in]** lower bound
///\param upper **[in | default = Index2<T>()]** upper bound
  HERMES_DEVICE_CALLABLE Index2Range(Index2<T> lower, Index2<T> upper) :
      lower_(lower), upper_(upper) {}
/// \brief Constructs an index range ``[0, upper)``
/// \param upper **[in]** upper bound
  HERMES_DEVICE_CALLABLE explicit Index2Range(size2 upper) :
      lower_(Index2<T>()), upper_(Index2<T>(upper.width, upper.height)) {}
  HERMES_DEVICE_CALLABLE bool contains(const Index2<T> &ij) const {
    return ij >= lower_ && ij < upper_;
  }
// *******************************************************************************************************************
//                                                                                                        OPERATORS
// *******************************************************************************************************************
//                                                                                                       relational
  HERMES_DEVICE_CALLABLE bool operator==(const Index2Range<T> &r) const {
    return lower_ == r.lower_ && upper_ == r.upper_;
  }
// *******************************************************************************************************************
//                                                                                                          METHODS
// *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Index2Iterator<T> begin() const {
    return Index2Iterator<T>(lower_, upper_, lower_);
  }
  HERMES_DEVICE_CALLABLE Index2Iterator<T> end() const {
    return Index2Iterator<T>(lower_, upper_, upper_);
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE const Index2<T> &lower() const { return lower_; }
  [[nodiscard]] HERMES_DEVICE_CALLABLE const Index2<T> &upper() const { return upper_; }
  HERMES_DEVICE_CALLABLE T area() const {
    auto d = upper_ - lower_;
    return d.i * d.j;
  }

private:
  Index2<T> lower_, upper_;
};

// *********************************************************************************************************************
//                                                                                                             Index2
// *********************************************************************************************************************
/// Holds 3-dimensional index coordinates
///\tparam T must be an integer type
template<typename T> struct Index3 {
  static_assert(std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
                    std::is_same<T, i32>::value || std::is_same<T, i64>::value,
                "Index3 must hold an integer type!");
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP)                                                                                           \
  template<typename U>                                                                                              \
  HERMES_DEVICE_CALLABLE friend Index3<T> operator OP (const Size3<U> &b, const Index3<T> &a) {                     \
    return Index3<T>(b.width OP a.i, b.height OP a.j, b.depth OP a.k);  }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE Index3() : i(0), j(0), k(0) {}
  HERMES_DEVICE_CALLABLE explicit Index3(T v) : i(v), j(v), k(v) {}
  ///\brief Construct a new Index2 object
  ///\param i **[in]** i coordinate value
  ///\param j **[in]** j coordinate value
  ///\param k **[in]** k coordinate value
  HERMES_DEVICE_CALLABLE Index3(T i, T j, T k) : i(i), j(j), k(k) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                           access
  HERMES_DEVICE_CALLABLE T operator[](int _i) const { return (&i)[_i]; }
  HERMES_DEVICE_CALLABLE T &operator[](int _i) { return (&i)[_i]; }
  //                                                                                                       arithmetic
#define ARITHMETIC_OP(OP)                                                                                           \
  HERMES_DEVICE_CALLABLE Index3<T>& operator OP##= (const Index3<T> &b) {                                           \
    i OP##= b.i; j OP##= b.j; k OP##= b.k; return *this; }                                                          \
  HERMES_DEVICE_CALLABLE Index3<T> operator OP (const Index3<T>& b) const {                                         \
    return {i OP b.i, j OP b.j, k OP b.k}; }                                                                        \
  template<typename U>                                                                                              \
  HERMES_DEVICE_CALLABLE Index3<T>& operator OP##= (const Size3<U> &b) {                                            \
    i OP##= b.width; j OP##= b.height; k OP##= b.depth; return *this; }                                             \
  template<typename U>                                                                                              \
  HERMES_DEVICE_CALLABLE Index3<T> operator OP (const Size3<U>& b) const {                                          \
    return {i OP static_cast<T>(b.width), j OP static_cast<T>(b.height),                                            \
    k OP static_cast<T>(b.depth)}; }
  ARITHMETIC_OP(+)
  ARITHMETIC_OP(-)
  ARITHMETIC_OP(*)
  ARITHMETIC_OP(/)
#undef ARITHMETIC_OP
  //                                                                                                       relational
#define RELATIONAL_OP(OP, CO)                                                                                       \
  HERMES_DEVICE_CALLABLE bool operator OP (const Index3<T> &b) const {                                              \
    return i OP b.i CO j OP b.j CO k OP b.k; }                                                                      \
  template<typename U>                                                                                              \
  HERMES_DEVICE_CALLABLE bool operator OP (const Size3<U>& b)  const {                                              \
    return i OP static_cast<T>(b.width) CO                                                                          \
           j OP static_cast<T>(b.height) CO k OP static_cast<T>(b.depth); }
  RELATIONAL_OP(==, &&)
  RELATIONAL_OP(!=, ||)
  RELATIONAL_OP(>=, &&)
  RELATIONAL_OP(<=, &&)
  RELATIONAL_OP(<, &&)
  RELATIONAL_OP(>, &&)
#undef RELATIONAL_OP
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  T i{0};
  T j{0};
  T k{0};
};

// *********************************************************************************************************************
//                                                                                                     Index3Iterator
// *********************************************************************************************************************
/// \tparam T
template<typename T> class Index3Iterator {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  ///\brief Construct a new Index3Iterator object
  ///\param lower **[in]** lower bound
  ///\param upper **[in]** upper bound
  ///\param start **[in]** starting coordinate
  HERMES_DEVICE_CALLABLE Index3Iterator(Index3<T> lower, Index3<T> upper, Index3<T> start)
      : index_(start), lower_(lower), upper_(upper) {}
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                           access
  ///\return const Index3<T>& current index coordinate
  HERMES_DEVICE_CALLABLE const Index3<T> &operator*() const { return index_; }
  //                                                                                                       arithmetic
  ///\return Index3Iterator&
  HERMES_DEVICE_CALLABLE Index3Iterator &operator++() {
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
  ///\brief are equal? operator
  ///\param other **[in]**
  ///\return bool true if current indices are equal
  HERMES_DEVICE_CALLABLE bool operator==(const Index3Iterator<T> &other) const {
    return index_ == other.index_;
  }
  ///\brief are different? operator
  ///\param other **[in]**
  ///\return bool true if current indices are different
  HERMES_DEVICE_CALLABLE bool operator!=(const Index3Iterator<T> &other) const {
    return index_ != other.index_;
  }

private:
  Index3<T> index_, lower_, upper_;
};

// *********************************************************************************************************************
//                                                                                                        Index3Range
// *********************************************************************************************************************
/// Represents a closed-open range of indices [lower, upper),
/// Can be used in a for each loop
///\tparam T must be an integer type
template<typename T> class Index3Range {
public:
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE friend Index3Range<T> intersect(const Index3Range<T> &a, const Index3Range<T> &b) {
#ifdef HERMES_DEVICE_CODE
    return {Index3<T>(max(a.lower_.i, b.lower_.i), max(a.lower_.i, b.lower_.j),
                      max(a.lower_.k, b.lower_.k)),
            Index3<T>(min(a.upper_.i, b.upper_.i), min(a.upper_.i, b.upper_.j),
                      min(a.upper_.k, b.upper_.k))};
#else
    return {{std::max(a.lower_.i, b.lower_.i), std::max(a.lower_.i, b.lower_.j),
             std::max(a.lower_.k, b.lower_.k)},
            {std::min(a.upper_.i, b.upper_.i), std::min(a.upper_.i, b.upper_.j),
             std::min(a.upper_.k, b.upper_.k)}};
#endif
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  ///\brief Construct a new Index3Range object
  ///\param upper_i **[in]** upper bound i
  ///\param upper_j **[in]** upper bound j
  ///\param upper_k **[in]** upper bound k
  HERMES_DEVICE_CALLABLE Index3Range(T upper_i, T upper_j, T upper_k)
      : lower_(Index3<T>()), upper_(Index3<T>(upper_i, upper_j, upper_k)) {}
  ///\brief Construct a new Index3Range object
  ///\param upper **[in]** upper bound
  ///\param lower **[in]** lower bound
  HERMES_DEVICE_CALLABLE explicit Index3Range(Index3<T> upper)
      : upper_(upper) {}
  /// \param upper
  HERMES_DEVICE_CALLABLE explicit Index3Range(size3 upper)
      : lower_(Index3<T>()),
        upper_(Index3<T>(upper.width, upper.height, upper.depth)) {}
  /// \param lower
  /// \param upper
  HERMES_DEVICE_CALLABLE Index3Range(Index3<T> lower, Index3<T> upper)
      : lower_(lower), upper_(upper) {}
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  ///\return Index3Iterator<T>
  HERMES_DEVICE_CALLABLE Index3Iterator<T> begin() const {
    return Index3Iterator<T>(lower_, upper_, lower_);
  }
  ///\return Index3Iterator<T>
  HERMES_DEVICE_CALLABLE Index3Iterator<T> end() const {
    return Index3Iterator<T>(lower_, upper_, upper_);
  }
  [[nodiscard]] HERMES_DEVICE_CALLABLE size3 size() const {
#ifdef HERMES_DEVICE_CODE
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

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<typename T>
std::ostream &operator<<(std::ostream &o, const Index2<T> &ij) {
  o << "Index[" << ij.i << ", " << ij.j << "]";
  return o;
}
template<typename T>
std::ostream &operator<<(std::ostream &o, const Index3<T> &ijk) {
  o << "Index[" << ijk.i << ", " << ijk.j << ", " << ijk.k << "]";
  return o;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using range2 = Index2Range<i32>;
using range3 = Index3Range<i32>;
using index2 = Index2<i32>;
using index2_8 = Index2<i8>;
using index2_16 = Index2<i16>;
using index2_32 = Index2<i32>;
using index2_64 = Index2<i64>;
using index3 = Index3<i32>;
using index3_8 = Index3<i8>;
using index3_16 = Index3<i16>;
using index3_32 = Index3<i32>;
using index3_64 = Index3<i64>;

} // namespace hermes

#endif