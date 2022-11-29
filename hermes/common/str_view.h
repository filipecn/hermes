/// Copyright (c) 2022, FilipeCN.
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
///\file str_view.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-11-12
///
///\brief

#ifndef HERMES_COMMON_STR_VIEW_H
#define HERMES_COMMON_STR_VIEW_H

#include <hermes/common/result.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                       ConstStrView
// *********************************************************************************************************************
class ConstStrView {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  /// \brief Constructs a sub-string view from a std::string object.
  /// \param str input string.
  /// \param pos position of the first character of the sub-string in str.
  /// \param len number of characters of the sub-string. If len = -1, then the size is str.size() - pos.
  /// \return The ConstStrView object representing the sub-string of str.
  static Result<ConstStrView> from(const std::string &str, size_t pos = 0, i64 len = -1);
  /// \brief Constructs a sub-string view reference from a raw string data.
  /// \note This function does not check for the real ending of str.
  /// \param str input raw string pointer.
  /// \param len number of characters in the sub-string (assumed to be within str bounds).
  /// \param pos position of the first character of the sub-string in str (assumed to be within str bounds).
  /// \return The ConstStrView object representing the sub-string of str.
  static Result<ConstStrView> from(const char *str, size_t len, size_t pos = 0);
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  //                                                                                                       arithmetic
  //                                                                                                          boolean
  /// \brief Checks if this sub-string with the string s are equal.
  /// \param s
  /// \return true if both string are equal.
  bool operator==(const std::string &s) const;
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Gets the size of the view.
  /// \return the number of characters in the sub-string.
  [[nodiscard]] size_t size() const;
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
private:
  // *******************************************************************************************************************
  //                                                                                             PRIVATE CONSTRUCTORS
  // *******************************************************************************************************************
  /// Parameter constructor
  /// \note This function does not check for the real ending of str.
  /// \param str raw string data
  /// \param start position of the first character of the sub-string in str (assumed to be within the str bounds).
  /// \param end one position after the last character of the sub-string  (assumed to respect the str bounds).
  ConstStrView(const char *str, size_t start, size_t end);

  const char *str_{nullptr};
  size_t start_{0};
  size_t end_{0};
  size_t size_{0};
};

}

#endif //HERMES_COMMON_STR_VIEW_H
