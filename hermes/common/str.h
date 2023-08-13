//// Copyright (c) 2020, FilipeCN.
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
///\file str.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-10-07
///
///\brief String utils
///
///\ingroup common
///\addtogroup common
/// @{

#ifndef HERMES_COMMON_STR_H
#define HERMES_COMMON_STR_H

#include <string>
#include <sstream>
#include <vector>
#include <regex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <hermes/common/defs.h>
#include <hermes/common/str_view.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                                Str
// *********************************************************************************************************************
/// \brief String class and set of string functions
class Str {
public:
  // *******************************************************************************************************************
  //                                                                                                    STATIC FIELDS
  // *******************************************************************************************************************
  struct regex {
    static const char floating_point_number[];
    static const char integer_number[];
    static const char alpha_numeric_word[];
    static const char c_identifier[];
    //                                                                                                            regex
    /// \brief Checks if a string s matches exactly a regular expression
    /// \param s input string
    /// \param pattern regex pattern
    /// \param flags [optional] controls how pattern is matched
    /// \return true if s matches exactly the pattern
    static bool match(const std::string &s, const std::string &pattern,
                      std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
    /// \brief Checks if any substring of s matches a regular expression
    /// \param s input string
    /// \param pattern regex pattern
    /// \param flags [optional] controls how pattern is matched
    /// \return true if s contains the pattern
    static bool contains(const std::string &s, const std::string &pattern,
                         std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
    /// \brief Search the first substrings of s that matches the pattern
    /// \param s input string
    /// \param pattern regular expression pattern
    /// \param flags [optional] controls how pattern is matched
    /// \return std match object containing the first match
    static std::smatch search(const std::string &s, const std::string &pattern,
                              std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
    /// \brief Iterate over all substrings of s that match the pattern
    /// \param s input string
    /// \param pattern regular expression pattern
    /// \param callback called for each match
    /// \param flags [optional] controls how pattern is matched
    /// \return true if any match occurred
    static bool search(std::string s,
                       const std::string &pattern,
                       const std::function<void(const std::smatch &)> &callback,
                       std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
    /// \brief Replaces all matches of pattern in s by format
    /// \param s input string
    /// \param pattern regular expression pattern
    /// \param format replacement format
    /// \param flags [optional] controls how pattern is matched and how format is replaced
    /// \return A copy of s with all replacements
    static std::string replace(const std::string &s, const std::string &pattern, const std::string &format,
                               std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  };
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  //                                                                                                          queries
  /// \brief Checks if s has prefix p.
  /// \param p prefix string
  /// \param s string
  /// \return true if p is prefix of s
  static bool isPrefix(const std::string& p, const std::string& s);
  //                                                                                                       formatting
  /// \brief Abbreviates a string to fit in a string of width characters.
  /// \note If width >= string size, no abbreviation occurs.
  /// \param s input string
  /// \param width final character count
  /// \param fmt a three-character string describing the abbreviation type: ("..s", "s.s", ".s.", or "..s").
  ///         where 's' represents the input string contents and '.' the abbreviated portion of s.
  /// \return abbreviated string
  static std::string abbreviate(const std::string &s, size_t width, const char fmt[4] = "s.s");
  /// \brief Right justifies string value
  /// \tparam T string-convertible type
  /// \param value
  /// \param width output size
  /// \param fill_char
  /// \return
  template<typename T>
  static std::string rjust(const T &value, size_t width, char fill_char = ' ') {
    Str s;
    s = s << value;
    if (s.str().size() >= width)
      return s.str();
    return std::string(width - s.str().size(), fill_char) + s.str();
  }
  /// \brief Left justifies string value
  /// \tparam T string-convertible type
  /// \param value
  /// \param width output size
  /// \param fill_char
  /// \return
  template<typename T>
  static std::string ljust(const T &value, size_t width, char fill_char = ' ') {
    Str s;
    s = s << value;
    if (s.str().size() >= width)
      return s.str();
    return s.str() + std::string(width - s.str().size(), fill_char);
  }
  /// \brief Center justifies string value
  /// \tparam T string-convertible type
  /// \param value
  /// \param width output size
  /// \param fill_char
  /// \return
  template<typename T>
  static std::string cjust(const T &value, size_t width, char fill_char = ' ') {
    Str s;
    s = s << value;
    if (s.str().size() >= width)
      return s.str();
    size_t pad = (width - s.str().size()) / 2;
    return std::string(pad, fill_char) + s.str() + std::string(pad, fill_char);
  }
  /// \tparam Ts
  /// \param fmt
  /// \param args
  /// \return
  template<typename... Ts>
  static std::string format(const std::string &fmt, Ts &&... args) {
    std::stringstream s;
    std::string r;
    if constexpr(sizeof...(args) > 0) {
      format_r_(s, fmt, 0, std::forward<Ts>(args) ...);
      r = s.str();
    } else
      r = fmt;
    return r;
  }
  /// \brief Generates hexadecimal representation from number
  /// \note Calls std::hex on `i`
  /// \tparam T
  /// \param i number
  /// \param leading_zeros puts leading zeros up to the size of `T`
  /// \param zero_x puts the "0x" suffix
  /// \return
  template<typename T>
  static std::string toHex(T i, bool leading_zeros = false, bool zero_x = false) {
    std::stringstream stream;
    if (zero_x)
      stream << "0x";
    if (leading_zeros)
      stream << std::setfill('0') << std::setw(sizeof(T) * 2)
             << std::hex << i;
    else
      stream << std::hex << i;
    if (!i)
      stream << '0';
    return stream.str();
  }
  ///
  /// \param s
  /// \param patterns
  /// \return
  static std::string strip(const std::string &s, const std::string &patterns = " \t\n");
  //                                                                                                   concatenation
  /// \brief Concatenates multiple elements_ into a single string.
  /// \tparam Args
  /// \param args
  /// \return a single string of the resulting concatenation
  template<class... Args>
  static std::string concat(const Args &... args) {
    std::stringstream s;
    (s << ... << args);
    return s.str();
  }
  /// \brief Concatenate strings together separated by a separator
  /// \param v array of strings
  /// \param separator **[in | ""]**
  /// \return final string
  static std::string join(const std::vector<std::string> &v, const std::string &separator = "");
  /// \brief Concatenate elements together separates by a separator
  /// \note Element type must be able to perform << operator with `std::stringstream`
  /// \tparam T element type
  /// \param v
  /// \param separator
  /// \return
  template<typename T>
  static std::string join(const std::vector<T> &v, const std::string &separator = "") {
    bool first = true;
    std::stringstream r;
    for (const auto &s : v) {
      if (!first)
        r << separator;
      first = false;
      r << s;
    }
    return r.str();
  }
  //                                                                                                       separation
  /// \brief Splits a string into tokens separated by delimiters
  /// \param s **[in]** input string
  /// \param delimiters **[in | default = " "]** delimiters
  /// \return a vector of substrings
  static std::vector<std::string> split(const std::string &s,
                                        const std::string &delimiters = " ");
  //                                                                                                          numeric
  /// \brief Print bits in big-endian order
  /// \param n
  /// \return
  static std::string printBits(u32 n) {
    std::string r;
    for (int i = 31; i >= 0; i--)
      if ((1 << i) & n)
        r += '1';
      else
        r += '0';
    return r;
  }
  /// \brief Get ascii representation of raw bit data of `input_n`
  /// \tparam T
  /// \param input_n
  /// \param uppercase
  /// \param strip_leading_zeros
  /// \return
  template<typename T>
  static std::string binaryToHex(T input_n, bool uppercase = true, bool strip_leading_zeros = false) {
    static const char digits[] = "0123456789abcdef";
    static const char DIGITS[] = "0123456789ABCDEF";
    unsigned long long n = 0;
    std::memcpy(&n, &input_n, sizeof(T));
    std::string s;
    for (int i = sizeof(T) - 1; i >= 0; --i) {
      u8 a = n >> (8 * i + 4) & 0xf;
      u8 b = (n >> (8 * i)) & 0xf;
      if (a)
        strip_leading_zeros = false;
      if (!strip_leading_zeros)
        s += (uppercase) ? DIGITS[a] : digits[a];
      if (b)
        strip_leading_zeros = false;
      if (!strip_leading_zeros)
        s += (uppercase) ? DIGITS[b] : digits[b];
    }
    return s;
  }
  /// \brief Generates hexadecimal representation of memory address
  /// \param ptr
  /// \param digit_count
  /// \return
  static std::string addressOf(uintptr_t ptr, u32 digit_count = 8) {
    std::string s;
    // TODO: assuming little endianess
    for (i8 i = 7; i >= 0; --i) {
      auto h = binaryToHex((ptr >> (i * 8)) & 0xff, true);
      s += h.substr(h.size() - 2);
    }
    return "0x" + s.substr(s.size() - digit_count, digit_count);
  }
  /// \brief Binary representation of byte
  /// \param b
  /// \return
  static std::string byteToBinary(byte b) {
    std::string s;
    for (int i = 7; i >= 0; i--)
      s += std::to_string((b >> i) & 1);
    return s;
  }
  /// \brief Checks if string represents an integer
  /// \note Checks the pattern [+|-]?[1-9]+
  /// \param s
  /// \return
  static bool isInteger(const std::string &s);
  /// \brief Checks if string represents a number
  /// \note Checks the pattern [+|-]?([1-9]+ or .[0-9]+f? or e[1-9]+)
  /// \param s
  /// \return
  static bool isNumber(const std::string &s);
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  /// \brief Concatenate
  /// \param s
  /// \param str
  /// \return
  inline friend Str operator<<(const char *s, const Str &str) {
    return {str.str() + s};
  }
  /// \brief Concatenate
  /// \param s
  /// \param str
  /// \return
  inline friend Str operator+(const std::string &s, const Str &str) {
    std::stringstream ss;
    ss << s << str.s_;
    return {ss.str()};
  }
  //                                                                                                          boolean
  /// \brief `const char*` pointer comparison
  /// \param ss
  /// \param s
  /// \return
  inline friend bool operator==(const char *ss, const Str &s) {
    return s.str() == ss;
  }
  /// \brief Character-wise comparison
  /// \tparam T
  /// \param t
  /// \param s
  /// \return
  template<typename T>
  inline bool friend operator==(const T &t, const Str &s) {
    std::stringstream ss;
    ss << t;
    return s.str() == ss.str();
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \brief Default constructor
  Str();
  /// \brief Constructor from `std::string`
  /// \param s
  Str(std::string s);
  /// \brief Constructor from `const char*`'s contents copy
  /// \param s
  Str(const char *s);
  /// \brief Copy constructor
  /// \param other
  Str(const Str &other);
  /// \brief Move constructor
  /// \param other
  Str(Str &&other) noexcept;
  ///
  ~Str();
  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  /// \brief Get `std::string` object
  /// \return
  [[nodiscard]] inline const std::string &str() const { return s_; }
  /// \brief Get `const char*` pointer
  /// \return
  [[nodiscard]] inline const char *c_str() const { return s_.c_str(); }
  /// \brief Get the number of characters on the string.
  /// \return number of characters on the string.
  [[nodiscard]] inline size_t size() const { return s_.size(); }
  /// \brief Checks if string is empty.
  /// \return true if string size is zero.
  [[nodiscard]] inline bool empty() const { return s_.empty(); }
  /// \brief Get a sub-string view from this object
  /// \param pos position of the first character of the sub-string in str.
  /// \param len number of characters of the sub-string. If len = -1, then the size is str.size() - pos.
  /// \return const view reference of the sub-string
  Result<ConstStrView> substr(size_t pos = 0, i64 len = -1);
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  /// \brief Append arguments to this Str
  /// \note Arguments must support << operator from `std::ostringstream`
  /// \tparam Args
  /// \param args
  template<class... Args>
  void append(const Args &... args) {
    std::ostringstream s;
    (s << ... << args);
    s_ += s.str();
  }
  /// \brief Append arguments to this Str followed by a breakline
  /// \note Arguments must support << operator from `std::ostringstream`
  /// \tparam Args
  /// \param args
  template<class... Args>
  void appendLine(const Args &... args) {
    std::ostringstream s;
    (s << ... << args);
    s_ += s.str() + '\n';
  }
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  /// Copy assignment
  /// \param s
  /// \return
  Str &operator=(const Str &s) = default;
  /// \brief String of value assignment
  /// \note Argument must support << operator from `std::stringstream`
  /// \tparam T
  /// \param t
  /// \return
  template<typename T>
  Str &operator=(const T &t) {
    std::stringstream ss;
    ss << t;
    s_ = ss.str();
    return *this;
  }
  //                                                                                                       arithmetic
  /// \brief Simple concatenation with `other`
  /// \param other
  /// \return
  Str &operator+=(const Str &other) {
    s_ += other.s_;
    return *this;
  }
  /// \brief Simple concatenation with string of value
  /// \note Argument must support << operator from `std::stringstream`
  /// \tparam T
  /// \param t
  /// \return
  template<typename T>
  Str &operator+=(const T &t) {
    std::stringstream ss;
    ss << t;
    s_ += ss.str();
    return *this;
  }
  /// \brief Generates a copy appended by `t`
  /// \note Argument must support << operator from `std::stringstream`
  /// \tparam T
  /// \param t
  /// \return
  template<typename T>
  inline Str operator+(const T &t) const {
    std::stringstream ss;
    ss << s_ << t;
    return ss.str();
  }
  /// \brief Generates a copy appended by `s`
  /// \param s
  /// \return
  inline Str operator<<(const char *s) const {
    return {s_ + s};
  }
  //                                                                                                          boolean
  /// \brief Performs const char* comparison
  /// \param ss
  /// \return
  inline bool operator==(const char *ss) const {
    return s_ == ss;
  }
  /// \brief Performs character comparison with string value of `t`
  /// \note Argument must support << operator from `std::stringstream`
  template<typename T>
  inline bool operator==(const T &t) const {
    std::stringstream ss;
    ss << t;
    return s_ == ss.str();
  }
private:
  // *******************************************************************************************************************
  //                                                                                                  PRIVATE METHODS
  // *******************************************************************************************************************
  template<typename T>
  static void format_r_(std::stringstream &s, const std::string &fmt, u32 i, const T &first) {
    auto first_i = i;
    while (i + 1 < fmt.size() && !(fmt[i] == '{' && fmt[i + 1] == '}'))
      ++i;
    if (i + 1 < fmt.size()) {
      s << fmt.substr(first_i, i - first_i);
      s << first;
      s << fmt.substr(i + 2, fmt.size() - i - 2);
    } else
      s << fmt.substr(first_i, fmt.size() - first_i);
  }
  template<typename T, typename...Ts>
  static void format_r_(std::stringstream &s, const std::string &fmt, u32 i, const T &first, Ts &&... rest) {
    // iterate until first occurrence of pair {}
    auto first_i = i;
    while (i + 1 < fmt.size() && !(fmt[i] == '{' && fmt[i + 1] == '}'))
      ++i;
    if (i + 1 < fmt.size()) {
      s << fmt.substr(first_i, i - first_i);
      s << first;
      if constexpr(sizeof ...(rest) > 0)
        format_r_(s, fmt, i + 2, std::forward<Ts>(rest)...);
      else
        s << fmt.substr(i + 2, fmt.size() - i - 2);
    } else
      s << fmt.substr(first_i, fmt.size() - first_i);
  }
  // *******************************************************************************************************************
  //                                                                                                   PRIVATE FIELDS
  // *******************************************************************************************************************
  std::string s_;
};

// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
/// \brief Str support for `std::ostream`'s << operator
/// \param os
/// \param s
/// \return
inline std::ostream &operator<<(std::ostream &os, const Str &s) {
  os << s.str();
  return os;
}
/// \brief Value support for Str << operator
/// \tparam T
/// \param s
/// \param t
/// \return
template<typename T>
inline Str operator<<(const Str &s, T t) {
  std::stringstream ss;
  ss << t;
  return {s + ss.str()};
}
/// \brief `std::string` support for Str << operator
/// \tparam T
/// \param t
/// \param s
/// \return
template<typename T, std::enable_if_t<std::is_same_v<T, std::string> == false>>
inline Str operator<<(T t, const Str &s) {
  std::stringstream ss;
  ss << t;
  return {s + ss.str()};
}

}

#endif //HERMES_COMMON_STR_H

/// @}
