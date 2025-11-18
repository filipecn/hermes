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

/// \file   str.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2020-10-07
/// \brief  String utils

#pragma once

#include <hermes/core/types.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace hermes {

// *****************************************************************************
//                                                                        Str
// *****************************************************************************

/// \brief String class and set of string functions
template <typename CharType> class Str {
public:
  using StringType = std::basic_string<CharType>;
  using StringTypeIterator =
      typename std::basic_string<CharType>::const_iterator;
  using StringStreamType = std::basic_stringstream<CharType>;

  struct regex {
    using MatchResults = std::match_results<StringTypeIterator>;
    static inline constexpr char floating_point_number[] =
        "[-+]?[0-9]*\\.?[0-9]*e?[-+]?[0-9]+";
    static inline const char integer_number[] = "[-+]?[0-9]+";
    static inline const char alpha_numeric_word[] = "[a-zA-Z0-9]+";
    static inline const char c_identifier[] = "[_a-zA-Z]+[0-9a-zA-Z_]*";

    /// \brief Checks if a string s matches exactly a regular expression
    /// \param s input string
    /// \param pattern regex pattern
    /// \param flags [optional] controls how pattern is matched
    /// \return true if s matches exactly the pattern
    static bool match(const StringType &s, const StringType &pattern,
                      std::regex_constants::match_flag_type flags =
                          std::regex_constants::match_default) {
      MatchResults m;
      return std::regex_match(s, m, std::basic_regex<CharType>(pattern), flags);
    }
    /// \brief Checks if any substring of s matches a regular expression
    /// \param s input string
    /// \param pattern regex pattern
    /// \param flags [optional] controls how pattern is matched
    /// \return true if s contains the pattern
    static bool contains(const StringType &s, const StringType &pattern,
                         std::regex_constants::match_flag_type flags =
                             std::regex_constants::match_default) {
      MatchResults m;
      return std::regex_search(s, m, std::basic_regex<CharType>(pattern),
                               flags);
    }
    /// \brief Search the first substrings of s that matches the pattern
    /// \param s input string
    /// \param pattern regular expression pattern
    /// \param flags [optional] controls how pattern is matched
    /// \return std match object containing the first match
    static MatchResults search(const StringType &s, const StringType &pattern,
                               std::regex_constants::match_flag_type flags =
                                   std::regex_constants::match_default) {
      MatchResults m;
      std::regex_search(s, m, std::basic_regex<CharType>(pattern), flags);
      return m;
    }
    /// \brief Iterate over all substrings of s that match the pattern
    /// \param s input string
    /// \param pattern regular expression pattern
    /// \param callback called for each match
    /// \param flags [optional] controls how pattern is matched
    /// \return true if any match occurred
    static bool
    search(StringType s, const StringType &pattern,
           const std::function<void(const MatchResults &)> &callback,
           std::regex_constants::match_flag_type flags =
               std::regex_constants::match_default) {
      MatchResults result;
      std::basic_regex<CharType> r(pattern);
      bool found = false;
      while (std::regex_search(s, result, r, flags)) {
        callback(result);
        s = result.suffix().str();
        found = true;
      }
      return found;
    }
    /// \brief Replaces all matches of pattern in s by format
    /// \param s input string
    /// \param pattern regular expression pattern
    /// \param format replacement format
    /// \param flags [optional] controls how pattern is matched and how format
    /// is replaced
    /// \return A copy of s with all replacements
    static StringType replace(const StringType &s, const StringType &pattern,
                              const StringType &format,
                              std::regex_constants::match_flag_type flags =
                                  std::regex_constants::match_default) {
      return std::regex_replace(s, std::basic_regex<CharType>(pattern), format,
                                flags);
    }
  };

  // ***************************************************************************
  //                                                           STATIC METHODS
  // ***************************************************************************

  //                                                                   queries

  /// \brief Checks if s has prefix p.
  /// \param p prefix string
  /// \param s string
  /// \return true if p is prefix of s
  static bool isPrefix(const StringType &p, const StringType &s) {
    if (s.size() < p.size())
      return false;
    if (p.empty())
      return true;
    for (size_t i = 0; i < p.size(); ++i)
      if (s[i] != p[i])
        return false;
    return true;
  }
  //                                                                formatting

  /// \brief Abbreviates a string to fit in a string of width characters.
  /// \note If width >= string size, no abbreviation occurs.
  /// \param s input string
  /// \param width final character count
  /// \param fmt a three-character string describing the abbreviation type:
  /// ("..s", "s.s", ".s.", or "..s").
  ///         where 's' represents the input string contents and '.' the
  ///         abbreviated portion of s.
  /// \return abbreviated string
  static StringType abbreviate(const StringType &s, size_t width,
#ifdef _WIN32
                               const CharType fmt[4] = L"s.s")
#else
                               const CharType fmt[4] = "s.s")
#endif
  {
    if (!width || s.empty())
      return {};
    if (width >= s.size())
      return s;

    // case .s.
    if (fmt[0] == '.' && fmt[1] == 's' && fmt[2] == '.') {
      size_t dot_sizes_for_small_widths[] = {
          /* here the number of dots must be pair
          0  1  2  3  4  5  6  7  8  9 10 11 12*/
          0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
      size_t number_of_dots =
          width <= 12 ? dot_sizes_for_small_widths[width] : 3;
      StringType dots =
          number_of_dots > 0 ? StringType(number_of_dots, '.') : StringType();
      auto s_size = width - 2 * dots.size();
      return dots + s.substr(s.size() / 2 - s_size / 2, s_size) + dots;
    }

    // case s.s
    if (fmt[0] == 's' && fmt[1] == '.' && fmt[2] == 's') {
      if (width == 1)
        return s.substr(0, 1);
      //                                     0  1  2  3  4  5  6  7  8
      size_t dot_sizes_for_small_widths[] = {0, 0, 0, 1, 2, 1, 2, 1, 2};
      size_t number_of_dots =
          width <= 8 ? dot_sizes_for_small_widths[width] : 3;
      StringType dots =
          number_of_dots > 0 ? StringType(number_of_dots, '.') : StringType();
      size_t half_width = (width - dots.size()) / 2;
      return s.substr(0, half_width) + dots + s.substr(s.size() - half_width);
    }

    // for the remaining cases the number of dots is computed as
    auto number_of_dots = std::min(3, std::max((int)width - 1, 0));
    StringType dots =
        number_of_dots > 0 ? StringType(number_of_dots, '.') : StringType();

    // case ..s
    if (fmt[0] == '.' && fmt[1] == '.' && fmt[2] == 's')
      return dots + s.substr(s.size() - width + dots.size());

    // case s..
    if (fmt[0] == 's' && fmt[1] == '.' && fmt[2] == '.')
      return s.substr(0, width - dots.size()) + dots;

    return s;
  }
  /// \brief Right justifies string value
  /// \tparam T string-convertible type
  /// \param value
  /// \param width output size
  /// \param fill_char
  /// \return
  template <typename T>
  static StringType rjust(const T &value, size_t width,
                          CharType fill_char = ' ') {
    Str s;
    s = s << value;
    if (s.str().size() >= width)
      return s.str();
    return StringType(width - s.str().size(), fill_char) + s.str();
  }
  /// \brief Left justifies string value
  /// \tparam T string-convertible type
  /// \param value
  /// \param width output size
  /// \param fill_char
  /// \return
  template <typename T>
  static StringType ljust(const T &value, size_t width, char fill_char = ' ') {
    Str s;
    s = s << value;
    if (s.str().size() >= width)
      return s.str();
    return s.str() + StringType(width - s.str().size(), fill_char);
  }
  /// \brief Center justifies string value
  /// \tparam T string-convertible type
  /// \param value
  /// \param width output size
  /// \param fill_char
  /// \return
  template <typename T>
  static StringType cjust(const T &value, size_t width, char fill_char = ' ') {
    Str s;
    s = s << value;
    if (s.str().size() >= width)
      return s.str();
    size_t pad = (width - s.str().size()) / 2;
    return StringType(pad, fill_char) + s.str() + StringType(pad, fill_char);
  }
  /// \tparam Ts
  /// \param fmt
  /// \param args
  /// \return
  static inline StringType format() { return {}; }
  template <typename... Ts>
  static StringType format(const StringType &fmt, Ts &&...args) {
    StringStreamType s;
    StringType r;
    if constexpr (sizeof...(args) > 0) {
      s << std::vformat(fmt, std::make_format_args(args...));
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
  template <typename T>
  static StringType toHex(T i, bool leading_zeros = false,
                          bool zero_x = false) {
    StringStreamType stream;
    if (zero_x)
      stream << "0x";
    if (leading_zeros)
      stream << std::setfill('0') << std::setw(sizeof(T) * 2) << std::hex << i;
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
  static StringType strip(const StringType &s,
#ifdef _WIN32
                          const StringType &patterns = L" \t\n")
#else
                          const StringType &patterns = " \t\n")
#endif
  {
    if (s.empty())
      return s;
    int lpos = 0;
    bool found = false;
    do {
      found = false;
      for (auto p : patterns)
        if (s[lpos] == p)
          found = true;
    } while (found && static_cast<std::size_t>(++lpos) < s.size());
    i32 rpos = static_cast<i32>(s.size()) - 1;
    do {
      found = false;
      for (auto p : patterns)
        if (s[rpos] == p)
          found = true;
    } while (found && --rpos >= 0);
    return s.substr(lpos, rpos - lpos + 1);
  }

  //                                                             concatenation

  /// \brief Concatenates multiple elements_ into a single string.
  /// \tparam Args
  /// \param args
  /// \return a single string of the resulting concatenation
  template <class... Args> static StringType concat(const Args &...args) {
    StringStreamType s;
    (s << ... << args);
    return s.str();
  }
  /// \brief Concatenate strings together separated by a separator
  /// \param v array of strings
  /// \param separator **[in | ""]**
  /// \return final string
  static StringType join(const std::vector<StringType> &v,
                         const StringType &separator = {}) {
    StringType r;
    bool first = true;
    for (const auto &ss : v) {
      if (!first)
        r += separator;
      first = false;
      r += ss;
    }
    return r;
  }
  /// \brief Concatenate elements together separates by a separator
  /// \note Element type must be able to perform << operator with
  /// `StringStreamType`
  /// \tparam T element type
  /// \param v
  /// \param separator
  /// \return
  template <typename T>
  static StringType join(const std::vector<T> &v,
                         const StringType &separator = {}) {
    bool first = true;
    StringStreamType r;
    for (const auto &s : v) {
      if (!first)
        r << separator;
      first = false;
      r << s;
    }
    return r.str();
  }

  //                                                                separation

  /// \brief Splits a string into tokens separated by delimiters
  /// \param s **[in]** input string
  /// \param delimiters **[in | default = " "]** delimiters
  /// \return a vector of substrings
  static std::vector<StringType>
  split(const StringType &s, const StringType &delimiters = StringType(" ")) {
    std::vector<StringType> tokens;

    if (s.empty())
      return tokens;

    typename StringType::size_type lastPos = s.find_first_not_of(delimiters, 0);
    typename StringType::size_type pos = s.find_first_of(delimiters, lastPos);

    while (StringType::npos != pos || StringType::npos != lastPos) {
      tokens.push_back(s.substr(lastPos, pos - lastPos));
      lastPos = s.find_first_not_of(delimiters, pos);
      pos = s.find_first_of(delimiters, lastPos);
    }
    return tokens;
  }

  //                                                                   numeric

  /// \brief Print bits in big-endian order
  /// \param n
  /// \return
  static StringType printBits(u32 n) {
    StringType r;
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
  template <typename T>
  static StringType binaryToHex(T input_n, bool uppercase = true,
                                bool strip_leading_zeros = false) {
    static const char digits[] = "0123456789abcdef";
    static const char DIGITS[] = "0123456789ABCDEF";
    unsigned long long n = 0;
    std::memcpy(&n, &input_n, sizeof(T));
    StringType s;
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
  static StringType addressOf(uintptr_t ptr, u32 digit_count = 8) {
    StringType s;
    // TODO: assuming little endianess
    for (i8 i = 7; i >= 0; --i) {
      auto h = binaryToHex((ptr >> (i * 8)) & 0xff, true);
      s += h.substr(h.size() - 2);
    }
#ifdef _WIN32
    return L"0x" + s.substr(s.size() - digit_count, digit_count);
#else
    return "0x" + s.substr(s.size() - digit_count, digit_count);
#endif
  }
  /// \brief Binary representation of byte
  /// \param b
  /// \return
  static StringType byteToBinary(h_byte b) {
    StringType s;
    for (int i = 7; i >= 0; i--)
#ifdef _WIN32
      s += std::to_wstring((b >> i) & 1);
#else
      s += std::to_string((int)((b >> i) & (h_byte)1));
#endif
    return s;
  }
  /// \brief Checks if string represents an integer
  /// \note Checks the pattern [+|-]?[1-9]+
  /// \param s
  /// \return
  static bool isInteger(const StringType &s) {
    auto ss = strip(s, " \n");
    if (ss.empty())
      return false;
    size_t i = 0;
    if (!std::isdigit(ss[0])) {
      i = 1;
      if (ss.size() == 1)
        return false;
      if (ss[0] != '-' && ss[0] != '+')
        return false;
    }
    for (; i < ss.size(); ++i)
      if (!std::isdigit(ss[i]))
        return false;
    return true;
  }
  /// \brief Checks if string represents a number
  /// \note Checks the pattern [+|-]?([1-9]+ or .[0-9]+f? or e[1-9]+)
  /// \param s
  /// \return
  static bool isNumber(const StringType &s) {
    auto ss = strip(s, " \n");
    if (ss.empty())
      return false;
    auto p = split(ss, "e");
    if (p.size() > 2)
      return false;
    // check floating piece
    size_t i = 0;
    int point_count = 0;
    if (!std::isdigit(p[0][0])) {
      i = 1;
      if (p[0].size() == 1)
        return false;
      if (p[0][0] != '-' && p[0][0] != '+' && s[0] != '.')
        return false;
      if (p[0][0] == '.')
        point_count++;
    }
    for (; i < p[0].size(); ++i) {
      if (p[0][i] == '.') {
        point_count++;
        if (point_count > 1)
          return false;
        continue;
      }
      if (p[0][i] == 'f' && i != p[0].size() - 1)
        return false;
      if (p[0][i] == 'f')
        continue;
      if (!std::isdigit(p[0][i]))
        return false;
    }
    if (p.size() > 1)
      return isInteger(p[1]);
    return true;
  }

  // ***************************************************************************
  //                                                         FRIEND FUNCTIONS
  // ***************************************************************************

  /// \brief Concatenate
  /// \param s
  /// \param str
  /// \return
  inline friend Str operator<<(const CharType *s, const Str &str) {
    return {str.str() + s};
  }
  /// \brief Concatenate
  /// \param s
  /// \param str
  /// \return
  inline friend Str operator+(const StringType &s, const Str &str) {
#ifdef _WIN32
    std::wstringstream ss;
#else
    StringStreamType ss;
#endif
    ss << s << str.s_;
    return {ss.str()};
  }

  //                                                                   boolean

  /// \brief `const char*` pointer comparison
  /// \param ss
  /// \param s
  /// \return
  inline friend bool operator==(const CharType *ss, const Str &s) {
    return s.str() == ss;
  }
  /// \brief Character-wise comparison
  /// \tparam T
  /// \param t
  /// \param s
  /// \return
  template <typename T>
  inline bool friend operator==(const T &t, const Str &s) {
#ifdef _WIN32
    std::wstringstream ss;
#else
    StringStreamType ss;
#endif
    ss << t;
    return s.str() == ss.str();
  }

  // ***************************************************************************
  //                                                             CONSTRUCTORS
  // ***************************************************************************

  /// \brief Default constructor
  Str() = default;
  /// \brief Constructor from `std::string`
  /// \param s
  Str(StringType s) : s_{std::move(s)} {}
  /// \brief Constructor from `const char*`'s contents copy
  /// \param s
  Str(const CharType *s) : s_{s} {}
  /// \brief Copy constructor
  /// \param other
  Str(const Str &other) = default;
  /// \brief Move constructor
  /// \param other
  Str(Str &&other) HERMES_NOEXCEPT : s_{std::move(other.s_)} {}
  ///
  ~Str() = default;

  // ***************************************************************************
  //                                                                   ACCESS
  // ***************************************************************************

  /// \brief Get `std::string` object
  /// \return
  HERMES_NODISCARD inline const StringType &str() const { return s_; }
  /// \brief Get `const char*` pointer
  /// \return
  HERMES_NODISCARD inline const CharType *c_str() const { return s_.c_str(); }
  /// \brief Get the number of characters on the string.
  /// \return number of characters on the string.
  HERMES_NODISCARD inline size_t size() const { return s_.size(); }
  /// \brief Checks if string is empty.
  /// \return true if string size is zero.
  HERMES_NODISCARD inline bool empty() const { return s_.empty(); }
  /// \brief Get a sub-string view from this object
  /// \param pos position of the first character of the sub-string in str.
  /// \param len number of characters of the sub-string. If len = -1, then the
  /// size is str.size() - pos.
  /// \return const view reference of the sub-string
  // Result<ConstStrView> substr(size_t pos = 0, i64 len = -1);

  // ***************************************************************************
  //                                                                  METHODS
  // ***************************************************************************

  /// \brief Append arguments to this Str
  /// \note Arguments must support << operator from `std::ostringstream`
  /// \tparam Args
  /// \param args
  template <class... Args> void append(const Args &...args) {
    std::basic_ostringstream<CharType> s;
    (s << ... << args);
    s_ += s.str();
  }
  /// \brief Append arguments to this Str followed by a breakline
  /// \note Arguments must support << operator from `std::ostringstream`
  /// \tparam Args
  /// \param args
  template <class... Args> void appendLine(const Args &...args) {
    std::basic_stringstream<CharType> s;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    (s << ... << args);
#pragma GCC diagnostic pop
    s << '\n';
    s_ += s.str();
  }

  // ***************************************************************************
  //                                                                OPERATORS
  // ***************************************************************************

  //                                                                conversion

  operator std::filesystem::path() const { return std::filesystem::path(s_); }

  //                                                                assignment

  /// Copy assignment
  /// \param s
  /// \return
  Str &operator=(const Str &s) = default;
  /// \brief String of value assignment
  /// \note Argument must support << operator from `StringStreamType`
  /// \tparam T
  /// \param t
  /// \return
  template <typename T> Str &operator=(const T &t) {
    std::basic_stringstream<CharType> ss;
    ss << t;
    s_ = ss.str();
    return *this;
  }

  //                                                                arithmetic

  /// \brief Simple concatenation with `other`
  /// \param other
  /// \return
  Str &operator+=(const Str &other) {
    s_ += other.s_;
    return *this;
  }
  /// \brief Simple concatenation with string of value
  /// \note Argument must support << operator from `StringStreamType`
  /// \tparam T
  /// \param t
  /// \return
  template <typename T> Str &operator+=(const T &t) {
    std::basic_stringstream<CharType> ss;
    ss << t;
    s_ += ss.str();
    return *this;
  }
  /// \brief Generates a copy appended by `t`
  /// \note Argument must support << operator from `StringStreamType`
  /// \tparam T
  /// \param t
  /// \return
  template <typename T> inline Str operator+(const T &t) const {
    std::basic_stringstream<CharType> ss;
    ss << s_ << t;
    return ss.str();
  }
  /// \brief Generates a copy appended by `s`
  /// \param s
  /// \return
  inline Str operator<<(const CharType *s) const { return {s_ + s}; }

  //                                                                   boolean

  /// \brief Performs const char* comparison
  /// \param ss
  /// \return
  inline bool operator==(const CharType *ss) const { return s_ == ss; }
  /// \brief Performs character comparison with string value of `t`
  /// \note Argument must support << operator from `StringStreamType`
  template <typename T> inline bool operator==(const T &t) const {
    std::basic_stringstream<CharType> ss;
    ss << t;
    return s_ == ss.str();
  }

private:
  StringType s_;
};

// *****************************************************************************
//                                                                         IO
// *****************************************************************************

/// \brief Str support for `std::ostream`'s << operator
/// \param os
/// \param s
/// \return
template <typename CharType>
inline std::ostream &operator<<(std::ostream &os, const Str<CharType> &s) {
  os << s.str();
  return os;
}
/// \brief Value support for Str << operator
/// \tparam T
/// \param s
/// \param t
/// \return
template <typename CharType, typename T>
inline Str<CharType> operator<<(const Str<CharType> &s, T t) {
  std::basic_stringstream<CharType> ss;
  ss << t;
  return {s + ss.str()};
}
/// \brief `std::string` support for Str << operator
/// \tparam T
/// \param t
/// \param s
/// \return
template <typename T, std::enable_if_t<std::is_same_v<T, std::string> == false>>
inline Str<char> operator<<(T t, const Str<char> &s) {
  std::basic_stringstream<char> ss;
  ss << t;
  return {s + ss.str()};
}

using cstr = Str<char>;
using wstr = Str<wchar_t>;

} // namespace hermes
