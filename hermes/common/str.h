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

#ifndef HERMES_COMMON_STR_H
#define HERMES_COMMON_STR_H

#include <string>
#include <sstream>
#include <vector>
#include <regex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <hermes/common/defs.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                                Str
// *********************************************************************************************************************
class Str {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  //                                                                                                       formatting
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
  //                                                                                                   concatenation
  /// Concatenates multiple elements_ into a single string.
  /// \tparam Args
  /// \param args
  /// \return a single string of the resulting concatenation
  template<class... Args>
  static std::string concat(const Args &... args) {
    std::stringstream s;
    (s << ... << args);
    return s.str();
  }
  /// Concatenate strings together separated by a separator
  /// \param s array of strings
  /// \param separator **[in | ""]**
  /// \return final string
  static std::string join(const std::vector<std::string> &v, const std::string &separator = "");
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
  /// Splits a string into tokens separated by delimiters
  /// \param s **[in]** input string
  /// \param delimiters **[in | default = " "]** delimiters
  /// \return a vector of substrings
  static std::vector<std::string> split(const std::string &s,
                                        const std::string &delimiters = " ");
  //                                                                                                            regex
  /// Checks if a string s matches exactly a regular expression
  /// \param s input string
  /// \param pattern regex pattern
  /// \param flags [optional] controls how pattern is matched
  /// \return true if s matches exactly the pattern
  static bool match_r(const std::string &s, const std::string &pattern,
                      std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  /// Checks if any substring of s matches a regular expression
  /// \param s input string
  /// \param pattern regex pattern
  /// \param flags [optional] controls how pattern is matched
  /// \return true if s contains the pattern
  static bool contains_r(const std::string &s, const std::string &pattern,
                         std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  /// Search the first substrings of s that matches the pattern
  /// \param s input string
  /// \param pattern regular expression pattern
  /// \param flags [optional] controls how pattern is matched
  /// \return std match object containing the first match
  static std::smatch search_r(const std::string &s, const std::string &pattern,
                              std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  /// Iterate over all substrings of s that match the pattern
  /// \param s input string
  /// \param pattern regular expression pattern
  /// \param callback called for each match
  /// \param flags [optional] controls how pattern is matched
  /// \return true if any match occurred
  static bool search_r(std::string s,
                       const std::string &pattern,
                       const std::function<void(const std::smatch &)> &callback,
                       std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  /// Replaces all matches of pattern in s by format
  /// \param s input string
  /// \param pattern regular expression pattern
  /// \param format replacement format
  /// \param flags [optional] controls how pattern is matched and how format is replaced
  /// \return A copy of s with all replacements
  static std::string replace_r(const std::string &s, const std::string &pattern, const std::string &format,
                               std::regex_constants::match_flag_type flags = std::regex_constants::match_default);
  //                                                                                                          numeric
  /// \param n
  static std::string printBits(u32 n) {
    std::string r;
    for (int i = 31; i >= 0; i--)
      if ((1 << i) & n)
        r += '1';
      else
        r += '0';
    return r;
  }
  ///
  /// \tparam T
  /// \param n
  /// \param uppercase
  /// \param strip_leading_zeros
  /// \return
  template<typename T>
  static std::string binaryToHex(T n, bool uppercase = true, bool strip_leading_zeros = false) {
    static const char digits[] = "0123456789abcdef";
    static const char DIGITS[] = "0123456789ABCDEF";
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
  ///
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
  ///
  /// \param b
  /// \return
  static std::string byteToBinary(byte b) {
    std::string s;
    for (int i = 7; i >= 0; i--)
      s += std::to_string((b >> i) & 1);
    return s;
  }
  // *******************************************************************************************************************
  //                                                                                                 FRIEND FUNCTIONS
  // *******************************************************************************************************************
  inline friend Str operator<<(const char *s, const Str &str) {
    return {str.str() + s};
  }
  inline friend Str operator+(const std::string &s, const Str &str) {
    std::stringstream ss;
    ss << s << str.s_;
    return {ss.str()};
  }
  //                                                                                                          boolean
  inline friend bool operator==(const char *ss, const Str &s) {
    return s.str() == ss;
  }
  template<typename T>
  inline bool friend operator==(const T &t, const Str &s) {
    std::stringstream ss;
    ss << t;
    return s.str() == ss.str();
  }
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param s
  Str();
  Str(std::string s);
  Str(const char *s);
  Str(const Str &other);
  Str(Str &&other) noexcept;
  ~Str();
  // *******************************************************************************************************************
  //                                                                                                           ACCESS
  // *******************************************************************************************************************
  [[nodiscard]] inline const std::string &str() const { return s_; }
  [[nodiscard]] inline const char *c_str() const { return s_.c_str(); }
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  template<class... Args>
  void append(const Args &... args) {
    std::ostringstream s;
    (s << ... << args);
    s_ += s.str();
  }
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
  Str &operator=(const Str &s) = default;
  template<typename T>
  Str &operator=(const T &t) {
    std::stringstream ss;
    ss << t;
    s_ = ss.str();
    return *this;
  }
  //                                                                                                       arithmetic
  Str &operator+=(const Str &other) {
    s_ += other.s_;
    return *this;
  }
  template<typename T>
  Str &operator+=(const T &t) {
    std::stringstream ss;
    ss << t;
    s_ += ss.str();
    return *this;
  }
  template<typename T>
  inline Str operator+(const T &t) const {
    std::stringstream ss;
    ss << s_ << t;
    return ss.str();
  }
  inline Str operator<<(const char *s) const {
    return {s_ + s};
  }
  //                                                                                                          boolean
  inline bool operator==(const char *ss) const {
    return s_ == ss;
  }
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
inline std::ostream &operator<<(std::ostream &os, const Str &s) {
  os << s.str();
  return os;
}
template<typename T>
inline Str operator<<(const Str &s, T t) {
  std::stringstream ss;
  ss << t;
  return {s + ss.str()};
}
template<typename T, std::enable_if_t<std::is_same_v<T, std::string> == false>>
inline Str operator<<(T t, const Str &s) {
  std::stringstream ss;
  ss << t;
  return {s + ss.str()};
}

}

#endif //HERMES_COMMON_STR_H
