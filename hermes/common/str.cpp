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
///\file str.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2020-10-07
///
///\brief String utils

#include <hermes/common/str.h>
#include <hermes/common/debug.h>
#include <iostream>
#include <utility>

namespace hermes {

const char Str::regex::floating_point_number[] = "[-+]?[0-9]*\\.?[0-9]*e?[-+]?[0-9]+";
const char Str::regex::integer_number[] = "[-+]?[0-9]+";
const char Str::regex::alpha_numeric_word[] = "[a-zA-Z0-9]+";
const char Str::regex::c_identifier[] = "[_a-zA-Z]+[0-9a-zA-Z_]*";

bool Str::regex::match(const std::string &s, const std::string &pattern, std::regex_constants::match_flag_type flags) {
  std::smatch m;
  return std::regex_match(s, m, std::regex(pattern), flags);
}

bool Str::regex::contains(const std::string &s,
                          const std::string &pattern,
                          std::regex_constants::match_flag_type flags) {
  std::smatch m;
  return std::regex_search(s, m, std::regex(pattern), flags);
}

std::smatch Str::regex::search(const std::string &s,
                               const std::string &pattern,
                               std::regex_constants::match_flag_type flags) {
  std::smatch result;
  std::regex_search(s, result, std::regex(pattern), flags);
  return result;
}

bool Str::regex::search(std::string s,
                        const std::string &pattern,
                        const std::function<void(const std::smatch &)> &callback,
                        std::regex_constants::match_flag_type flags) {
  std::smatch result;
  std::regex r(pattern);
  bool found = false;
  while (std::regex_search(s, result, r, flags)) {
    callback(result);
    s = result.suffix().str();
    found = true;
  }
  return found;
}

std::string Str::regex::replace(const std::string &s,
                                const std::string &pattern,
                                const std::string &format,
                                std::regex_constants::match_flag_type flags) {
  return std::regex_replace(s, std::regex(pattern), format, flags);
}

bool Str::isPrefix(const std::string &p, const std::string &s) {
  if(s.size() < p.size())
    return false;
  if(p.empty())
    return true;
  for(size_t i = 0; i < p.size(); ++i)
    if(s[i] != p[i])
      return false;
  return true;
}

std::string Str::abbreviate(const std::string &s, size_t width, const char fmt[4]) {
  if (!width || s.empty())
    return "";
  if (width >= s.size())
    return s;

  // case .s.
  if (fmt[0] == '.' && fmt[1] == 's' && fmt[2] == '.') {
    // here the number of dots must be pair
    //                                     0  1  2  3  4  5  6  7  8  9 10 11 12
    size_t dot_sizes_for_small_widths[] = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
    size_t number_of_dots = width <= 12 ? dot_sizes_for_small_widths[width] : 3;
    std::string dots = number_of_dots > 0 ? std::string(number_of_dots, '.') : "";
    auto s_size = width - 2 * dots.size();
    return dots + s.substr(s.size() / 2 - s_size / 2, s_size) + dots;
  }

  // case s.s
  if (fmt[0] == 's' && fmt[1] == '.' && fmt[2] == 's') {
    if (width == 1)
      return s.substr(0, 1);
    //                                     0  1  2  3  4  5  6  7  8
    size_t dot_sizes_for_small_widths[] = {0, 0, 0, 1, 2, 1, 2, 1, 2};
    size_t number_of_dots = width <= 8 ? dot_sizes_for_small_widths[width] : 3;
    std::string dots = number_of_dots > 0 ? std::string(number_of_dots, '.') : "";
    size_t half_width = (width - dots.size()) / 2;
    return s.substr(0, half_width) + dots + s.substr(s.size() - half_width);
  }

  // for the remaining cases the number of dots is computed as
  auto number_of_dots = std::min(3, std::max((int) width - 1, 0));
  std::string dots = number_of_dots > 0 ? std::string(number_of_dots, '.') : "";

  // case ..s
  if (fmt[0] == '.' && fmt[1] == '.' && fmt[2] == 's')
    return dots + s.substr(s.size() - width + dots.size());

  // case s..
  if (fmt[0] == 's' && fmt[1] == '.' && fmt[2] == '.')
    return s.substr(0, width - dots.size()) + dots;

  return s;
}

std::string Str::join(const std::vector<std::string> &s, const std::string &separator) {
  std::string r;
  bool first = true;
  for (const auto &ss : s) {
    if (!first)
      r += separator;
    first = false;
    r += ss;
  }
  return r;
}

std::string Str::strip(const std::string &s, const std::string &patterns) {
  if (s.empty())
    return s;
  int lpos = 0;
  bool found = false;
  do {
    found = false;
    for (auto p : patterns)
      if (s[lpos] == p)
        found = true;
  } while (found && ++lpos < s.size());
  int rpos = s.size() - 1;
  do {
    found = false;
    for (auto p : patterns)
      if (s[rpos] == p)
        found = true;
  } while (found && --rpos >= 0);
  return s.substr(lpos, rpos - lpos + 1);
}

std::vector<std::string> Str::split(const std::string &s, const std::string &delimiters) {
  std::vector<std::string> tokens;

  if (s.empty())
    return tokens;

  std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
  std::string::size_type pos = s.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delimiters, pos);
    pos = s.find_first_of(delimiters, lastPos);
  }
  return tokens;
}

bool Str::isInteger(const std::string &s) {
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

bool Str::isNumber(const std::string &s) {
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

Str::Str() = default;

Str::Str(std::string s) : s_{std::move(s)} {}

Str::Str(const char *s) : s_{s} {}

Str::Str(Str &&other) noexcept: s_{std::move(other.s_)} {}

Str::Str(const Str &other) = default;

Str::~Str() = default;

Result<ConstStrView> Str::substr(size_t pos, i64 len) {
  return ConstStrView::from(s_, pos, len);
}

}

