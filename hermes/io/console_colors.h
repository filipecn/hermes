/* Copyright (c) 2021, FilipeCN.
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

/// \file console_colors.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date 2021-03-19
/// \brief Set of 256-terminal supported color codes

#pragma once

#include <hermes/core/types.h>

#include <string>
#include <thread>

namespace hermes::colors::console {

/// Set of 256-terminal color codes
/// \note Extracted from
/// https://misc.flogisoft.com/bash/tip_colors_and_formatting

// Set

constexpr char bold[5] = "\33[1m";
constexpr char dim[5] = "\33[2m";
constexpr char underlined[5] = "\33[4m";
constexpr char blink[5] = "\33[5m";
constexpr char inverted[5] = "\33[7m";
constexpr char hidden[5] = "\33[8m";

// RESET

constexpr char reset[5] = "\33[0m";
constexpr char reset_bold[6] = "\33[21m";
constexpr char reset_dim[6] = "\33[22m";
constexpr char reset_underlined[6] = "\33[24m";
constexpr char reset_blink[6] = "\33[25m";
constexpr char reset_inverted[6] = "\33[27m";
constexpr char reset_hidden[6] = "\33[28m";

// 8/16 Colors

constexpr char default_color[6] = "\33[39m";
constexpr char black[6] = "\33[30m";
constexpr char red[6] = "\33[31m";
constexpr char green[6] = "\33[32m";
constexpr char yellow[6] = "\33[33m";
constexpr char blue[6] = "\33[34m";
constexpr char magenta[6] = "\33[35m";
constexpr char cyan[6] = "\33[36m";
constexpr char light_gray[6] = "\33[37m";
constexpr char dark_gray[6] = "\33[90m";
constexpr char light_red[6] = "\33[91m";
constexpr char light_green[6] = "\33[92m";
constexpr char light_yellow[6] = "\33[93m";
constexpr char light_blue[6] = "\33[94m";
constexpr char light_magenta[6] = "\33[95m";
constexpr char light_cyan[6] = "\33[96m";
constexpr char white[6] = "\33[97m";

constexpr char background_default_color[6] = "\33[49m";
constexpr char background_black[6] = "\33[40m";
constexpr char background_red[6] = "\33[41m";
constexpr char background_green[6] = "\33[42m";
constexpr char background_yellow[6] = "\33[43m";
constexpr char background_blue[6] = "\33[44m";
constexpr char background_magenta[6] = "\33[45m";
constexpr char background_cyan[6] = "\33[46m";
constexpr char background_light_gray[6] = "\33[47m";
constexpr char background_dark_gray[7] = "\33[100m";
constexpr char background_light_red[7] = "\33[101m";
constexpr char background_light_green[7] = "\33[102m";
constexpr char background_light_yellow[7] = "\33[103m";
constexpr char background_light_blue[7] = "\33[104m";
constexpr char background_light_magenta[7] = "\33[105m";
constexpr char background_light_cyan[7] = "\33[106m";
constexpr char background_white[7] = "\33[107m";

/// \brief Get 88/256 color code
/// \param color_number
/// \return
inline std::string color(u8 color_number) {
  return std::string("\e[38;5;") + std::to_string(color_number) + "m";
}
/// \brief Get 88/256 background color code
/// \param color_number
/// \return
inline std::string background_color(u8 color_number) {
  return std::string("\e[48;5;") + std::to_string(color_number) + "m";
}
/// \brief Combine two color codes
/// \param a
/// \param b
/// \return
inline std::string combine(const std::string &a, const std::string &b) {
  return "\e[" + a.substr(2, a.size() - 3) + ";" + b.substr(2, b.size() - 3) +
         "m";
}

template <typename T> std::string numberColor(T n) {
  return color(static_cast<u8>(n));
}

inline std::string threadColor(std::thread::id thread_id) {
  return numberColor(std::hash<std::thread::id>()(thread_id));
}

inline std::string random() {
  static u8 next = 0;
  next += 13;
  return color(next);
}

} // namespace hermes::colors::console
