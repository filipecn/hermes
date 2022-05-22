/// Copyright (c) 2021, FilipeCN.
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
///\file console_colors.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-03-19
///
///\brief Set of 256-terminal supported color codes
///
///\ingroup logging
///\addtogroup logging
/// @{

#ifndef HERMES_LOG_CONSOLE_COLORS_H
#define HERMES_LOG_CONSOLE_COLORS_H

#include <hermes/common/defs.h>
#include <string>
#include <iostream>

namespace hermes {

/// \brief Set of 256-terminal color codes
/// \note Extracted from https://misc.flogisoft.com/bash/tip_colors_and_formatting
class ConsoleColors {
public:
  // SET
  static char bold[5];                        //!< "\e[1m"
  static char dim[5];                         //!< "\e[2m"
  static char underlined[5];                  //!< "\e[4m"
  static char blink[5];                       //!< "\e[5m"
  static char inverted[5];                    //!< "\e[7m"
  static char hidden[5];                      //!< "\e[8m"
  // RESET
  static char reset[5];                       //!< "\e[0m"
  static char reset_bold[6];                  //!< "\e[21m"
  static char reset_dim[6];                   //!< "\e[22m"
  static char reset_underlined[6];            //!< "\e[24m"
  static char reset_blink[6];                 //!< "\e[25m"
  static char reset_inverted[6];              //!< "\e[27m"
  static char reset_hidden[6];                //!< "\e[28m"
  // 8/16 Colors
  static char default_color[6];               //!< "\e[39m"
  static char black[6];                       //!< "\e[30m"
  static char red[6];                         //!< "\e[31m"
  static char green[6];                       //!< "\e[32m"
  static char yellow[6];                      //!< "\e[33m"
  static char blue[6];                        //!< "\e[34m"
  static char magenta[6];                     //!< "\e[35m"
  static char cyan[6];                        //!< "\e[36m"
  static char light_gray[6];                  //!< "\e[37m"
  static char dark_gray[6];                   //!< "\e[90m"
  static char light_red[6];                   //!< "\e[91m"
  static char light_green[6];                 //!< "\e[92m"
  static char light_yellow[6];                //!< "\e[93m"
  static char light_blue[6];                  //!< "\e[94m"
  static char light_magenta[6];               //!< "\e[95m"
  static char light_cyan[6];                  //!< "\e[96m"
  static char white[6];                       //!< "\e[97m"
  static char background_default_color[6];    //!< "\e[49m"
  static char background_black[6];            //!< "\e[40m"
  static char background_red[6];              //!< "\e[41m"
  static char background_green[6];            //!< "\e[42m"
  static char background_yellow[6];           //!< "\e[43m"
  static char background_blue[6];             //!< "\e[44m"
  static char background_magenta[6];          //!< "\e[45m"
  static char background_cyan[6];             //!< "\e[46m"
  static char background_light_gray[6];       //!< "\e[47m"
  static char background_dark_gray[7];        //!< "\e[100m"
  static char background_light_red[7];        //!< "\e[101m"
  static char background_light_green[7];      //!< "\e[102m"
  static char background_light_yellow[7];     //!< "\e[103m"
  static char background_light_blue[7];       //!< "\e[104m"
  static char background_light_magenta[7];    //!< "\e[105m"
  static char background_light_cyan[7];       //!< "\e[106m"
  static char background_white[7];            //!< "\e[107m"

  /// \brief Get 88/256 color code
  /// \param color_number
  /// \return
  inline static std::string color(u8 color_number) {
    return std::string("\e[38;5;") + std::to_string(color_number) + "m";
  }
  /// \brief Get 88/256 background color code
  /// \param color_number
  /// \return
  inline static std::string background_color(u8 color_number) {
    return std::string("\e[48;5;") + std::to_string(color_number) + "m";
  }
  /// \brief Combine two color codes
  /// \param a
  /// \param b
  /// \return
  inline static std::string combine(const std::string &a, const std::string &b) {
    return "\e[" + a.substr(2, a.size() - 3) + ";" + b.substr(2, b.size() - 3) + "m";
  }
};

}

#endif //HERMES_LOG_CONSOLE_COLORS_H

/// @}
