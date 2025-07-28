/* Copyright (c) 2022, FilipeCN.
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

/// \file   argb_colors.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2022-01-06
/// \note   This file was inspired on Sergey Yagovtsev's Easy profiler source
///         code
/// \note
/// https://github.com/yse/easy_profiler/blob/develop/easy_profiler_core/include/easy/details/profiler_colors.h

#pragma once

#include <hermes/core/types.h>

namespace hermes::colors::argb {

inline u32 constexpr argb2rgba(u32 argb) { return (argb >> 24) | (argb << 8); }

inline u32 constexpr fromRGBA(u8 red, u8 green, u8 blue, u8 alpha = 0xff) {
  return (static_cast<u32>(alpha) << 24) | (static_cast<u32>(red) << 16) |
         (static_cast<u32>(green) << 8) | static_cast<u32>(blue);
}

// Google material Design colors
// See https://material.google.com/style/color.html

constexpr u32 red_50 = 0xffffebee;
constexpr u32 red_100 = 0xffffcdd2;
constexpr u32 red_200 = 0xffef9a9a;
constexpr u32 red_300 = 0xffe57373;
constexpr u32 red_400 = 0xffef5350;
constexpr u32 red_500 = 0xfff44336;
constexpr u32 red_600 = 0xffe53935;
constexpr u32 red_700 = 0xffd32f2f;
constexpr u32 red_800 = 0xffc62828;
constexpr u32 red_900 = 0xffb71c1c;
constexpr u32 red_A100 = 0xffff8a80;
constexpr u32 red_A200 = 0xffff5252;
constexpr u32 red_A400 = 0xffff1744;
constexpr u32 red_A700 = 0xffd50000;

constexpr u32 pink_50 = 0xfffce4ec;
constexpr u32 pink_100 = 0xfff8bbd0;
constexpr u32 pink_200 = 0xfff48fb1;
constexpr u32 pink_300 = 0xfff06292;
constexpr u32 pink_400 = 0xffec407a;
constexpr u32 pink_500 = 0xffe91e63;
constexpr u32 pink_600 = 0xffd81b60;
constexpr u32 pink_700 = 0xffc2185b;
constexpr u32 pink_800 = 0xffad1457;
constexpr u32 pink_900 = 0xff880e4f;
constexpr u32 pink_A100 = 0xffff80ab;
constexpr u32 pink_A200 = 0xffff4081;
constexpr u32 pink_A400 = 0xfff50057;
constexpr u32 pink_A700 = 0xffc51162;

constexpr u32 purple50 = 0xfff3e5f5;
constexpr u32 purple100 = 0xffe1bee7;
constexpr u32 purple200 = 0xffce93d8;
constexpr u32 purple300 = 0xffba68c8;
constexpr u32 purple400 = 0xffab47bc;
constexpr u32 purple500 = 0xff9c27b0;
constexpr u32 purple600 = 0xff8e24aa;
constexpr u32 purple700 = 0xff7b1fa2;
constexpr u32 purple800 = 0xff6a1b9a;
constexpr u32 purple900 = 0xff4a148c;
constexpr u32 purpleA100 = 0xffea80fc;
constexpr u32 purpleA200 = 0xffe040fb;
constexpr u32 purpleA400 = 0xffd500f9;
constexpr u32 purpleA700 = 0xffaa00ff;

constexpr u32 deep_purple_50 = 0xffede7f6;
constexpr u32 deep_purple_100 = 0xffd1c4e9;
constexpr u32 deep_purple_200 = 0xffb39ddb;
constexpr u32 deep_purple_300 = 0xff9575cd;
constexpr u32 deep_purple_400 = 0xff7e57c2;
constexpr u32 deep_purple_500 = 0xff673ab7;
constexpr u32 deep_purple_600 = 0xff5e35b1;
constexpr u32 deep_purple_700 = 0xff512da8;
constexpr u32 deep_purple_800 = 0xff4527a0;
constexpr u32 deep_purple_900 = 0xff311b92;
constexpr u32 deep_purple_A100 = 0xffb388ff;
constexpr u32 deep_purple_A200 = 0xff7c4dff;
constexpr u32 deep_purple_A400 = 0xff651fff;
constexpr u32 deep_purple_A700 = 0xff6200ea;

constexpr u32 Indigo50 = 0xffe8eaf6;
constexpr u32 Indigo100 = 0xffc5cae9;
constexpr u32 Indigo200 = 0xff9fa8da;
constexpr u32 Indigo300 = 0xff7986cb;
constexpr u32 Indigo400 = 0xff5c6bc0;
constexpr u32 Indigo500 = 0xff3f51b5;
constexpr u32 Indigo600 = 0xff3949ab;
constexpr u32 Indigo700 = 0xff303f9f;
constexpr u32 Indigo800 = 0xff283593;
constexpr u32 Indigo900 = 0xff1a237e;
constexpr u32 IndigoA100 = 0xff8c9eff;
constexpr u32 IndigoA200 = 0xff536dfe;
constexpr u32 IndigoA400 = 0xff3d5afe;
constexpr u32 IndigoA700 = 0xff304ffe;

constexpr u32 blue_50 = 0xffe3f2fd;
constexpr u32 blue_100 = 0xffbbdefb;
constexpr u32 blue_200 = 0xff90caf9;
constexpr u32 blue_300 = 0xff64b5f6;
constexpr u32 blue_400 = 0xff42a5f5;
constexpr u32 blue_500 = 0xff2196f3;
constexpr u32 blue_600 = 0xff1e88e5;
constexpr u32 blue_700 = 0xff1976d2;
constexpr u32 blue_800 = 0xff1565c0;
constexpr u32 blue_900 = 0xff0d47a1;
constexpr u32 blue_A100 = 0xff82b1ff;
constexpr u32 blue_A200 = 0xff448aff;
constexpr u32 blue_A400 = 0xff2979ff;
constexpr u32 blue_A700 = 0xff2962ff;

constexpr u32 light_blue_50 = 0xffe1f5fe;
constexpr u32 light_blue_100 = 0xffb3e5fc;
constexpr u32 light_blue_200 = 0xff81d4fa;
constexpr u32 light_blue_300 = 0xff4fc3f7;
constexpr u32 light_blue_400 = 0xff29b6f6;
constexpr u32 light_blue_500 = 0xff03a9f4;
constexpr u32 light_blue_600 = 0xff039be5;
constexpr u32 light_blue_700 = 0xff0288d1;
constexpr u32 light_blue_800 = 0xff0277bd;
constexpr u32 light_blue_900 = 0xff01579b;
constexpr u32 light_blue_A100 = 0xff80d8ff;
constexpr u32 light_blue_A200 = 0xff40c4ff;
constexpr u32 light_blue_A400 = 0xff00b0ff;
constexpr u32 light_blue_A700 = 0xff0091ea;

constexpr u32 cyan_50 = 0xffe0f7fa;
constexpr u32 cyan_100 = 0xffb2ebf2;
constexpr u32 cyan_200 = 0xff80deea;
constexpr u32 cyan_300 = 0xff4dd0e1;
constexpr u32 cyan_400 = 0xff26c6da;
constexpr u32 cyan_500 = 0xff00bcd4;
constexpr u32 cyan_600 = 0xff00acc1;
constexpr u32 cyan_700 = 0xff0097a7;
constexpr u32 cyan_800 = 0xff00838f;
constexpr u32 cyan_900 = 0xff006064;
constexpr u32 cyan_A100 = 0xff84ffff;
constexpr u32 cyan_A200 = 0xff18ffff;
constexpr u32 cyan_A400 = 0xff00e5ff;
constexpr u32 cyan_A700 = 0xff00b8d4;

constexpr u32 teal_50 = 0xffe0f2f1;
constexpr u32 teal_100 = 0xffb2dfdb;
constexpr u32 teal_200 = 0xff80cbc4;
constexpr u32 teal_300 = 0xff4db6ac;
constexpr u32 teal_400 = 0xff26a69a;
constexpr u32 teal_500 = 0xff009688;
constexpr u32 teal_600 = 0xff00897b;
constexpr u32 teal_700 = 0xff00796b;
constexpr u32 teal_800 = 0xff00695c;
constexpr u32 teal_900 = 0xff004d40;
constexpr u32 teal_A100 = 0xffa7ffeb;
constexpr u32 teal_A200 = 0xff64ffda;
constexpr u32 teal_A400 = 0xff1de9b6;
constexpr u32 teal_A700 = 0xff00bfa5;

constexpr u32 green_50 = 0xffe8f5e9;
constexpr u32 green_100 = 0xffc8e6c9;
constexpr u32 green_200 = 0xffa5d6a7;
constexpr u32 green_300 = 0xff81c784;
constexpr u32 green_400 = 0xff66bb6a;
constexpr u32 green_500 = 0xff4caf50;
constexpr u32 green_600 = 0xff43a047;
constexpr u32 green_700 = 0xff388e3c;
constexpr u32 green_800 = 0xff2e7d32;
constexpr u32 green_900 = 0xff1b5e20;
constexpr u32 green_A100 = 0xffb9f6ca;
constexpr u32 green_A200 = 0xff69f0ae;
constexpr u32 green_A400 = 0xff00e676;
constexpr u32 green_A700 = 0xff00c853;

constexpr u32 light_green_50 = 0xfff1f8e9;
constexpr u32 light_green_100 = 0xffdcedc8;
constexpr u32 light_green_200 = 0xffc5e1a5;
constexpr u32 light_green_300 = 0xffaed581;
constexpr u32 light_green_400 = 0xff9ccc65;
constexpr u32 light_green_500 = 0xff8bc34a;
constexpr u32 light_green_600 = 0xff7cb342;
constexpr u32 light_green_700 = 0xff689f38;
constexpr u32 light_green_800 = 0xff558b2f;
constexpr u32 light_green_900 = 0xff33691e;
constexpr u32 light_green_A100 = 0xffccff90;
constexpr u32 light_green_A200 = 0xffb2ff59;
constexpr u32 light_green_A400 = 0xff76ff03;
constexpr u32 light_green_A700 = 0xff64dd17;

constexpr u32 lime_50 = 0xfff9ebe7;
constexpr u32 lime_100 = 0xfff0f4c3;
constexpr u32 lime_200 = 0xffe6ee9c;
constexpr u32 lime_300 = 0xffdce775;
constexpr u32 lime_400 = 0xffd4e157;
constexpr u32 lime_500 = 0xffcddc39;
constexpr u32 lime_600 = 0xffc0ca33;
constexpr u32 lime_700 = 0xffafb42b;
constexpr u32 lime_800 = 0xff9e9d24;
constexpr u32 lime_900 = 0xff827717;
constexpr u32 lime_A100 = 0xfff4ff81;
constexpr u32 lime_A200 = 0xffeeff41;
constexpr u32 lime_A400 = 0xffc6ff00;
constexpr u32 lime_A700 = 0xffaeea00;

constexpr u32 yellow_50 = 0xfffffde7;
constexpr u32 yellow_100 = 0xfffff9c4;
constexpr u32 yellow_200 = 0xfffff59d;
constexpr u32 yellow_300 = 0xfffff176;
constexpr u32 yellow_400 = 0xffffee58;
constexpr u32 yellow_500 = 0xffffeb3b;
constexpr u32 yellow_600 = 0xfffdd835;
constexpr u32 yellow_700 = 0xfffbc02d;
constexpr u32 yellow_800 = 0xfff9a825;
constexpr u32 yellow_900 = 0xfff57f17;
constexpr u32 yellow_A100 = 0xffffff8d;
constexpr u32 yellow_A200 = 0xffffff00;
constexpr u32 yellow_A400 = 0xffffea00;
constexpr u32 yellow_A700 = 0xffffd600;

constexpr u32 amber_50 = 0xfffff8e1;
constexpr u32 amber_100 = 0xffffecb3;
constexpr u32 amber_200 = 0xffffe082;
constexpr u32 amber_300 = 0xffffd54f;
constexpr u32 amber_400 = 0xffffca28;
constexpr u32 amber_500 = 0xffffc107;
constexpr u32 amber_600 = 0xffffb300;
constexpr u32 amber_700 = 0xffffa000;
constexpr u32 amber_800 = 0xffff8f00;
constexpr u32 amber_900 = 0xffff6f00;
constexpr u32 amber_A100 = 0xffffe57f;
constexpr u32 amber_A200 = 0xffffd740;
constexpr u32 amber_A400 = 0xffffc400;
constexpr u32 amber_A700 = 0xffffab00;

constexpr u32 orange_50 = 0xfffff3e0;
constexpr u32 orange_100 = 0xffffe0b2;
constexpr u32 orange_200 = 0xffffcc80;
constexpr u32 orange_300 = 0xffffb74d;
constexpr u32 orange_400 = 0xffffa726;
constexpr u32 orange_500 = 0xffff9800;
constexpr u32 orange_600 = 0xfffb8c00;
constexpr u32 orange_700 = 0xfff57c00;
constexpr u32 orange_800 = 0xffef6c00;
constexpr u32 orange_900 = 0xffe65100;
constexpr u32 orange_A100 = 0xffffd180;
constexpr u32 orange_A200 = 0xffffab40;
constexpr u32 orange_A400 = 0xffff9100;
constexpr u32 orange_A700 = 0xffff6d00;

constexpr u32 deep_orange_50 = 0xfffbe9e7;
constexpr u32 deep_orange_100 = 0xffffccbc;
constexpr u32 deep_orange_200 = 0xffffab91;
constexpr u32 deep_orange_300 = 0xffff8a65;
constexpr u32 deep_orange_400 = 0xffff7043;
constexpr u32 deep_orange_500 = 0xffff5722;
constexpr u32 deep_orange_600 = 0xfff4511e;
constexpr u32 deep_orange_700 = 0xffe64a19;
constexpr u32 deep_orange_800 = 0xffd84315;
constexpr u32 deep_orange_900 = 0xffbf360c;
constexpr u32 deep_orange_A100 = 0xffff9e80;
constexpr u32 deep_orange_A200 = 0xffff6e40;
constexpr u32 deep_orange_A400 = 0xffff3d00;
constexpr u32 deep_orange_A700 = 0xffdd2c00;

constexpr u32 brown_50 = 0xffefebe9;
constexpr u32 brown_100 = 0xffd7ccc8;
constexpr u32 brown_200 = 0xffbcaaa4;
constexpr u32 brown_300 = 0xffa1887f;
constexpr u32 brown_400 = 0xff8d6e63;
constexpr u32 brown_500 = 0xff795548;
constexpr u32 brown_600 = 0xff6d4c41;
constexpr u32 brown_700 = 0xff5d4037;
constexpr u32 brown_800 = 0xff4e342e;
constexpr u32 brown_900 = 0xff3e2723;

constexpr u32 grey_50 = 0xfffafafa;
constexpr u32 grey_100 = 0xfff5f5f5;
constexpr u32 grey_200 = 0xffeeeeee;
constexpr u32 grey_300 = 0xffe0e0e0;
constexpr u32 grey_400 = 0xffbdbdbd;
constexpr u32 grey_500 = 0xff9e9e9e;
constexpr u32 grey_600 = 0xff757575;
constexpr u32 grey_700 = 0xff616161;
constexpr u32 grey_800 = 0xff424242;
constexpr u32 grey_900 = 0xff212121;

constexpr u32 blue_grey_50 = 0xffeceff1;
constexpr u32 blue_grey_100 = 0xffcfd8dc;
constexpr u32 blue_grey_200 = 0xffb0bec5;
constexpr u32 blue_grey_300 = 0xff90a4ae;
constexpr u32 blue_grey_400 = 0xff78909c;
constexpr u32 blue_grey_500 = 0xff607d8b;
constexpr u32 blue_grey_600 = 0xff546e7a;
constexpr u32 blue_grey_700 = 0xff455a64;
constexpr u32 blue_grey_800 = 0xff37474f;
constexpr u32 blue_grey_900 = 0xff263238;

constexpr u32 black = 0xff000000;
constexpr u32 white = 0xffffffff;
constexpr u32 null = 0x00000000;

constexpr u32 red = red_500;
constexpr u32 dark_red = red_900;
constexpr u32 coral = red_200;
constexpr u32 rich_red = 0xffff0000;
constexpr u32 pink = pink_500;
constexpr u32 rose = pink_A100;
constexpr u32 purple = purple500;
constexpr u32 magenta = purpleA200;
constexpr u32 dark_magenta = purpleA700;
constexpr u32 deep_purple = deep_purple_500;
constexpr u32 indigo = Indigo500;
constexpr u32 blue = blue_500;
constexpr u32 dark_blue = blue_900;
constexpr u32 rich_blue = 0xff0000ff;
constexpr u32 light_blue = light_blue_500;
constexpr u32 sky_blue = light_blue_A100;
constexpr u32 navy = light_blue_800;
constexpr u32 cyan = cyan_500;
constexpr u32 dark_cyan = cyan_900;
constexpr u32 teal = teal_500;
constexpr u32 dark_teal = teal_900;
constexpr u32 green = green_500;
constexpr u32 dark_green = green_900;
constexpr u32 rich_green = 0xff00ff00;
constexpr u32 light_green = light_green_500;
constexpr u32 mint = light_green_900;
constexpr u32 lime = lime_500;
constexpr u32 olive = lime_900;
constexpr u32 yellow = yellow_500;
constexpr u32 rich_yellow = yellow_A200;
constexpr u32 amber = amber_500;
constexpr u32 gold = amber_300;
constexpr u32 pale_gold = amber_A100;
constexpr u32 orange = orange_500;
constexpr u32 skin = orange_100;
constexpr u32 deep_orange = deep_orange_500;
constexpr u32 brick = deep_orange_900;
constexpr u32 brown = brown_500;
constexpr u32 dark_brown = brown_900;
constexpr u32 cream_white = orange_50;
constexpr u32 wheat = amber_100;
constexpr u32 grey = grey_500;
constexpr u32 dark = grey_900;
constexpr u32 silver = grey_300;
constexpr u32 blue_grey = blue_grey_500;

constexpr u32 _default_ = red_300;

} // namespace hermes::colors::argb
