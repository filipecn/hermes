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
///\file argb_colors.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-01-06
///
///\brief
///\note
/// This file was inspired on Sergey Yagovtsev's Easy Profiler source code
/// https://github.com/yse/easy_profiler/blob/develop/easy_profiler_core/include/easy/details/profiler_colors.h

#ifndef HERMES_HERMES_LOGGING_ARGB_COLORS_H
#define HERMES_HERMES_LOGGING_ARGB_COLORS_H

#include <hermes/common/defs.h>

namespace hermes::argb_colors {

inline u32 constexpr argb2rgba(u32 argb) {
  return (argb >> 24) | (argb << 8);
}

inline u32 constexpr fromRGBA(u8 red, u8 green, u8 blue, u8 alpha = 0xff) {
  return (static_cast<u32>(alpha) << 24) | (static_cast<u32>(red) << 16) | (static_cast<u32>(green) << 8)
      | static_cast<u32>(blue);
}

// Google Material Design colors
// See https://material.google.com/style/color.html

constexpr u32 Red50 = 0xffffebee;
constexpr u32 Red100 = 0xffffcdd2;
constexpr u32 Red200 = 0xffef9a9a;
constexpr u32 Red300 = 0xffe57373;
constexpr u32 Red400 = 0xffef5350;
constexpr u32 Red500 = 0xfff44336;
constexpr u32 Red600 = 0xffe53935;
constexpr u32 Red700 = 0xffd32f2f;
constexpr u32 Red800 = 0xffc62828;
constexpr u32 Red900 = 0xffb71c1c;
constexpr u32 RedA100 = 0xffff8a80;
constexpr u32 RedA200 = 0xffff5252;
constexpr u32 RedA400 = 0xffff1744;
constexpr u32 RedA700 = 0xffd50000;

constexpr u32 Pink50 = 0xfffce4ec;
constexpr u32 Pink100 = 0xfff8bbd0;
constexpr u32 Pink200 = 0xfff48fb1;
constexpr u32 Pink300 = 0xfff06292;
constexpr u32 Pink400 = 0xffec407a;
constexpr u32 Pink500 = 0xffe91e63;
constexpr u32 Pink600 = 0xffd81b60;
constexpr u32 Pink700 = 0xffc2185b;
constexpr u32 Pink800 = 0xffad1457;
constexpr u32 Pink900 = 0xff880e4f;
constexpr u32 PinkA100 = 0xffff80ab;
constexpr u32 PinkA200 = 0xffff4081;
constexpr u32 PinkA400 = 0xfff50057;
constexpr u32 PinkA700 = 0xffc51162;

constexpr u32 Purple50 = 0xfff3e5f5;
constexpr u32 Purple100 = 0xffe1bee7;
constexpr u32 Purple200 = 0xffce93d8;
constexpr u32 Purple300 = 0xffba68c8;
constexpr u32 Purple400 = 0xffab47bc;
constexpr u32 Purple500 = 0xff9c27b0;
constexpr u32 Purple600 = 0xff8e24aa;
constexpr u32 Purple700 = 0xff7b1fa2;
constexpr u32 Purple800 = 0xff6a1b9a;
constexpr u32 Purple900 = 0xff4a148c;
constexpr u32 PurpleA100 = 0xffea80fc;
constexpr u32 PurpleA200 = 0xffe040fb;
constexpr u32 PurpleA400 = 0xffd500f9;
constexpr u32 PurpleA700 = 0xffaa00ff;

constexpr u32 DeepPurple50 = 0xffede7f6;
constexpr u32 DeepPurple100 = 0xffd1c4e9;
constexpr u32 DeepPurple200 = 0xffb39ddb;
constexpr u32 DeepPurple300 = 0xff9575cd;
constexpr u32 DeepPurple400 = 0xff7e57c2;
constexpr u32 DeepPurple500 = 0xff673ab7;
constexpr u32 DeepPurple600 = 0xff5e35b1;
constexpr u32 DeepPurple700 = 0xff512da8;
constexpr u32 DeepPurple800 = 0xff4527a0;
constexpr u32 DeepPurple900 = 0xff311b92;
constexpr u32 DeepPurpleA100 = 0xffb388ff;
constexpr u32 DeepPurpleA200 = 0xff7c4dff;
constexpr u32 DeepPurpleA400 = 0xff651fff;
constexpr u32 DeepPurpleA700 = 0xff6200ea;

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

constexpr u32 Blue50 = 0xffe3f2fd;
constexpr u32 Blue100 = 0xffbbdefb;
constexpr u32 Blue200 = 0xff90caf9;
constexpr u32 Blue300 = 0xff64b5f6;
constexpr u32 Blue400 = 0xff42a5f5;
constexpr u32 Blue500 = 0xff2196f3;
constexpr u32 Blue600 = 0xff1e88e5;
constexpr u32 Blue700 = 0xff1976d2;
constexpr u32 Blue800 = 0xff1565c0;
constexpr u32 Blue900 = 0xff0d47a1;
constexpr u32 BlueA100 = 0xff82b1ff;
constexpr u32 BlueA200 = 0xff448aff;
constexpr u32 BlueA400 = 0xff2979ff;
constexpr u32 BlueA700 = 0xff2962ff;

constexpr u32 LightBlue50 = 0xffe1f5fe;
constexpr u32 LightBlue100 = 0xffb3e5fc;
constexpr u32 LightBlue200 = 0xff81d4fa;
constexpr u32 LightBlue300 = 0xff4fc3f7;
constexpr u32 LightBlue400 = 0xff29b6f6;
constexpr u32 LightBlue500 = 0xff03a9f4;
constexpr u32 LightBlue600 = 0xff039be5;
constexpr u32 LightBlue700 = 0xff0288d1;
constexpr u32 LightBlue800 = 0xff0277bd;
constexpr u32 LightBlue900 = 0xff01579b;
constexpr u32 LightBlueA100 = 0xff80d8ff;
constexpr u32 LightBlueA200 = 0xff40c4ff;
constexpr u32 LightBlueA400 = 0xff00b0ff;
constexpr u32 LightBlueA700 = 0xff0091ea;

constexpr u32 Cyan50 = 0xffe0f7fa;
constexpr u32 Cyan100 = 0xffb2ebf2;
constexpr u32 Cyan200 = 0xff80deea;
constexpr u32 Cyan300 = 0xff4dd0e1;
constexpr u32 Cyan400 = 0xff26c6da;
constexpr u32 Cyan500 = 0xff00bcd4;
constexpr u32 Cyan600 = 0xff00acc1;
constexpr u32 Cyan700 = 0xff0097a7;
constexpr u32 Cyan800 = 0xff00838f;
constexpr u32 Cyan900 = 0xff006064;
constexpr u32 CyanA100 = 0xff84ffff;
constexpr u32 CyanA200 = 0xff18ffff;
constexpr u32 CyanA400 = 0xff00e5ff;
constexpr u32 CyanA700 = 0xff00b8d4;

constexpr u32 Teal50 = 0xffe0f2f1;
constexpr u32 Teal100 = 0xffb2dfdb;
constexpr u32 Teal200 = 0xff80cbc4;
constexpr u32 Teal300 = 0xff4db6ac;
constexpr u32 Teal400 = 0xff26a69a;
constexpr u32 Teal500 = 0xff009688;
constexpr u32 Teal600 = 0xff00897b;
constexpr u32 Teal700 = 0xff00796b;
constexpr u32 Teal800 = 0xff00695c;
constexpr u32 Teal900 = 0xff004d40;
constexpr u32 TealA100 = 0xffa7ffeb;
constexpr u32 TealA200 = 0xff64ffda;
constexpr u32 TealA400 = 0xff1de9b6;
constexpr u32 TealA700 = 0xff00bfa5;

constexpr u32 Green50 = 0xffe8f5e9;
constexpr u32 Green100 = 0xffc8e6c9;
constexpr u32 Green200 = 0xffa5d6a7;
constexpr u32 Green300 = 0xff81c784;
constexpr u32 Green400 = 0xff66bb6a;
constexpr u32 Green500 = 0xff4caf50;
constexpr u32 Green600 = 0xff43a047;
constexpr u32 Green700 = 0xff388e3c;
constexpr u32 Green800 = 0xff2e7d32;
constexpr u32 Green900 = 0xff1b5e20;
constexpr u32 GreenA100 = 0xffb9f6ca;
constexpr u32 GreenA200 = 0xff69f0ae;
constexpr u32 GreenA400 = 0xff00e676;
constexpr u32 GreenA700 = 0xff00c853;

constexpr u32 LightGreen50 = 0xfff1f8e9;
constexpr u32 LightGreen100 = 0xffdcedc8;
constexpr u32 LightGreen200 = 0xffc5e1a5;
constexpr u32 LightGreen300 = 0xffaed581;
constexpr u32 LightGreen400 = 0xff9ccc65;
constexpr u32 LightGreen500 = 0xff8bc34a;
constexpr u32 LightGreen600 = 0xff7cb342;
constexpr u32 LightGreen700 = 0xff689f38;
constexpr u32 LightGreen800 = 0xff558b2f;
constexpr u32 LightGreen900 = 0xff33691e;
constexpr u32 LightGreenA100 = 0xffccff90;
constexpr u32 LightGreenA200 = 0xffb2ff59;
constexpr u32 LightGreenA400 = 0xff76ff03;
constexpr u32 LightGreenA700 = 0xff64dd17;

constexpr u32 Lime50 = 0xfff9ebe7;
constexpr u32 Lime100 = 0xfff0f4c3;
constexpr u32 Lime200 = 0xffe6ee9c;
constexpr u32 Lime300 = 0xffdce775;
constexpr u32 Lime400 = 0xffd4e157;
constexpr u32 Lime500 = 0xffcddc39;
constexpr u32 Lime600 = 0xffc0ca33;
constexpr u32 Lime700 = 0xffafb42b;
constexpr u32 Lime800 = 0xff9e9d24;
constexpr u32 Lime900 = 0xff827717;
constexpr u32 LimeA100 = 0xfff4ff81;
constexpr u32 LimeA200 = 0xffeeff41;
constexpr u32 LimeA400 = 0xffc6ff00;
constexpr u32 LimeA700 = 0xffaeea00;

constexpr u32 Yellow50 = 0xfffffde7;
constexpr u32 Yellow100 = 0xfffff9c4;
constexpr u32 Yellow200 = 0xfffff59d;
constexpr u32 Yellow300 = 0xfffff176;
constexpr u32 Yellow400 = 0xffffee58;
constexpr u32 Yellow500 = 0xffffeb3b;
constexpr u32 Yellow600 = 0xfffdd835;
constexpr u32 Yellow700 = 0xfffbc02d;
constexpr u32 Yellow800 = 0xfff9a825;
constexpr u32 Yellow900 = 0xfff57f17;
constexpr u32 YellowA100 = 0xffffff8d;
constexpr u32 YellowA200 = 0xffffff00;
constexpr u32 YellowA400 = 0xffffea00;
constexpr u32 YellowA700 = 0xffffd600;

constexpr u32 Amber50 = 0xfffff8e1;
constexpr u32 Amber100 = 0xffffecb3;
constexpr u32 Amber200 = 0xffffe082;
constexpr u32 Amber300 = 0xffffd54f;
constexpr u32 Amber400 = 0xffffca28;
constexpr u32 Amber500 = 0xffffc107;
constexpr u32 Amber600 = 0xffffb300;
constexpr u32 Amber700 = 0xffffa000;
constexpr u32 Amber800 = 0xffff8f00;
constexpr u32 Amber900 = 0xffff6f00;
constexpr u32 AmberA100 = 0xffffe57f;
constexpr u32 AmberA200 = 0xffffd740;
constexpr u32 AmberA400 = 0xffffc400;
constexpr u32 AmberA700 = 0xffffab00;

constexpr u32 Orange50 = 0xfffff3e0;
constexpr u32 Orange100 = 0xffffe0b2;
constexpr u32 Orange200 = 0xffffcc80;
constexpr u32 Orange300 = 0xffffb74d;
constexpr u32 Orange400 = 0xffffa726;
constexpr u32 Orange500 = 0xffff9800;
constexpr u32 Orange600 = 0xfffb8c00;
constexpr u32 Orange700 = 0xfff57c00;
constexpr u32 Orange800 = 0xffef6c00;
constexpr u32 Orange900 = 0xffe65100;
constexpr u32 OrangeA100 = 0xffffd180;
constexpr u32 OrangeA200 = 0xffffab40;
constexpr u32 OrangeA400 = 0xffff9100;
constexpr u32 OrangeA700 = 0xffff6d00;

constexpr u32 DeepOrange50 = 0xfffbe9e7;
constexpr u32 DeepOrange100 = 0xffffccbc;
constexpr u32 DeepOrange200 = 0xffffab91;
constexpr u32 DeepOrange300 = 0xffff8a65;
constexpr u32 DeepOrange400 = 0xffff7043;
constexpr u32 DeepOrange500 = 0xffff5722;
constexpr u32 DeepOrange600 = 0xfff4511e;
constexpr u32 DeepOrange700 = 0xffe64a19;
constexpr u32 DeepOrange800 = 0xffd84315;
constexpr u32 DeepOrange900 = 0xffbf360c;
constexpr u32 DeepOrangeA100 = 0xffff9e80;
constexpr u32 DeepOrangeA200 = 0xffff6e40;
constexpr u32 DeepOrangeA400 = 0xffff3d00;
constexpr u32 DeepOrangeA700 = 0xffdd2c00;

constexpr u32 Brown50 = 0xffefebe9;
constexpr u32 Brown100 = 0xffd7ccc8;
constexpr u32 Brown200 = 0xffbcaaa4;
constexpr u32 Brown300 = 0xffa1887f;
constexpr u32 Brown400 = 0xff8d6e63;
constexpr u32 Brown500 = 0xff795548;
constexpr u32 Brown600 = 0xff6d4c41;
constexpr u32 Brown700 = 0xff5d4037;
constexpr u32 Brown800 = 0xff4e342e;
constexpr u32 Brown900 = 0xff3e2723;

constexpr u32 Grey50 = 0xfffafafa;
constexpr u32 Grey100 = 0xfff5f5f5;
constexpr u32 Grey200 = 0xffeeeeee;
constexpr u32 Grey300 = 0xffe0e0e0;
constexpr u32 Grey400 = 0xffbdbdbd;
constexpr u32 Grey500 = 0xff9e9e9e;
constexpr u32 Grey600 = 0xff757575;
constexpr u32 Grey700 = 0xff616161;
constexpr u32 Grey800 = 0xff424242;
constexpr u32 Grey900 = 0xff212121;

constexpr u32 BlueGrey50 = 0xffeceff1;
constexpr u32 BlueGrey100 = 0xffcfd8dc;
constexpr u32 BlueGrey200 = 0xffb0bec5;
constexpr u32 BlueGrey300 = 0xff90a4ae;
constexpr u32 BlueGrey400 = 0xff78909c;
constexpr u32 BlueGrey500 = 0xff607d8b;
constexpr u32 BlueGrey600 = 0xff546e7a;
constexpr u32 BlueGrey700 = 0xff455a64;
constexpr u32 BlueGrey800 = 0xff37474f;
constexpr u32 BlueGrey900 = 0xff263238;

constexpr u32 Black = 0xff000000;
constexpr u32 White = 0xffffffff;
constexpr u32 Null = 0x00000000;

constexpr u32 Red = Red500;
constexpr u32 DarkRed = Red900;
constexpr u32 Coral = Red200;
constexpr u32 RichRed = 0xffff0000;
constexpr u32 Pink = Pink500;
constexpr u32 Rose = PinkA100;
constexpr u32 Purple = Purple500;
constexpr u32 Magenta = PurpleA200;
constexpr u32 DarkMagenta = PurpleA700;
constexpr u32 DeepPurple = DeepPurple500;
constexpr u32 Indigo = Indigo500;
constexpr u32 Blue = Blue500;
constexpr u32 DarkBlue = Blue900;
constexpr u32 RichBlue = 0xff0000ff;
constexpr u32 LightBlue = LightBlue500;
constexpr u32 SkyBlue = LightBlueA100;
constexpr u32 Navy = LightBlue800;
constexpr u32 Cyan = Cyan500;
constexpr u32 DarkCyan = Cyan900;
constexpr u32 Teal = Teal500;
constexpr u32 DarkTeal = Teal900;
constexpr u32 Green = Green500;
constexpr u32 DarkGreen = Green900;
constexpr u32 RichGreen = 0xff00ff00;
constexpr u32 LightGreen = LightGreen500;
constexpr u32 Mint = LightGreen900;
constexpr u32 Lime = Lime500;
constexpr u32 Olive = Lime900;
constexpr u32 Yellow = Yellow500;
constexpr u32 RichYellow = YellowA200;
constexpr u32 Amber = Amber500;
constexpr u32 Gold = Amber300;
constexpr u32 PaleGold = AmberA100;
constexpr u32 Orange = Orange500;
constexpr u32 Skin = Orange100;
constexpr u32 DeepOrange = DeepOrange500;
constexpr u32 Brick = DeepOrange900;
constexpr u32 Brown = Brown500;
constexpr u32 DarkBrown = Brown900;
constexpr u32 CreamWhite = Orange50;
constexpr u32 Wheat = Amber100;
constexpr u32 Grey = Grey500;
constexpr u32 Dark = Grey900;
constexpr u32 Silver = Grey300;
constexpr u32 BlueGrey = BlueGrey500;

constexpr u32 Default = Wheat;

}

#endif //HERMES_HERMES_LOGGING_ARGB_COLORS_H
