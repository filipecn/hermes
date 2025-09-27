/* Copyright (c) 2025, FilipeCN.
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

/// \file   color.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2025-01-06

#pragma once

#include <hermes/core/types.h>
#include <hermes/geometry/vector.h>

namespace hermes::colors {

// *****************************************************************************
//                                                                   RGB_Color
// *****************************************************************************

struct RGBA_Color;

struct RGB_Color {
  /// Constructs color from rgb [0,255] values
  /// \param r red component.
  /// \param g green component.
  /// \param b blue component.
  static RGB_Color fromU32(u32 r, u32 g, u32 b) {
    return {r / 255.f, g / 255.f, b / 255.f};
  }
  /// \brief Extracts RGB components from unsigned integer.
  /// \param color
  static RGB_Color rbgFromU32(u32 color) {
    return fromU32((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF);
  }
  /// \brief Creates color black representation
  static RGB_Color Black() { return {0.f, 0.f, 0.f}; }
  /// \brief Creates color white representation
  static RGB_Color White() { return {1.f, 1.f, 1.f}; }
  /// \brief Creates color red representation
  static RGB_Color Red() { return {1.f, 0.f, 0.f}; }
  /// \brief Creates color green representation
  static RGB_Color Green() { return {0.f, 1.f, 0.f}; }
  /// \brief Creates color blue representation
  static RGB_Color Blue() { return {0.f, 0.f, 1.f}; }
  /// \brief Creates color purple representation
  static RGB_Color Purple() { return {1.f, 0.f, 1.f}; }
  /// \brief Creates color yellow representation
  static RGB_Color Yellow() { return {1.f, 1.f, 0.f}; }
  /// \brief Creates color gray representation
  /// \param v rgb value
  static RGB_Color Gray(f32 v) { return {v, v, v}; }

  /// \brief Default constructor (color black)
  RGB_Color() = default;
  /// \brief rgb [0,1] components constructor.
  /// \param r red component.
  /// \param g green component.
  /// \param b blue component.
  RGB_Color(f32 r, f32 g, f32 b);

  /// \brief Gets component array pointer
  /// \return
  HERMES_NODISCARD const f32 *asArray() const;
  /// \brief Creates a copy with a given opacity value
  /// \param alpha
  HERMES_NODISCARD RGBA_Color withAlpha(f32 alpha) const;

  f32 r{0.f}; //!< red component in [0,1] interval
  f32 g{0.f}; //!< green component in [0,1] interval
  f32 b{0.f}; //!< blue component in [0,1] interval

  HERMES_TO_STRING_FRIEND(RGB_Color)
};

// *****************************************************************************
//                                                                  RGBA_Color
// *****************************************************************************

/// RGBA color representation
struct RGBA_Color : public RGB_Color {
  /// Constructs color from rgba [0,255] values
  /// \param r red component.
  /// \param g green component.
  /// \param b blue component.
  /// \param a alpha component.
  static RGBA_Color fromU32(u32 r, u32 g, u32 b, u32 a = 255) {
    return {r / 255.f, g / 255.f, b / 255.f, a / 255.f};
  }
  /// \brief Extracts RGB components from unsigned integer.
  /// \param color
  static RGBA_Color rbgFromU32(u32 color) {
    return fromU32((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF);
  }

  /// \brief Creates color transparent
  static RGBA_Color Transparent() { return {0.f, 0.f, 0.f, 0.f}; }
  /// \brief Creates color black representation
  /// \param alpha opacity value
  static RGBA_Color Black(f32 alpha = 1.f) { return {0.f, 0.f, 0.f, alpha}; }
  /// \brief Creates color white representation
  /// \param alpha opacity value
  static RGBA_Color White(f32 alpha = 1.f) { return {1.f, 1.f, 1.f, alpha}; }
  /// \brief Creates color red representation
  /// \param alpha opacity value
  static RGBA_Color Red(f32 alpha = 1.f) { return {1.f, 0.f, 0.f, alpha}; }
  /// \brief Creates color green representation
  /// \param alpha opacity value
  static RGBA_Color Green(f32 alpha = 1.f) { return {0.f, 1.f, 0.f, alpha}; }
  /// \brief Creates color blue representation
  /// \param alpha opacity value
  static RGBA_Color Blue(f32 alpha = 1.f) { return {0.f, 0.f, 1.f, alpha}; }
  /// \brief Creates color purple representation
  /// \param alpha opacity value
  static RGBA_Color Purple(f32 alpha = 1.f) { return {1.f, 0.f, 1.f, alpha}; }
  /// \brief Creates color yellow representation
  /// \param alpha opacity value
  static RGBA_Color Yellow(f32 alpha = 1.f) { return {1.f, 1.f, 0.f, alpha}; }
  /// \brief Creates color gray representation
  /// \param v rgb value
  /// \param alpha opacity value
  static RGBA_Color Gray(f32 v, f32 alpha = 1.f) { return {v, v, v, alpha}; }

  /// \brief Default constructor (color black)
  RGBA_Color() = default;
  /// rgba [0,1] components constructor
  /// \param r red component.
  /// \param g green component.
  /// \param b blue component.
  /// \param a alpha component.
  RGBA_Color(f32 r, f32 g, f32 b, f32 a = 1.f);

  /// \brief Gets rgb components vector
  HERMES_NODISCARD RGB_Color rgb() const;

  f32 a{1.f}; //!< opacity component in [0,1] interval

  HERMES_TO_STRING_FRIEND(RGBA_Color)
};

/// \brief Linearly interpolates between two colors
/// \param t
/// \param a
/// \param b
/// \return
// inline RGBA_Color mix(f32 t, const RGBA_Color &a, const RGBA_Color &b) {
//   return {hermes::interpolation::lerp(t, a.r, b.r),
//           hermes::interpolation::lerp(t, a.g, b.g),
//           hermes::interpolation::lerp(t, a.b, b.b)};
// }

} // namespace hermes::colors

namespace hermes {

HERMES_DECLARE_TO_STRING_DEBUG_METHOD(colors::RGBA_Color);
HERMES_DECLARE_TO_STRING_DEBUG_METHOD(colors::RGB_Color);

} // namespace hermes
