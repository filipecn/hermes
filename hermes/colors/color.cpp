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

/// \file   color.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2025-01-06

#include <hermes/colors/color.h>

namespace hermes::colors {

RGB_Color::RGB_Color(f32 r, f32 g, f32 b) : r{r}, g{g}, b{b} {}

const f32 *RGB_Color::asArray() const { return &r; }

RGBA_Color RGB_Color::withAlpha(f32 alpha) const { return {r, g, b, alpha}; }

RGBA_Color::RGBA_Color(f32 r, f32 g, f32 b, f32 a) : RGB_Color(r, g, b), a{a} {}

RGB_Color RGBA_Color::rgb() const { return {r, g, b}; }

} // namespace hermes::colors

namespace hermes {
HERMES_TO_STRING_METHOD_BEGIN(colors::RGB_Color)
HERMES_TO_STRING_METHOD_LINE("C[{}, {}, {}]", object.r, object.g, object.b)
HERMES_TO_STRING_METHOD_END

HERMES_TO_STRING_METHOD_BEGIN(colors::RGBA_Color)
HERMES_TO_STRING_METHOD_LINE("C[{}, {}, {}, {}]", object.r, object.g, object.b,
                             object.a)
HERMES_TO_STRING_METHOD_END

} // namespace hermes
