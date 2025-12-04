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

/// \file   math.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2019-17-09
/// \brief  numbers functions

#include <hermes/math/space_filling.h>
#include <hermes/numeric/numeric.h>

namespace hermes::math::space_filling {

h_size mortonEncode(const hermes::index2 &coordinates) {
  return numbers::interleaveBits(coordinates.i, coordinates.j);
}

uint32_t morton_1(uint64_t x_or_y_bits) {
  x_or_y_bits = x_or_y_bits & 0x5555555555555555; // Selects odd/even bits
  x_or_y_bits = (x_or_y_bits | (x_or_y_bits >> 1)) & 0x3333333333333333;
  x_or_y_bits = (x_or_y_bits | (x_or_y_bits >> 2)) & 0x0F0F0F0F0F0F0F0F;
  x_or_y_bits = (x_or_y_bits | (x_or_y_bits >> 4)) & 0x00FF00FF00FF00FF;
  x_or_y_bits = (x_or_y_bits | (x_or_y_bits >> 8)) & 0x0000FFFF0000FFFF;
  x_or_y_bits = (x_or_y_bits | (x_or_y_bits >> 16)) & 0x00000000FFFFFFFF;
  return (uint32_t)x_or_y_bits;
}

hermes::index2 mortonDecode2(h_size z) {
  return hermes::index2(morton_1(z), morton_1(z >> 1));
}

MortonRange::iterator::iterator(h_size z) noexcept : z_(z) {}

MortonRange::iterator &MortonRange::iterator::operator++() {
  z_++;
  return *this;
}

const MortonRange::iterator &MortonRange::iterator::operator*() const {
  return *this;
}

bool MortonRange::iterator::operator==(const MortonRange::iterator &rhs) const {
  return z_ == rhs.z_;
}

bool MortonRange::iterator::operator!=(const MortonRange::iterator &rhs) const {
  return z_ != rhs.z_;
}

h_size MortonRange::iterator::index() const { return z_; }

hermes::index2 MortonRange::iterator::coord2() const {
  return mortonDecode2(z_);
}

MortonRange::MortonRange(h_size start, h_size end) noexcept
    : start_(start), end_(end) {}

MortonRange::iterator MortonRange::begin() const { return {start_}; }

MortonRange::iterator MortonRange::end() const { return {end_}; }

} // namespace hermes::math::space_filling
