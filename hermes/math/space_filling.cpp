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

OnionRange::iterator::iterator(const index2 &lower, const index2 &upper,
                               h_size layer) noexcept
    : lower_(lower), upper_(upper), direction_(1, 0), position_(layer, layer),
      layer_(layer) {
  auto size = upper - lower + index2(1, 1);

  flat_index_ = layer * size.i * 2 + (size.j - 2 * layer) * layer * 2;

  if (size.j - 2 * static_cast<i32>(layer) < 0)
    flat_index_ = size.i * size.j;

  if ((size.i == 1 || size.j == 1) && layer > 0)
    flat_index_ = size.i * size.j;
}

OnionRange::iterator &OnionRange::iterator::operator++() {
  auto new_position = position_ + direction_;
  if (direction_.i == 1) { // >
    if (upper_.i - new_position.i < static_cast<i32>(layer_)) {
      direction_ = {0, 1};
    }
  }
  if (direction_.j == 1) { // ^
    if (upper_.j - new_position.j < static_cast<i32>(layer_)) {
      direction_ = {-1, 0};
    }
  }
  if (direction_.i == -1) { // <
    if (new_position.i - lower_.i < static_cast<i32>(layer_)) {
      direction_ = {0, -1};
    }
  }
  if (direction_.j == -1) { // v
    if (new_position.j - lower_.j < static_cast<i32>(layer_ + 1)) {
      direction_ = {1, 0};
      layer_++;
    }
  }
  position_ += direction_;
  flat_index_++;
  return *this;
}

const OnionRange::iterator &OnionRange::iterator::operator*() const {
  return *this;
}

h_size OnionRange::iterator::index() const { return flat_index_; }

index2 OnionRange::iterator::coord2() const { return position_; }

bool OnionRange::iterator::operator==(const OnionRange::iterator &rhs) const {
  return flat_index_ == rhs.flat_index_;
}

bool OnionRange::iterator::operator!=(const OnionRange::iterator &rhs) const {
  return !(*this == rhs);
}

OnionRange::OnionRange(const size2 &size, h_size layer_count,
                       h_size start_layer) noexcept
    : lower_(0, 0), upper_(size - index2(1, 1)), start_layer_(start_layer) {
  h_size min_extent = std::min(size.width, size.height);
  h_size max_layers = min_extent / 2U + (min_extent % 2 ? 1 : 0);
  if (layer_count)
    end_layer_ = std::min(start_layer_ + layer_count, max_layers);
  else
    end_layer_ = max_layers;
  if (end_layer_ == start_layer_)
    end_layer_++;
}

OnionRange::iterator OnionRange::begin() const {
  return {lower_, upper_, start_layer_};
}

OnionRange::iterator OnionRange::end() const {
  return {lower_, upper_, end_layer_};
}

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

index2 mortonDecode2(h_size z) { return index2(morton_1(z), morton_1(z >> 1)); }

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

index2 MortonRange::iterator::coord2() const { return mortonDecode2(z_); }

MortonRange::MortonRange(h_size start, h_size end) noexcept
    : start_(start), end_(end) {}

MortonRange::iterator MortonRange::begin() const { return {start_}; }

MortonRange::iterator MortonRange::end() const { return {end_}; }

} // namespace hermes::math::space_filling
