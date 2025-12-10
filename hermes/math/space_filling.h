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

#pragma once

#include <hermes/base/index.h>
#include <hermes/core/debug.h>

namespace hermes::math::space_filling {

///
///    11 10 09 08 07
///    12 19 18 17 06
///    13 14 15 16 05
///    00 01 02 03 04

class OnionRange {
public:
  class iterator {
  public:
    HERMES_CPU_GPU iterator(const index2 &lower, const index2 &upper,
                            h_size layer) noexcept;
    HERMES_CPU_GPU iterator &operator++();
    HERMES_CPU_GPU const iterator &operator*() const;
    HERMES_CPU_GPU h_size index() const;
    HERMES_CPU_GPU index2 coord2() const;

    HERMES_CPU_GPU bool operator==(const iterator &rhs) const;
    HERMES_CPU_GPU bool operator!=(const iterator &rhs) const;

  private:
    index2 lower_{};
    index2 upper_{};
    index2 direction_{};
    index2 position_{};
    h_size layer_{};
    h_size flat_index_{};
  };

  HERMES_CPU_GPU OnionRange(const size2 &size, h_size layer_count = 0,
                            h_size start_layer = 0) noexcept;
  HERMES_CPU_GPU iterator begin() const;
  HERMES_CPU_GPU iterator end() const;

private:
  index2 lower_{};
  index2 upper_{};
  h_size start_layer_{0};
  h_size end_layer_{0};
};

///
HERMES_CPU_GPU h_size mortonEncode(const index2 &coordinates);
///
HERMES_CPU_GPU index2 mortonDecode2(h_size z);

class MortonRange {
public:
  class iterator {
  public:
    HERMES_CPU_GPU iterator(h_size z = 0) noexcept;
    HERMES_CPU_GPU iterator &operator++();
    HERMES_CPU_GPU const iterator &operator*() const;
    HERMES_CPU_GPU h_size index() const;
    HERMES_CPU_GPU hermes::index2 coord2() const;

    HERMES_CPU_GPU bool operator==(const iterator &rhs) const;
    HERMES_CPU_GPU bool operator!=(const iterator &rhs) const;

  private:
    h_size z_;
  };

  HERMES_CPU_GPU MortonRange(h_size start, h_size end) noexcept;
  HERMES_CPU_GPU iterator begin() const;
  HERMES_CPU_GPU iterator end() const;

private:
  h_size start_{};
  h_size end_{};
};

} // namespace hermes::math::space_filling
