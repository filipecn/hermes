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

#include <hermes/sampling/sampling.h>

namespace hermes::sampling {

std::vector<geo::point2> sampleGrid(const geo::bounds::bbox2 &bounds,
                                    const size2 &resolution) {
  std::vector<geo::point2> samples(resolution.total());
  geo::vec2 cell_size =
      bounds.extends() / geo::vec2(resolution.width - 1, resolution.height - 1);
  u32 fij = 0;
  for (auto ij : range2(resolution)) {
    samples[fij].x = bounds.lower.x + static_cast<f32>(ij.i) * cell_size.x;
    samples[fij++].y = bounds.lower.y + static_cast<f32>(ij.j) * cell_size.y;
  }
  return samples;
}

} // namespace hermes::sampling
