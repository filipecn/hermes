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
///\file memory.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-09-24
///
///\brief

#include <hermes/storage/memory.h>

namespace hermes {

[[maybe_unused]] u32 mem::cache_l1_size = 64;

void *mem::allocAligned(size_t size, size_t align) {
  // allocate align more bytes to store shift value
  size_t actual_bytes = size + align;
  // allocate unaligned block
  u8 *p_raw_mem = new u8[actual_bytes];
  // align block
  u8 *p_aligned_mem = alignPointer(p_raw_mem, align);
  // if no alignment occurred, shift it up the full 'align' bytes to make room
  // to store the shift
  if (p_aligned_mem == p_raw_mem)
    p_aligned_mem += align;
  // determine the shift and store it
  ptrdiff_t shift = p_aligned_mem - p_raw_mem;
  // alignment can't be greater than 256
  HERMES_ASSERT(shift > 0 && shift <= 256)
  p_aligned_mem[-1] = static_cast<u8>(shift & 0xFF);
  return p_aligned_mem;
}

void mem::freeAligned(void *p_mem) {
  if (p_mem) {
    u8 *p_aligned_mem = reinterpret_cast<u8 *>(p_mem);
    // extract the shift
    ptrdiff_t shift = p_aligned_mem[-1];
    if (shift == 0)
      shift = 256;
    // back up to the actual allocated address and array-delete it
    u8 *p_raw_mem = p_aligned_mem - shift;
    delete[] p_raw_mem;
  }
}

}

