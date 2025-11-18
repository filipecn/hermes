/* Copyright (c) 2021, FilipeCN.
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

/// \file   memory.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2021-09-22

#pragma once

#include <hermes/base/size.h>
#include <hermes/core/debug.h>
#include <hermes/core/result.h>
#include <hermes/core/types.h>

namespace hermes::mem {

/// Object returned by memory allocators and other memory-related classes
/// Each class puts a meaning into its value.
struct AddressIndex {
  HERMES_CPU_GPU AddressIndex(h_size id = 0) : id(id) {}
  /// handle identifier, a value of zero identifies an invalid address
  h_size id{0};
  HERMES_NODISCARD HERMES_CPU_GPU inline bool isValid() const {
    return id != 0;
  }
};

// *****************************************************************************
//                                                                       Sizes
// *****************************************************************************
struct sizes {
  [[maybe_unused]] static u32 cache_l1_size;
};

// *****************************************************************************
//                                                                   Alignment
// *****************************************************************************
struct alignment {

  /// \param number_of_bytes
  /// \param align alignment size in number of bytes
  /// \return the actual amount of bytes necessary to store number_of_bytes
  /// under the alignment
  HERMES_CPU_GPU static h_size alignTo(h_size number_of_bytes, h_size align);

  /// \param address
  /// \param align
  /// \return
  HERMES_CPU_GPU static h_size leftAlignShift(uintptr_t address, h_size align);

  /// \param address
  /// \param align
  /// \return
  HERMES_CPU_GPU static h_size rightAlignShift(uintptr_t address, h_size align);

  /// Shifts **address** upwards if necessary to ensure it is aligned to
  /// **align** number of bytes.
  /// \param address **[in]** memory address
  /// \param align **[in]** number of bytes
  /// \return aligned address
  HERMES_CPU_GPU static uintptr_t alignAddress(uintptr_t address, h_size align);

  /// Shifts pointer **ptr** upwards if necessary to ensure it is aligned to
  /// **align** number of bytes.
  /// \tparam T data type
  /// \param ptr **[in]** pointer
  /// \param align **[in]** number of bytes
  /// \return aligned pointer
  template <typename T>
  HERMES_CPU_GPU static T *alignPointer(T *ptr, h_size align) {
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t addr_aligned = alignAddress(addr, align);
    return reinterpret_cast<T *>(addr_aligned);
  }
};

// *****************************************************************************
//                                                                  Allocation
// *****************************************************************************
struct allocation {
  /// Allocates **size** bytes of memory aligned by **align** bytes.
  /// \param size **[in]** memory size in bytes
  /// \param align **[in]** number of bytes of alignment
  /// \return pointer to allocated memory
  static void *allocAligned(h_size size, h_size align);
  /// Frees memory allocated by allocAligned function
  /// \param p_mem pointer to aligned memory block
  static void freeAligned(void *p_mem);
  ///
  HERMES_NODISCARD static HeError freeMemory(void *data,
                                             MemoryLocation memory_location);

  HERMES_NODISCARD static Result<void *>
  allocate(h_size byte_count, MemoryLocation memory_location);

  HERMES_NODISCARD static Result<std::tuple<void *, h_size>>
  allocate(const size2 &size, MemoryLocation memory_location);

  HERMES_NODISCARD static Result<std::tuple<void *, h_size>>
  allocate(const size3 &size, MemoryLocation memory_location);
};

// *****************************************************************************
//                                                                     Writing
// *****************************************************************************
struct writes {

  // linear memory

  HERMES_NODISCARD static HeError copyDevice2Device(void *dst, void *src,
                                                    h_size byte_count);

  HERMES_NODISCARD static HeError copyDevice2Host(void *dst, void *src,
                                                  h_size byte_count);

  HERMES_NODISCARD static HeError copyHost2Device(void *dst, void *src,
                                                  h_size byte_count);

  HERMES_NODISCARD static HeError copyHost2Host(void *dst, void *src,
                                                h_size byte_count);

  HERMES_NODISCARD static HeError copy(MemoryLocation dst_location, void *dst,
                                       MemoryLocation src_location, void *src,
                                       h_size byte_count);

  // 2d memory

  HERMES_NODISCARD static HeError copyHost2Device(void *dst, h_size dst_pitch,
                                                  void *src, h_size src_pitch,
                                                  const size2 &src_size);

  HERMES_NODISCARD static HeError copyDevice2Host(void *dst, h_size dst_pitch,
                                                  void *src, h_size src_pitch,
                                                  const size2 &src_size);

  HERMES_NODISCARD static HeError copyDevice2Device(void *dst, h_size dst_pitch,
                                                    void *src, h_size src_pitch,
                                                    const size2 &src_size);

  HERMES_NODISCARD static HeError copy(MemoryLocation dst_location, void *dst,
                                       h_size dst_pitch,
                                       MemoryLocation src_location, void *src,
                                       h_size src_pitch, const size2 &src_size);

  // 3d memory

  HERMES_NODISCARD static HeError copyHost2Device(void *dst, h_size dst_pitch,
                                                  const size3 &dst_size,
                                                  void *src, h_size src_pitch,
                                                  const size3 &src_size);

  HERMES_NODISCARD static HeError copyDevice2Host(void *dst, h_size dst_pitch,
                                                  const size3 &dst_size,
                                                  void *src, h_size src_pitch,
                                                  const size3 &src_size);

  HERMES_NODISCARD static HeError copyDevice2Device(void *dst, h_size dst_pitch,
                                                    const size3 &dst_size,
                                                    void *src, h_size src_pitch,
                                                    const size3 &src_size);

  HERMES_NODISCARD static HeError copy(MemoryLocation dst_location, void *dst,
                                       h_size dst_pitch, const size3 &dst_size,
                                       MemoryLocation src_location, void *src,
                                       h_size src_pitch, const size3 &src_size);
};

} // namespace hermes::mem
