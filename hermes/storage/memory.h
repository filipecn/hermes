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
///\file address_index.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-09-22
///
///\brief

#ifndef HERMES_HERMES_STORAGE_MEMORY_H
#define HERMES_HERMES_STORAGE_MEMORY_H

#include <hermes/common/defs.h>
#include <hermes/common/debug.h>

namespace hermes {

/// Object returned by memory allocators and other memory-related classes
/// Each class puts a meaning into its value
struct AddressIndex {
  HERMES_DEVICE_CALLABLE AddressIndex(std::size_t id = 0) : id(id) {}
  /// handle identifier, a value of zero identifies an invalid address
  std::size_t id{0};
  [[nodiscard]] HERMES_DEVICE_CALLABLE inline bool isValid() const { return id != 0; }
};

/// Memory Manager Singleton
/// This class is responsible for managing all memory used in the system by
/// allocating all memory first and controlling how the allocated memory is
/// used.
class mem {
public:
  /****************************************************************************
                               STATIC PUBLIC FIELDS
  ****************************************************************************/
  [[maybe_unused]] static u32 cache_l1_size;
  /****************************************************************************
                               INLINE STATIC METHODS
  ****************************************************************************/
  /// \param number_of_bytes
  /// \param align alignment size in number of bytes
  /// \return the actual amount of bytes necessary to store number_of_bytes
  /// under the alignment
  HERMES_DEVICE_CALLABLE static inline std::size_t alignTo(std::size_t number_of_bytes, std::size_t align) {
    return number_of_bytes > 0 ? (1u + (number_of_bytes - 1u) / align) * align : 0;
  }
  /// \param address
  /// \param align
  /// \return
  HERMES_DEVICE_CALLABLE static inline std::size_t leftAlignShift(uintptr_t address, std::size_t align) {
    const std::size_t mask = align - 1;
    HERMES_ASSERT((align & mask) == 0);
    return address - (address & ~mask);
  }
  /// \param address
  /// \param align
  /// \return
  HERMES_DEVICE_CALLABLE static inline std::size_t rightAlignShift(uintptr_t address, std::size_t align) {
    const std::size_t mask = align - 1;
    HERMES_ASSERT((align & mask) == 0);
    return ((address + mask) & ~mask) - address;
  }
  /// Shifts **address** upwards if necessary to ensure it is aligned to
  /// **align** number of bytes.
  /// \param address **[in]** memory address
  /// \param align **[in]** number of bytes
  /// \return aligned address
  HERMES_DEVICE_CALLABLE static inline uintptr_t alignAddress(uintptr_t address, std::size_t align) {
    const std::size_t mask = align - 1;
    HERMES_ASSERT((align & mask) == 0);
    return (address + mask) & ~mask;
  }
  /// Shifts pointer **ptr** upwards if necessary to ensure it is aligned to
  /// **align** number of bytes.
  /// \tparam T data type
  /// \param ptr **[in]** pointer
  /// \param align **[in]** number of bytes
  /// \return aligned pointer
  template<typename T>
  HERMES_DEVICE_CALLABLE static inline T *alignPointer(T *ptr, std::size_t align) {
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t addr_aligned = alignAddress(addr, align);
    return reinterpret_cast<T *>(addr_aligned);
  }
  /// Allocates **size** bytes of memory aligned by **align** bytes.
  /// \param size **[in]** memory size in bytes
  /// \param align **[in]** number of bytes of alignment
  /// \return pointer to allocated memory
  static void *allocAligned(std::size_t size, std::size_t align);
  /// Frees memory allocated by allocAligned function
  /// \param p_mem pointer to aligned memory block
  static void freeAligned(void *p_mem);
};

}

#endif //HERMES_HERMES_STORAGE_MEMORY_H
