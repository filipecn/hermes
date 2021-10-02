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
///\file stack_allocator.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-09-22
///
///\brief

#ifndef HERMES_HERMES_STORAGE_STACK_ALLOCATOR_H
#define HERMES_HERMES_STORAGE_STACK_ALLOCATOR_H

#include <hermes/storage/memory.h>
#include <hermes/storage/memory_block.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                        Memory Stack Allocator View
// *********************************************************************************************************************
/// Stack Allocator Accessor
/// To be used mainly by GPU code
class StackAllocatorView {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  HERMES_DEVICE_CALLABLE StackAllocatorView(byte *data, std::size_t capacity_in_bytes, std::size_t marker = 0);
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                             size
  /// \return available size that can be allocated
  [[nodiscard]] HERMES_DEVICE_CALLABLE std::size_t availableSizeInBytes() const;
  //                                                                                                       allocation
  /// Allocates a new block from stack top
  /// \param block_size_in_bytes
  /// \return pointer to the new allocated block
  HERMES_DEVICE_CALLABLE AddressIndex allocate(std::size_t block_size_in_bytes, std::size_t align = 1);
  ///
  /// \tparam T
  /// \tparam P
  /// \param params
  /// \return
  template<typename T, class... P>
  HERMES_DEVICE_CALLABLE AddressIndex allocateAligned(P &&... params) {
    auto handle = allocate(sizeof(T), alignof(T));
    if (!handle.id)
      return handle;
    T *ptr = reinterpret_cast<T *>(data_ + ((handle.id & 0xffffff) - 1));
    new(ptr) T(std::forward<P>(params)...);
    return handle;
  }
  ///
  /// \tparam T
  /// \param handle
  /// \param value
  /// \return
  template<typename T>
  HERMES_DEVICE_CALLABLE HeResult set(AddressIndex handle, const T &value) {
    if (handle.id == 0 || handle.id >= capacity_in_bytes)
      return HeResult::INVALID_INPUT;
    *reinterpret_cast<T *>(data_ + ((handle.id & 0xffffff) - 1)) = value;
    return HeResult::SUCCESS;
  }
  ///
  /// \tparam T
  /// \param handle
  /// \return
  template<typename T>
  HERMES_DEVICE_CALLABLE  T *get(AddressIndex handle) {
    return reinterpret_cast<T *>(data_ + ((handle.id & 0xffffff) - 1));
  }

  /// Roll the stack back to a previous marker point
  /// \param marker
  HERMES_DEVICE_CALLABLE HeResult freeTo(AddressIndex handle);
  /// Roll stack back to zero
  HERMES_DEVICE_CALLABLE void clear();
  /// Total stack capacity in bytes
  const std::size_t capacity_in_bytes;
private:
  byte *data_{nullptr};
  std::size_t marker_{0};
};

// *********************************************************************************************************************
//                                                                                             Memory Stack Allocator
// *********************************************************************************************************************
/// RAII Stack Allocator
///
/// \note Handle Construction:
/// \note The most significant byte of the handle is used to store the alignment
/// shift and the rest is used to store the address offset + 1 of the first
/// byte of the allocated block. Suppose the alignment requires a shift of
/// 3 bytes and the allocated block would start at byte with offset 10.
/// A 32 bit handle id in this case will have the value of 0x3000000A.
template<MemoryLocation L>
class MemoryStackAllocator {};

// *********************************************************************************************************************
//                                                                                        HOST Memory Stack Allocator
// *********************************************************************************************************************
template<>
class MemoryStackAllocator<MemoryLocation::HOST> {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend MemoryStackAllocator<MemoryLocation::DEVICE>;
  friend MemoryStackAllocator<MemoryLocation::UNIFIED>;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param size_in_bytes
  explicit MemoryStackAllocator(std::size_t size_in_bytes = 0);
  explicit MemoryStackAllocator(std::size_t size_in_bytes, byte *buffer);
  ///
  ~MemoryStackAllocator();
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                             view
  ///
  /// \return
  StackAllocatorView view();
  //                                                                                                             size
  /// \return total stack capacity (in bytes)
  [[nodiscard]] std::size_t capacityInBytes() const;
  /// \return available size that can be allocated
  [[nodiscard]] std::size_t availableSizeInBytes() const;
  /// All previous data is deleted and markers get invalid
  /// \param size_in_bytes total memory capacity
  HeResult resize(std::size_t size_in_bytes);
  //                                                                                                       allocation
  /// Allocates a new block from stack top
  /// \param block_size_in_bytes
  /// \return pointer to the new allocated block
  AddressIndex allocate(std::size_t block_size_in_bytes, std::size_t align = 1);
  ///
  /// \tparam T
  /// \tparam P
  /// \param params
  /// \return
  template<typename T, class... P>
  AddressIndex allocateAligned(P &&... params) {
    auto handle = allocate(sizeof(T), alignof(T));
    if (!handle.id)
      return handle;
    T *ptr = reinterpret_cast<T *>(data_ + ((handle.id & 0xffffff) - 1));
    new(ptr) T(std::forward<P>(params)...);
    return handle;
  }
  ///
  /// \tparam T
  /// \param handle
  /// \param value
  /// \return
  template<typename T>
  HeResult set(AddressIndex handle, const T &value) {
    if (handle.id == 0 || handle.id >= capacity_)
      return HeResult::INVALID_INPUT;
    *reinterpret_cast<T *>(data_ + ((handle.id & 0xffffff) - 1)) = value;
    return HeResult::SUCCESS;
  }
  ///
  /// \tparam T
  /// \param handle
  /// \param value
  /// \return
//  template<typename T>
//  HeResult set(AddressIndex handle, T &&value) {
//    if (handle.id == 0 || handle.id >= capacity_)
//      return HeResult::INVALID_INPUT;
//    T* ptr = reinterpret_cast<T *>(data_);// + ((handle.id & 0xffffff) - 1));// = std::forward<T>(value);
//    return HeResult::SUCCESS;
//  }
  ///
  /// \tparam T
  /// \param handle
  /// \return
  template<typename T>
  T *get(AddressIndex handle) {
    HERMES_ASSERT(handle.id > 0 && ((handle.id & 0xffffff) - 1) < capacity_)
    return reinterpret_cast<T *>(data_ + ((handle.id & 0xffffff) - 1));
  }

  /// Roll the stack back to a previous marker point
  /// \param marker
  HeResult freeTo(AddressIndex handle);
  /// Roll stack back to zero
  void clear();

private:
  MemoryBlock<MemoryLocation::HOST> mem_block_;
  byte *data_{nullptr};
  std::size_t capacity_{0};
  std::size_t marker_{0};
  bool using_external_memory_{false};
};

// *********************************************************************************************************************
//                                                                                     UNIFIED Memory Stack Allocator
// *********************************************************************************************************************
template<>
class MemoryStackAllocator<MemoryLocation::UNIFIED> {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend MemoryStackAllocator<MemoryLocation::DEVICE>;
  friend MemoryStackAllocator<MemoryLocation::HOST>;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param size_in_bytes
  explicit MemoryStackAllocator(std::size_t size_in_bytes = 0);
  explicit MemoryStackAllocator(std::size_t size_in_bytes, byte *buffer);
  ///
  ~MemoryStackAllocator();
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  ///
  /// \return
  StackAllocatorView view();
  //                                                                                                             size
  /// \return total stack capacity (in bytes)
  [[nodiscard]] std::size_t capacityInBytes() const;
  /// \return available size that can be allocated
  [[nodiscard]] std::size_t availableSizeInBytes() const;
  /// All previous data is deleted and markers get invalid
  /// \param size_in_bytes total memory capacity
  HeResult resize(std::size_t size_in_bytes);
  //                                                                                                       allocation
  /// Allocates a new block from stack top
  /// \param block_size_in_bytes
  /// \return pointer to the new allocated block
  AddressIndex allocate(std::size_t block_size_in_bytes, std::size_t align = 1);
  ///
  /// \tparam T
  /// \tparam P
  /// \param params
  /// \return
  template<typename T, class... P>
  AddressIndex allocateAligned(P &&... params) {
    auto handle = allocate(sizeof(T), alignof(T));
    if (!handle.id)
      return handle;
    T *ptr = reinterpret_cast<T *>(data_ + ((handle.id & 0xffffff) - 1));
    new(ptr) T(std::forward<P>(params)...);
    return handle;
  }
  ///
  /// \tparam T
  /// \param handle
  /// \param value
  /// \return
  template<typename T>
  HeResult set(AddressIndex handle, const T &value) {
    if (handle.id == 0 || handle.id >= capacity_)
      return HeResult::INVALID_INPUT;
    *reinterpret_cast<T *>(data_ + ((handle.id & 0xffffff) - 1)) = value;
    return HeResult::SUCCESS;
  }
  ///
  /// \tparam T
  /// \param handle
  /// \param value
  /// \return
//  template<typename T>
//  HeResult set(AddressIndex handle, T &&value) {
//    if (handle.id == 0 || handle.id >= capacity_)
//      return HeResult::INVALID_INPUT;
//    T* ptr = reinterpret_cast<T *>(data_);// + ((handle.id & 0xffffff) - 1));// = std::forward<T>(value);
//    return HeResult::SUCCESS;
//  }
  ///
  /// \tparam T
  /// \param handle
  /// \return
  template<typename T>
  T *get(AddressIndex handle) {
    HERMES_ASSERT(handle.id > 0 && ((handle.id & 0xffffff) - 1) < capacity_)
    return reinterpret_cast<T *>(data_ + ((handle.id & 0xffffff) - 1));
  }

  /// Roll the stack back to a previous marker point
  /// \param marker
  HeResult freeTo(AddressIndex handle);
  /// Roll stack back to zero
  void clear();

private:
  MemoryBlock<MemoryLocation::UNIFIED> mem_block_;
  byte *data_{nullptr};
  std::size_t capacity_{0};
  std::size_t marker_{0};
  bool using_external_memory_{false};
};

// *********************************************************************************************************************
//                                                                                      DEVICE Memory Stack Allocator
// *********************************************************************************************************************
template<>
class MemoryStackAllocator<MemoryLocation::DEVICE> {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend MemoryStackAllocator<MemoryLocation::UNIFIED>;
  friend MemoryStackAllocator<MemoryLocation::HOST>;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  /// \param size_in_bytes
  explicit MemoryStackAllocator(std::size_t size_in_bytes = 0);
  explicit MemoryStackAllocator(std::size_t size_in_bytes, byte *buffer);
  MemoryStackAllocator(const MemoryStackAllocator<MemoryLocation::HOST>& other);
  ///
  ~MemoryStackAllocator();
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                             view
  ///
  /// \return
  StackAllocatorView view();
  //                                                                                                             size
  /// \return total stack capacity (in bytes)
  [[nodiscard]] std::size_t capacityInBytes() const;
  /// All previous data is deleted and markers get invalid
  /// \param size_in_bytes total memory capacity
  HeResult resize(std::size_t size_in_bytes);

private:
  MemoryBlock<MemoryLocation::DEVICE> mem_block_;
  byte *data_{nullptr};
  std::size_t capacity_{0};
  bool using_external_memory_{false};
};

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using StackAllocator = MemoryStackAllocator<MemoryLocation::HOST>;
using DeviceStackAllocator = MemoryStackAllocator<MemoryLocation::DEVICE>;
using UnifiedStackAllocator = MemoryStackAllocator<MemoryLocation::UNIFIED>;
}

#endif //HERMES_HERMES_STORAGE_STACK_ALLOCATOR_H
