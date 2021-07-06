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
///\file memory_block.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-29
///
///\brief

#ifndef HERMES_HERMES_STORAGE_MEMORY_BLOCK_H
#define HERMES_HERMES_STORAGE_MEMORY_BLOCK_H

#include <hermes/common/debug.h>
#include <hermes/common/size.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                        MemoryBlock
// *********************************************************************************************************************
/// The MemoryBlock is a generic allocated region of memory.
/// The block can be allocated from device memory as well when CUDA is enabled.
/// \tparam L memory location
template<MemoryLocation L>
class MemoryBlock {};

// *********************************************************************************************************************
//                                                                                                   HOST MemoryBlock
// *********************************************************************************************************************
template<>
class MemoryBlock<MemoryLocation::HOST> {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend MemoryBlock<MemoryLocation::DEVICE>;
  friend MemoryBlock<MemoryLocation::UNIFIED>;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  //                                                                                                              new
  MemoryBlock();
  /// \param size_in_bytes
  explicit MemoryBlock(size_t size_in_bytes);
  /// \param size width in bytes
  explicit MemoryBlock(size2 size);
  /// \param size width in bytes
  explicit MemoryBlock(size3 size);
  //                                                                                                           delete
  virtual ~MemoryBlock();
  //                                                                                                       assignment
  MemoryBlock(const MemoryBlock<MemoryLocation::HOST> &other);
  MemoryBlock(const MemoryBlock<MemoryLocation::DEVICE> &other);
  MemoryBlock(MemoryBlock<MemoryLocation::HOST> &&other) noexcept;
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  MemoryBlock &operator=(MemoryBlock<MemoryLocation::HOST> &&other) noexcept;
  MemoryBlock &operator=(const MemoryBlock<MemoryLocation::HOST> &other);
  MemoryBlock &operator=(const MemoryBlock<MemoryLocation::DEVICE> &other);
  //                                                                                                       arithmetic
  //                                                                                                          boolean
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                             size
  [[nodiscard]] size_t sizeInBytes() const;
  [[nodiscard]] size_t pitch() const;
  size3 size() const;
  /// \param new_size_in_bytes
  void resize(size_t new_size_in_bytes);
  /// \param new_size width in bytes
  void resize(size2 new_size, size_t new_pitch = 0);
  /// \param new_size width in bytes
  void resize(size3 new_size, size_t new_pitch = 0);
  ///
  void clear();
  //                                                                                                           access
  ///
  byte *ptr();
  ///
  [[nodiscard]] const byte *ptr() const;
  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  const MemoryLocation location{MemoryLocation::HOST};
private:
  size3 size_;
  size_t pitch_{0};
  mutable byte *data_{nullptr};
};

#ifdef ENABLE_CUDA
// *********************************************************************************************************************
//                                                                                                 DEVICE MemoryBlock
// *********************************************************************************************************************
template<>
class MemoryBlock<MemoryLocation::DEVICE> {
public:
  // *******************************************************************************************************************
  //                                                                                                   FRIEND STRUCTS
  // *******************************************************************************************************************
  friend MemoryBlock<MemoryLocation::HOST>;
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  //                                                                                                              new
  MemoryBlock();
  /// \param size_in_bytes
  explicit MemoryBlock(size_t size_in_bytes);
  /// \param size width in bytes
  explicit MemoryBlock(size2 size);
  /// \param size width in bytes
  explicit MemoryBlock(size3 size);
  //                                                                                                           delete
  virtual ~MemoryBlock();
  //                                                                                                       assignment
  MemoryBlock(const MemoryBlock<MemoryLocation::HOST> &other);
  MemoryBlock(const MemoryBlock<MemoryLocation::DEVICE> &other);
  explicit MemoryBlock(MemoryBlock<MemoryLocation::DEVICE> &&other) noexcept;
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  MemoryBlock &operator=(MemoryBlock<MemoryLocation::DEVICE> &&other) noexcept;
  MemoryBlock &operator=(const MemoryBlock<MemoryLocation::DEVICE> &other);
  MemoryBlock &operator=(const MemoryBlock<MemoryLocation::HOST> &other);
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                             size
  /// \return total size in bytes
  [[nodiscard]] size_t sizeInBytes() const;
  [[nodiscard]] size_t pitch() const;
  size3 size() const;
  /// \param new_size_in_bytes
  void resize(size_t new_size_in_bytes);
  /// \param new_size width in bytes
  void resize(size2 new_size);
  /// \param new_size width in bytes
  void resize(size3 new_size);
  ///
  void clear();
  //                                                                                                           access
  ///
  byte *ptr();
  /// \return
  [[nodiscard]] const byte *ptr() const;

  /// \return
  cudaPitchedPtr pitchedData() {
    cudaPitchedPtr pd{};
    pd.ptr = data_;
    pd.pitch = pitch_;
    pd.xsize = size_.width;
    pd.ysize = size_.height;
    return pd;
  }

  // *******************************************************************************************************************
  //                                                                                                    PUBLIC FIELDS
  // *******************************************************************************************************************
  const MemoryLocation location{MemoryLocation::DEVICE};
private:
  size3 size_;
  size_t pitch_{0};
  mutable byte *data_{nullptr};
};

// *********************************************************************************************************************
//                                                                                                UNIFIED MemoryBlock
// *********************************************************************************************************************
template<>
class MemoryBlock<MemoryLocation::UNIFIED> {
public:
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  //                                                                                                              new
  MemoryBlock();
  ~MemoryBlock();
  /// \param size_in_bytes
  explicit MemoryBlock(size_t size_in_bytes);
  //                                                                                                       assignment
  MemoryBlock(const MemoryBlock<MemoryLocation::HOST> &other);
  MemoryBlock(const MemoryBlock<MemoryLocation::DEVICE> &other);
  MemoryBlock(MemoryBlock<MemoryLocation::UNIFIED> &&other) noexcept;
  // *******************************************************************************************************************
  //                                                                                                        OPERATORS
  // *******************************************************************************************************************
  //                                                                                                       assignment
  MemoryBlock &operator=(const MemoryBlock<MemoryLocation::HOST> &other);
  MemoryBlock &operator=(const MemoryBlock<MemoryLocation::DEVICE> &other);
  MemoryBlock &operator=(const MemoryBlock<MemoryLocation::UNIFIED> &other);
  MemoryBlock &operator=(MemoryBlock<MemoryLocation::UNIFIED> &&other) noexcept;
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  //                                                                                                             size
  /// \return total size in bytes
  [[nodiscard]] size_t sizeInBytes() const;
  /// \param new_size_in_bytes
  void resize(size_t new_size_in_bytes);
  ///
  void clear();
  //                                                                                                           access
  ///
  byte *ptr();
  ///
  [[nodiscard]] const byte *ptr() const;
private:
  size3 size_;
  size_t pitch_{0};
  mutable byte *data_{nullptr};
};

#endif
// *********************************************************************************************************************
//                                                                                                                 IO
// *********************************************************************************************************************
template<MemoryLocation L>
std::ostream &operator<<(std::ostream &o, const MemoryBlock<L> &m) {
  o << "[MemoryBlock][" << L << "] total size (bytes):" << m.sizeInBytes() << " with layout ";
  o << "(" << m.size().width << " bytes, " << m.size().height << ", " << m.size().depth;
  o << ") and pitch: " << m.pitch() << " bytes";
  return o;
}

// *********************************************************************************************************************
//                                                                                                           TYPEDEFS
// *********************************************************************************************************************
using HostMemory = MemoryBlock<MemoryLocation::HOST>;
using DeviceMemory = MemoryBlock<MemoryLocation::DEVICE>;
using UnifiedMemory = MemoryBlock<MemoryLocation::UNIFIED>;

}

#endif //HERMES_HERMES_STORAGE_MEMORY_BLOCK_H
