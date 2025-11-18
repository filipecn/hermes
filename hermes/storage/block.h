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

/// \file   memory_block.h
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2021-09-22

#pragma once

#include <hermes/storage/memory.h>

namespace hermes::mem {

// *****************************************************************************
//                                                                 Block
// *****************************************************************************
/// The Block is a generic allocated region of memory.
/// \note The block can be allocated from device memory as well when CUDA is
///       enabled.
class Block {
public:
  struct Config {
    Config &setLocation(MemoryLocation memory_location);
    Config &setSize(h_size size_in_bytes);
    Config &setSize(const size2 &size);
    Config &setSize(const size3 &size);
    Config &setPitch(h_size pitch_size);

    Result<Block> create() const;

  private:
    MemoryLocation location_{MemoryLocation::HOST};
    size3 size_;
    h_size pitch_{0};
    u32 dimensions_{1};
  };

  Block() noexcept = default;
  Block(const Block &rhs);
  Block(Block &&rhs) noexcept;
  ~Block() noexcept;
  Block &operator=(const Block &rhs);
  Block &operator=(Block &&rhs) noexcept;

  /// \param new_size_in_bytes
  HERMES_NODISCARD HeError resize(h_size new_size_in_bytes);
  /// \param new_size width in bytes
  HERMES_NODISCARD HeError resize(const size2 &new_size, h_size new_pitch = 0);
  /// \param new_size width in bytes
  HERMES_NODISCARD HeError resize(const size3 &new_size, h_size new_pitch = 0);
  /// Delete memory block.
  HERMES_NODISCARD HeError clear() noexcept;
  /// Swap object data.
  void swap(Block &rhs);

  /// Copy from other block
  /// \param memory_block
  HERMES_NODISCARD HeError copy(const Block &memory_block);
  /// Copy content from data
  /// \param data
  /// \param size_in_bytes
  /// \param offset offset into memory block
  /// \param data_location
  HERMES_NODISCARD HeError
  copy(void *data, h_size size_in_bytes, h_size offset,
       MemoryLocation data_location = MemoryLocation::HOST);
  /// \tparam T
  /// \param data
  /// \param offset offset into memory block
  template <typename T>
  HERMES_NODISCARD HeError
  copy(T *data, h_size offset = 0,
       MemoryLocation data_location = MemoryLocation::HOST) {
    return copy(reinterpret_cast<void *>(data), sizeof(T), offset,
                data_location);
  }

  h_size sizeInBytes() const;
  h_size pitch() const;
  const size3 &size() const;
  void *data();
  const void *data() const;
  h_byte *bytes();
  const h_byte *bytes() const;
  void *operator*();
  const void *operator*() const;
  MemoryLocation location() const;

private:
  MemoryLocation location_{MemoryLocation::HOST};
  size3 size_;
  h_size pitch_{0};
  mutable void *data_{nullptr};

  HERMES_TO_STRING_FRIEND(Block);
};

} // namespace hermes::mem

namespace hermes {
HERMES_DECLARE_TO_STRING_DEBUG_METHOD(mem::Block);
} // namespace hermes
