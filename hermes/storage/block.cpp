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

/// \file   memory_block.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2021-09-22

#include <hermes/storage/block.h>

namespace hermes::mem {

Block::Config &Block::Config::setLocation(MemoryLocation memory_location) {
  location_ = memory_location;
  return *this;
}

Block::Config &Block::Config::setSize(h_size size) {
  size_.width = size;
  size_.height = 1;
  size_.depth = 1;
  dimensions_ = 1;
  return *this;
}

Block::Config &Block::Config::setSize(const size2 &size) {
  size_.width = size.width;
  size_.height = size.height;
  size_.depth = 1;
  dimensions_ = 2;
  return *this;
}

Block::Config &Block::Config::setSize(const size3 &size) {
  size_ = size;
  dimensions_ = 3;
  return *this;
}

Block::Config &Block::Config::setPitch(h_size pitch_size) {
  pitch_ = pitch_size;
  return *this;
}

Result<Block> Block::Config::create() const {
  Block block;
  block.location_ = location_;
  if (dimensions_ == 3) {
    HERMES_RETURN_BAD_RESULT(block.resize(size_, pitch_));
  } else if (dimensions_ == 2) {
    HERMES_RETURN_BAD_RESULT(block.resize(size_.slice(0, 1), pitch_));
  } else {
    HERMES_RETURN_BAD_RESULT(block.resize(size_.total()));
  }
  return Result<Block>(std::move(block));
}

Block::Block(const Block &rhs) { *this = rhs; }

Block::Block(Block &&rhs) noexcept { *this = std::move(rhs); }

Block::~Block() noexcept { HERMES_CHECK_HE_RESULT(clear()); }

Block &Block::operator=(const Block &rhs) {
  HERMES_CHECK_HE_RESULT(clear());
  auto err = resize(rhs.size_, rhs.pitch_);
  if (err != HeError::None) {
    HERMES_ERROR("Could not copy assign memory block.");
    return *this;
  }
  HERMES_CHECK_HE_RESULT(copy(rhs));
  return *this;
}

Block &Block::operator=(Block &&rhs) noexcept {
  HERMES_CHECK_HE_RESULT(clear());
  swap(rhs);
  return *this;
}

HeError Block::clear() noexcept {
  auto err = allocation::freeMemory(data_, location_);
  if (err == HeError::None) {
    data_ = nullptr;
    size_ = {0, 0, 0};
  }
  return err;
}

void Block::swap(Block &rhs) {
  std::swap(location_, rhs.location_);
  std::swap(size_, rhs.size_);
  std::swap(pitch_, rhs.pitch_);
  std::swap(data_, rhs.data_);
}

HeError Block::resize(h_size new_size_in_bytes) {
  if (new_size_in_bytes == 0) {
    HERMES_RETURN_HE_ERROR(clear());
    size_ = {0, 0, 0};
    return HeError::None;
  }
  size3 new_size(new_size_in_bytes, 1, 1);
  if (size_ == new_size)
    return HeError::None;
  HERMES_RETURN_HE_ERROR(clear());
  HERMES_ASSIGN_OR_RETURN_HE_ERROR(
      data_, allocation::allocate(new_size_in_bytes, location_));
  size_ = new_size;
  pitch_ = new_size_in_bytes;
  return HeError::None;
}

HeError Block::resize(const size2 &new_size, h_size new_pitch) {
  auto s3 = size3(new_size.width, new_size.height, 1);
  if (size_ == s3 && pitch_ == new_pitch)
    return HeError::None;
  HERMES_RETURN_HE_ERROR(clear());
  HERMES_ASSIGN_OR_RETURN_HE_ERROR(std::tie(data_, pitch_),
                                   allocation::allocate(new_size, location_));
  size_ = s3;
  return HeError::None;
}

HeError Block::resize(const size3 &new_size, h_size new_pitch) {
  if (size_ == new_size && pitch_ == new_pitch)
    return HeError::None;
  HERMES_RETURN_HE_ERROR(clear());
  HERMES_ASSIGN_OR_RETURN_HE_ERROR(std::tie(data_, pitch_),
                                   allocation::allocate(new_size, location_));
  size_ = new_size;
  return HeError::None;
}

HeError Block::copy(const Block &memory_block) {
  HERMES_ASSERT(sizeInBytes() >= memory_block.sizeInBytes());
  return writes::copy(location_, data_, pitch_, size_, memory_block.location_,
                      memory_block.data_, memory_block.pitch_,
                      memory_block.size_);
}

HeError Block::copy(void *data, h_size size_in_bytes, h_size offset,
                    MemoryLocation data_location) {
  HERMES_ASSERT(offset == 0);
  HERMES_UNUSED_VARIABLE(offset);
  HERMES_ASSERT(sizeInBytes() >= size_in_bytes);
  return writes::copy(location_, data_, data_location, data, size_in_bytes);
}

h_size Block::sizeInBytes() const {
  return pitch_ * size_.height * size_.depth;
}

h_size Block::pitch() const { return pitch_; }

const size3 &Block::size() const { return size_; }

void *Block::data() { return data_; }

const void *Block::data() const { return data_; }

h_byte *Block::bytes() { return reinterpret_cast<h_byte *>(data_); }

const h_byte *Block::bytes() const {
  return reinterpret_cast<const h_byte *>(data_);
}

void *Block::operator*() { return data_; }

const void *Block::operator*() const { return data_; }

MemoryLocation Block::location() const { return location_; }

} // namespace hermes::mem
