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
///\file memory_block.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-30
///
///\brief

#include <hermes/storage/memory_block.h>
#include <hermes/common/debug.h>
#ifdef HERMES_DEVICE_ENABLED
#include <hermes/common/cuda_utils.h>
#endif

namespace hermes {

// *********************************************************************************************************************
//                                                                                                   HOST MemoryBlock
// *********************************************************************************************************************

MemoryBlock<MemoryLocation::HOST>::MemoryBlock() = default;

MemoryBlock<MemoryLocation::HOST>::MemoryBlock(size_t size_in_bytes) {
  resize(size_in_bytes);
}

MemoryBlock<MemoryLocation::HOST>::MemoryBlock(size2 size) {
  resize(size, 0);
}

MemoryBlock<MemoryLocation::HOST>::MemoryBlock(size3 size) {
  resize(size, 0);
}

MemoryBlock<MemoryLocation::HOST>::~MemoryBlock() {
  clear();
}

MemoryBlock<MemoryLocation::HOST>::MemoryBlock(const MemoryBlock<MemoryLocation::HOST> &other) {
  *this = other;
}

MemoryBlock<MemoryLocation::HOST>::MemoryBlock(const MemoryBlock<MemoryLocation::DEVICE> &other) {
  *this = other;
}

MemoryBlock<MemoryLocation::HOST>::MemoryBlock(MemoryBlock<MemoryLocation::HOST> &&other) noexcept {
  *this = std::move(other);
}

MemoryBlock<MemoryLocation::HOST> &
MemoryBlock<MemoryLocation::HOST>::operator=(MemoryBlock<MemoryLocation::HOST> &&other) noexcept {
  if (this == &other)
    return *this;
  clear();
  size_ = other.size_;
  pitch_ = other.pitch_;
  data_ = other.data_;
  other.data_ = nullptr;
  other.size_ = {0, 0, 0};
  other.pitch_ = 0;
  return *this;
}

MemoryBlock<MemoryLocation::HOST> &
MemoryBlock<MemoryLocation::HOST>::operator=(const MemoryBlock<MemoryLocation::DEVICE> &other) {
#ifdef HERMES_DEVICE_ENABLED
  resize(other.size_, other.pitch_);
  HERMES_CHECK_CUDA(cudaMemcpy(data_, other.data_, other.sizeInBytes(), cudaMemcpyDeviceToHost));
  size_ = other.size_;
  pitch_ = other.pitch_;
#endif
  return *this;
}

MemoryBlock<MemoryLocation::HOST> &
MemoryBlock<MemoryLocation::HOST>::operator=(const MemoryBlock<MemoryLocation::HOST> &other) {
  if (this == &other)
    return *this;
  resize(other.size_, other.pitch_);
  std::memcpy(data_, other.data_, other.sizeInBytes());
  return *this;
}

size_t MemoryBlock<MemoryLocation::HOST>::sizeInBytes() const {
  return pitch_ * size_.height * size_.depth;
}

size_t MemoryBlock<MemoryLocation::HOST>::pitch() const {
  return pitch_;
}

size3 MemoryBlock<MemoryLocation::HOST>::size() const {
  return size_;
}

void MemoryBlock<MemoryLocation::HOST>::resize(size_t new_size_in_bytes) {
  size3 new_size(new_size_in_bytes, 1, 1);
  if (size_ == new_size)
    return;
  clear();
  size_ = {static_cast<u32>(new_size_in_bytes), 1, 1};
  pitch_ = new_size_in_bytes;
  data_ = new u8[this->sizeInBytes()];
}

void MemoryBlock<MemoryLocation::HOST>::resize(size2 new_size, size_t new_pitch) {
  if (size_ == size3(new_size.width, new_size.height, 1) && pitch_ == new_pitch)
    return;
  clear();
  pitch_ = new_pitch;
  size_ = {new_size.width, new_size.height, 1};
  if (pitch_ == 0)
    pitch_ = size_.width;
  data_ = new u8[this->sizeInBytes()];
}

void MemoryBlock<MemoryLocation::HOST>::resize(size3 new_size, size_t new_pitch) {
  if (size_ == new_size && pitch_ == new_pitch)
    return;
  clear();
  pitch_ = new_pitch;
  size_ = new_size;
  if (pitch_ == 0)
    pitch_ = size_.width;
  data_ = new u8[this->sizeInBytes()];
}

void MemoryBlock<MemoryLocation::HOST>::clear() {
  delete[] data_;
  size_ = {0, 0, 0};
  pitch_ = 0;
  data_ = nullptr;
}

byte *MemoryBlock<MemoryLocation::HOST>::ptr() {
  return data_;
}

const byte *MemoryBlock<MemoryLocation::HOST>::ptr() const {
  return data_;
}

void MemoryBlock<MemoryLocation::HOST>::copy(const void *data,
                                             size_t size_in_bytes,
                                             size_t offset,
                                             MemoryLocation data_location) {
  HERMES_CHECK_EXP(size_in_bytes <= sizeInBytes() - offset)
  if (data_location != MemoryLocation::DEVICE)
    std::memcpy(data_ + offset, data, size_in_bytes);
  else {
#ifdef HERMES_DEVICE_ENABLED
    HERMES_NOT_IMPLEMENTED
#else
    HERMES_NOT_IMPLEMENTED
#endif
  }
}

// *********************************************************************************************************************
//                                                                                                 DEVICE MemoryBlock
// *********************************************************************************************************************

MemoryBlock<MemoryLocation::DEVICE>::MemoryBlock() = default;

MemoryBlock<MemoryLocation::DEVICE>::MemoryBlock(size_t size_in_bytes) {
  resize(size_in_bytes);
}

MemoryBlock<MemoryLocation::DEVICE>::~MemoryBlock() {
  clear();
}

MemoryBlock<MemoryLocation::DEVICE>::MemoryBlock(const MemoryBlock<MemoryLocation::HOST> &other) {
  *this = other;
}

MemoryBlock<MemoryLocation::DEVICE>::MemoryBlock(const MemoryBlock<MemoryLocation::DEVICE> &other) {
  *this = other;
}

MemoryBlock<MemoryLocation::DEVICE>::MemoryBlock(MemoryBlock<MemoryLocation::DEVICE> &&other) noexcept {
  *this = std::move(other);
}

MemoryBlock<MemoryLocation::DEVICE> &
MemoryBlock<MemoryLocation::DEVICE>::operator=(MemoryBlock<MemoryLocation::DEVICE> &&other) noexcept {
  if (this == &other)
    return *this;
  clear();
  size_ = other.size_;
  pitch_ = other.pitch_;
  data_ = other.data_;
  other.data_ = nullptr;
  other.size_ = {0, 0, 0};
  other.pitch_ = 0;
  return *this;
}

MemoryBlock<MemoryLocation::DEVICE> &
MemoryBlock<MemoryLocation::DEVICE>::operator=(const MemoryBlock<MemoryLocation::DEVICE> &other) {
  if (this == &other)
    return *this;
#ifdef HERMES_DEVICE_CODE
  resize(other.size_);
  if (other.size_.height == 1 && other.size_.depth == 1) {
    // linear memory
    HERMES_CHECK_CUDA(cudaMemcpy(data_, other.data_, size_.width, cudaMemcpyDeviceToDevice));
  } else if (other.size_.depth == 1) {
    // 2d pitched memory
    HERMES_CHECK_CUDA(cudaMemcpy2D(data_, pitch_, other.data_, other.pitch_, other.size_.width,
                                   other.size_.height, cudaMemcpyDeviceToDevice));
  } else {
    // 3d pitched memory
    cudaMemcpy3DParms p = {};
    p.srcPtr.ptr = other.data_;
    p.srcPtr.pitch = other.pitch_;
    p.srcPtr.xsize = other.size_.width;
    p.srcPtr.ysize = other.size_.height;
    p.dstPtr.ptr = data_;
    p.dstPtr.pitch = pitch_;
    p.dstPtr.xsize = size_.width;
    p.dstPtr.ysize = size_.height;
    p.extent.width = other.size_.width;
    p.extent.height = other.size_.height;
    p.extent.depth = other.size_.depth;
    p.kind = cudaMemcpyDeviceToDevice;
    HERMES_CHECK_CUDA(cudaMemcpy3D(&p));
  }
#endif
  return *this;
}

MemoryBlock<MemoryLocation::DEVICE> &
MemoryBlock<MemoryLocation::DEVICE>::operator=(const MemoryBlock<MemoryLocation::HOST> &other) {
#ifdef HERMES_DEVICE_CODE
  resize(other.size_);
  if (other.size_.height == 1 && other.size_.depth == 1) {
    // linear region
    HERMES_CHECK_CUDA(cudaMemcpy(data_, other.data_, size_.width, cudaMemcpyHostToDevice));
  } else if (other.size_.depth == 1) {
    // 2d pitched memory
    HERMES_CHECK_CUDA(cudaMemcpy2D(data_, pitch_, other.data_, other.pitch_, other.size_.width,
                                   other.size_.height, cudaMemcpyHostToDevice));
  } else {
    // 3d pitched memory
    cudaMemcpy3DParms p = {};
    p.srcPtr.ptr = other.data_;
    p.srcPtr.pitch = other.pitch_;
    p.srcPtr.xsize = other.size_.width;
    p.srcPtr.ysize = other.size_.height;
    p.dstPtr.ptr = data_;
    p.dstPtr.pitch = pitch_;
    p.dstPtr.xsize = size_.width;
    p.dstPtr.ysize = size_.height;
    p.extent.width = other.size_.width;
    p.extent.height = other.size_.height;
    p.extent.depth = other.size_.depth;
    p.kind = cudaMemcpyHostToDevice;
    HERMES_CHECK_CUDA(cudaMemcpy3D(&p));
  }
#endif
  return *this;
}

byte *MemoryBlock<MemoryLocation::DEVICE>::ptr() {
  return data_;
}

const byte *MemoryBlock<MemoryLocation::DEVICE>::ptr() const {
  return data_;
}

void MemoryBlock<MemoryLocation::DEVICE>::resize(size_t new_size_in_bytes) {
#ifdef HERMES_DEVICE_CODE
  size3 new_size(new_size_in_bytes, 1, 1);
  if (size_ == new_size)
    return;
  clear();
  size_ = {static_cast<u32>(new_size_in_bytes), 1, 1};
  pitch_ = new_size_in_bytes;
  HERMES_CHECK_CUDA(cudaMalloc(&data_, size_.total()));
#endif
}

void MemoryBlock<MemoryLocation::DEVICE>::resize(size2 new_size, size_t new_pitch) {
  HERMES_UNUSED_VARIABLE(new_pitch);
#ifdef HERMES_DEVICE_CODE
  if (size_ == size3(new_size.width, new_size.height, 1))
    return;
  if (new_size.height == 1) {
    resize(new_size.width);
    return;
  }
  clear();
  size_ = {new_size.width, new_size.height, 1};
  HERMES_CHECK_CUDA(cudaMallocPitch(&data_, &pitch_, size_.width, size_.height));
#endif
}

void MemoryBlock<MemoryLocation::DEVICE>::resize(size3 new_size, size_t new_pitch) {
#ifdef HERMES_DEVICE_CODE
  HERMES_UNUSED_VARIABLE(new_pitch);
  if (size_ == new_size)
    return;
  if (new_size.height == 1 && new_size.depth == 1) {
    resize(new_size.width);
    return;
  }
  if (new_size.depth == 1) {
    resize({new_size.width, new_size.height});
    return;
  }
  clear();
  size_ = new_size;
  cudaPitchedPtr pdata{};
  cudaExtent extent = make_cudaExtent(size_.width, size_.height, size_.depth);
  HERMES_CHECK_CUDA(cudaMalloc3D(&pdata, extent));
  pitch_ = pdata.pitch;
#endif
}

void MemoryBlock<MemoryLocation::DEVICE>::clear() {
#ifdef HERMES_DEVICE_CODE
  HERMES_CHECK_CUDA(cudaFree(data_))
#endif
  size_ = {0, 0, 0};
  pitch_ = 0;
  data_ = nullptr;
}

size_t MemoryBlock<MemoryLocation::DEVICE>::sizeInBytes() const {
  return pitch_ * size_.height * size_.depth;
}

size3 MemoryBlock<MemoryLocation::DEVICE>::size() const {
  return size_;
}

size_t MemoryBlock<MemoryLocation::DEVICE>::pitch() const {
  return pitch_;
}

void MemoryBlock<MemoryLocation::DEVICE>::copy(const void *data,
                                               size_t size_in_bytes,
                                               size_t offset,
                                               MemoryLocation data_location) {
  HERMES_CHECK_EXP(size_in_bytes <= sizeInBytes() - offset)
#ifdef HERMES_DEVICE_ENABLED
  if (size_.height == 1 && size_.depth == 1) {
    // linear region
    HERMES_CHECK_CUDA(cudaMemcpy(data_ + offset, data, size_in_bytes, cudaMemcpyHostToDevice));
  } else if (size_.depth == 1) {
    // 2d pitched memory
    HERMES_NOT_IMPLEMENTED
  } else {
    // 3d pitched memory
    HERMES_NOT_IMPLEMENTED
  }
#else
  HERMES_NOT_IMPLEMENTED
#endif
}
// *********************************************************************************************************************
//                                                                                                UNIFIED MemoryBlock
// *********************************************************************************************************************

MemoryBlock<MemoryLocation::UNIFIED>::MemoryBlock() = default;

MemoryBlock<MemoryLocation::UNIFIED>::~MemoryBlock() { clear(); }

MemoryBlock<MemoryLocation::UNIFIED>::MemoryBlock(size_t size_in_bytes) {
  resize(size_in_bytes);
}

MemoryBlock<MemoryLocation::UNIFIED>::MemoryBlock(const MemoryBlock<MemoryLocation::HOST> &other) {
  *this = other;
}

MemoryBlock<MemoryLocation::UNIFIED>::MemoryBlock(const MemoryBlock<MemoryLocation::DEVICE> &other) {
  *this = other;
}

MemoryBlock<MemoryLocation::UNIFIED>::MemoryBlock(MemoryBlock<MemoryLocation::UNIFIED> &&other) noexcept {
  *this = std::move(other);
}

size_t MemoryBlock<MemoryLocation::UNIFIED>::sizeInBytes() const { return pitch_ * size_.height * size_.depth; }

size_t MemoryBlock<MemoryLocation::UNIFIED>::pitch() const {
  return pitch_;
}

void MemoryBlock<MemoryLocation::UNIFIED>::resize(size_t new_size_in_bytes) {
#ifdef HERMES_DEVICE_CODE
  size3 new_size(new_size_in_bytes, 1, 1);
  if (size_ == new_size)
    return;
  clear();
  size_ = {static_cast<u32>(new_size_in_bytes), 1, 1};
  pitch_ = new_size_in_bytes;
  if (new_size_in_bytes) HERMES_CHECK_CUDA(cudaMallocManaged(&data_, this->sizeInBytes()))
#endif
}

void MemoryBlock<MemoryLocation::UNIFIED>::resize(size2 new_size, size_t new_pitch) {
#ifdef HERMES_DEVICE_CODE
  if (size_ == size3(new_size.width, new_size.height, 1) && pitch_ == new_pitch)
    return;
  clear();
  pitch_ = new_pitch;
  size_ = {new_size.width, new_size.height, 1};
  if (pitch_ == 0)
    pitch_ = size_.width;
  HERMES_CHECK_CUDA(cudaMallocManaged(&data_, this->sizeInBytes()))
#endif
}

void MemoryBlock<MemoryLocation::UNIFIED>::resize(size3 new_size, size_t new_pitch) {
#ifdef HERMES_DEVICE_CODE
  if (size_ == new_size && pitch_ == new_pitch)
    return;
  clear();
  pitch_ = new_pitch;
  size_ = new_size;
  if (pitch_ == 0)
    pitch_ = size_.width;
  HERMES_CHECK_CUDA(cudaMallocManaged(&data_, this->sizeInBytes()))
#endif
}

void MemoryBlock<MemoryLocation::UNIFIED>::clear() {
#ifdef HERMES_DEVICE_CODE
  if (data_) HERMES_CHECK_CUDA(cudaFree(data_))
#endif
  size_ = {0, 0, 0};
  data_ = nullptr;
}

MemoryBlock<MemoryLocation::UNIFIED> &
MemoryBlock<MemoryLocation::UNIFIED>::operator=(const MemoryBlock<MemoryLocation::DEVICE> &other) {
  HERMES_UNUSED_VARIABLE(other);
  HERMES_NOT_IMPLEMENTED
  return *this;
}

MemoryBlock<MemoryLocation::UNIFIED> &
MemoryBlock<MemoryLocation::UNIFIED>::operator=(const MemoryBlock<MemoryLocation::HOST> &other) {
  HERMES_UNUSED_VARIABLE(other);
  HERMES_NOT_IMPLEMENTED
  return *this;
}

MemoryBlock<MemoryLocation::UNIFIED> &
MemoryBlock<MemoryLocation::UNIFIED>::operator=(const MemoryBlock<MemoryLocation::UNIFIED> &other) {
  if (this == &other)
    return *this;
  HERMES_NOT_IMPLEMENTED
  return *this;
}

MemoryBlock<MemoryLocation::UNIFIED> &
MemoryBlock<MemoryLocation::UNIFIED>::operator=(MemoryBlock<MemoryLocation::UNIFIED> &&other) noexcept {
  if (this == &other)
    return *this;
  clear();
  size_ = other.size_;
  pitch_ = other.pitch_;
  data_ = other.data_;
  other.data_ = nullptr;
  other.size_ = {0, 0, 0};
  other.pitch_ = 0;
  return *this;
}

byte *MemoryBlock<MemoryLocation::UNIFIED>::ptr() {
  return data_;
}

const byte *MemoryBlock<MemoryLocation::UNIFIED>::ptr() const {
  return data_;
}

void MemoryBlock<MemoryLocation::UNIFIED>::copy(const void *data,
                                                size_t size_in_bytes,
                                                size_t offset,
                                                MemoryLocation data_location) {
  HERMES_CHECK_EXP(size_in_bytes <= sizeInBytes() - offset)
  if (data_location != MemoryLocation::DEVICE)
    std::memcpy(data_ + offset, data, size_in_bytes);
  else {
#ifdef HERMES_DEVICE_ENABLED
    HERMES_NOT_IMPLEMENTED
#else
    HERMES_NOT_IMPLEMENTED
#endif
  }
}

}