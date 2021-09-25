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
///\file stack_allocator.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-09-22
///
///\brief

#include <hermes/storage/stack_allocator.h>

namespace hermes {

#define SA_EXTRACT_MARKER(HANDLE) \
  ((HANDLE & 0xffffff) - 1)

#define SA_BUILD_HANDLE(MARKER, SHIFT) \
  ((MARKER + 1u) | (SHIFT << 24u))

// *********************************************************************************************************************
//                                                                                        Memory Stack Allocator View
// *********************************************************************************************************************

HERMES_DEVICE_CALLABLE StackAllocatorView::StackAllocatorView(byte *data,
                                                              std::size_t capacity_in_bytes,
                                                              std::size_t marker)
    : data_(data), capacity_in_bytes(capacity_in_bytes), marker_(marker) {

}

HERMES_DEVICE_CALLABLE std::size_t StackAllocatorView::availableSizeInBytes() const {
  return capacity_in_bytes - marker_;
}

HERMES_DEVICE_CALLABLE AddressIndex StackAllocatorView::allocate(std::size_t block_size_in_bytes, std::size_t align) {
  std::size_t
      actual_size = block_size_in_bytes + mem::rightAlignShift(reinterpret_cast<uintptr_t >(data_ ) + marker_, align);
  std::size_t shift = actual_size - block_size_in_bytes;
  if (actual_size > capacity_in_bytes - marker_)
    return {0};
  const auto marker = marker_;
  marker_ += actual_size;
  return {SA_BUILD_HANDLE(marker + shift, shift)};
}

HERMES_DEVICE_CALLABLE HeResult StackAllocatorView::freeTo(AddressIndex handle) {
  if (!marker_)
    return HeResult::BAD_OPERATION;
  if (!handle.id)
    return HeResult::INVALID_INPUT;
  marker_ = SA_EXTRACT_MARKER(handle.id);
  return HeResult::SUCCESS;
}

HERMES_DEVICE_CALLABLE void StackAllocatorView::clear() {
  marker_ = 0;
}

// *********************************************************************************************************************
//                                                                                        HOST Memory Stack Allocator
// *********************************************************************************************************************
MemoryStackAllocator<MemoryLocation::HOST>::MemoryStackAllocator(std::size_t size_in_bytes) {
  resize(size_in_bytes);
}

MemoryStackAllocator<MemoryLocation::HOST>::MemoryStackAllocator(std::size_t size_in_bytes, byte *buffer) :
    data_(buffer), capacity_(size_in_bytes), using_external_memory_{true} {
}

MemoryStackAllocator<MemoryLocation::HOST>::~MemoryStackAllocator() = default;

std::size_t MemoryStackAllocator<MemoryLocation::HOST>::capacityInBytes() const {
  return capacity_;
}

std::size_t MemoryStackAllocator<MemoryLocation::HOST>::availableSizeInBytes() const {
  return capacity_ - marker_;
}

HeResult MemoryStackAllocator<MemoryLocation::HOST>::resize(std::size_t size_in_bytes) {
  if (using_external_memory_)
    return HeResult::BAD_OPERATION;
  marker_ = 0;
  mem_block_.resize(size_in_bytes);
  capacity_ = mem_block_.sizeInBytes();
  data_ = mem_block_.ptr();
  return capacity_ >= size_in_bytes ? HeResult::SUCCESS : HeResult::BAD_ALLOCATION;
}

AddressIndex MemoryStackAllocator<MemoryLocation::HOST>::allocate(std::size_t block_size_in_bytes, std::size_t align) {
  std::size_t
      actual_size = block_size_in_bytes + mem::rightAlignShift(reinterpret_cast<uintptr_t >(data_ ) + marker_, align);
  std::size_t shift = actual_size - block_size_in_bytes;
  if (actual_size > capacity_ - marker_)
    return {0};
  const auto marker = marker_;
  marker_ += actual_size;
  return {SA_BUILD_HANDLE(marker + shift, shift)};
}

HeResult MemoryStackAllocator<MemoryLocation::HOST>::freeTo(AddressIndex handle) {
  if (!marker_)
    return HeResult::BAD_OPERATION;
  if (!handle.id)
    return HeResult::INVALID_INPUT;
  marker_ = SA_EXTRACT_MARKER(handle.id);
  return HeResult::SUCCESS;
}

void MemoryStackAllocator<MemoryLocation::HOST>::clear() {
  marker_ = 0;
}

StackAllocatorView MemoryStackAllocator<MemoryLocation::HOST>::view() {
  return StackAllocatorView(data_, capacity_, marker_);
}

// *********************************************************************************************************************
//                                                                                     UNIFIED Memory Stack Allocator
// *********************************************************************************************************************
MemoryStackAllocator<MemoryLocation::UNIFIED>::MemoryStackAllocator(std::size_t size_in_bytes) {
  resize(size_in_bytes);
}

MemoryStackAllocator<MemoryLocation::UNIFIED>::MemoryStackAllocator(std::size_t size_in_bytes, byte *buffer) :
    data_(buffer), capacity_(size_in_bytes), using_external_memory_{true} {
}

MemoryStackAllocator<MemoryLocation::UNIFIED>::~MemoryStackAllocator() = default;

std::size_t MemoryStackAllocator<MemoryLocation::UNIFIED>::capacityInBytes() const {
  return capacity_;
}

std::size_t MemoryStackAllocator<MemoryLocation::UNIFIED>::availableSizeInBytes() const {
  return capacity_ - marker_;
}

HeResult MemoryStackAllocator<MemoryLocation::UNIFIED>::resize(std::size_t size_in_bytes) {
  if (using_external_memory_)
    return HeResult::BAD_OPERATION;
  marker_ = 0;
  mem_block_.resize(size_in_bytes);
  capacity_ = mem_block_.sizeInBytes();
  data_ = mem_block_.ptr();
  return capacity_ >= size_in_bytes ? HeResult::SUCCESS : HeResult::BAD_ALLOCATION;
}

AddressIndex MemoryStackAllocator<MemoryLocation::UNIFIED>::allocate(std::size_t block_size_in_bytes,
                                                                     std::size_t align) {
  std::size_t
      actual_size = block_size_in_bytes + mem::rightAlignShift(reinterpret_cast<uintptr_t >(data_ ) + marker_, align);
  std::size_t shift = actual_size - block_size_in_bytes;
  if (actual_size > capacity_ - marker_)
    return {0};
  const auto marker = marker_;
  marker_ += actual_size;
  return {SA_BUILD_HANDLE(marker + shift, shift)};
}

HeResult MemoryStackAllocator<MemoryLocation::UNIFIED>::freeTo(AddressIndex handle) {
  if (!marker_)
    return HeResult::BAD_OPERATION;
  if (!handle.id)
    return HeResult::INVALID_INPUT;
  marker_ = SA_EXTRACT_MARKER(handle.id);
  return HeResult::SUCCESS;
}

void MemoryStackAllocator<MemoryLocation::UNIFIED>::clear() {
  marker_ = 0;
}

StackAllocatorView MemoryStackAllocator<MemoryLocation::UNIFIED>::view() {
  return StackAllocatorView(data_, capacity_, marker_);
}

#undef SA_EXTRACT_MARKER
#undef SA_BUILD_HANDLE

}