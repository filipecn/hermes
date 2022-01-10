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
///\file struct_descriptor_.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-07-14
///
///\brief

#include <hermes/storage/struct_descriptor.h>
#include <hermes/common/debug.h>

namespace hermes {

StructDescriptorView::StructDescriptorView(const StructDescriptor &descriptor) :
    struct_size{descriptor.struct_size_} {
  HERMES_CHECK_EXP(descriptor.fields_.size() < HERMES_STRUCT_DESCRIPTOR_VIEW_MAX_FIELD_COUNT);
  for (size_t i = 0; i < descriptor.fields_.size() && i < HERMES_STRUCT_DESCRIPTOR_VIEW_MAX_FIELD_COUNT; ++i) {
    fields_[i].size = descriptor.fields_[i].size;
    fields_[i].offset = descriptor.fields_[i].offset;
    fields_[i].component_count = descriptor.fields_[i].component_count;
    fields_[i].type = descriptor.fields_[i].type;
  }
}

ptrdiff_t StructDescriptorView::addressOffsetOf(u64 field_id, u64 i) const {
  return i * struct_size + fields_[field_id].offset;
}

HERMES_DEVICE_CALLABLE u64 StructDescriptorView::offsetOf(u64 field_id) const {
  return fields_[field_id].offset;
}

HERMES_DEVICE_CALLABLE u64 StructDescriptorView::sizeOf(u64 field_id) const {
  return fields_[field_id].size;
}

ptrdiff_t StructDescriptor::addressOffsetOf(u64 field_id, u64 i) const {
  return i * struct_size_ + fields_[field_id].offset;
}

u64 StructDescriptor::offsetOf(const std::string &field_name) const {
  auto it = field_id_map_.find(field_name);
  if (it == field_id_map_.end()) {
    Log::error("Field {} not found.", field_name);
    return 0;
  }
  return offsetOf(it->second);
}

u64 StructDescriptor::offsetOf(u64 field_id) const {
  return fields_[field_id].offset;
}

u64 StructDescriptor::sizeOf(const std::string &field_name) const {
  auto it = field_id_map_.find(field_name);
  if (it == field_id_map_.end()) {
    Log::error("Field {} not found.", field_name);
    return 0;
  }
  return sizeOf(it->second);
}

u64 StructDescriptor::sizeOf(u64 field_id) const {
  return fields_[field_id].size;
}

}

