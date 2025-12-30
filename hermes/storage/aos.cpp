/* Copyright (c) 2020, FilipeCN.
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

/// \file   aos.cpp
/// \author FilipeCN (filipedecn@gmail.com)
/// \date   2020-12-10

#include <hermes/storage/aos.h>

#include <hermes/core/debug.h>

namespace hermes {

HERMES_TO_STRING_METHOD_BEGIN(mem::AoS::Layout)
HERMES_TO_STRING_METHOD_LINE("Struct (size in bytes: {})\n",
                             object.sizeInBytes())
HERMES_TO_STRING_METHOD_LINE("fields: ")
HERMES_TO_STRING_METHOD_ARRAY_FIELD_BEGIN(fields_, f)
HERMES_TO_STRING_METHOD_LINE("field #{} ({})\n", i, f.name);
HERMES_TO_STRING_METHOD_LINE("\tbase data type: {}[{}]\n",
                             hermes::to_string(f.type), f.component_count)
HERMES_TO_STRING_METHOD_LINE("\tbase data size in bytes: {}\n", f.size)
HERMES_TO_STRING_METHOD_LINE("\toffset in bytes: {}\n", f.offset)
HERMES_TO_STRING_METHOD_ARRAY_FIELD_END
HERMES_TO_STRING_METHOD_END

HERMES_TO_STRING_METHOD_BEGIN(mem::AoS)
HERMES_TO_STRING_METHOD_FIELD(size_)
HERMES_TO_STRING_METHOD_HERMES_FIELD(layout_)
HERMES_TO_STRING_METHOD_HERMES_FIELD(data_)
HERMES_TO_STRING_METHOD_LINE("data values:\n");
auto fs = [&](const mem::AoS::Layout::Field &field,
              const void *data) -> std::string {
#define MATCH_TYPE(T)                                                          \
  if (field.type == DataTypes::typeFrom<T>()) {                                \
    auto array =                                                               \
        std::span{reinterpret_cast<const T *>(data), field.component_count};   \
    return cstr::join(array, ", ").c_str();                                    \
  }
  MATCH_TYPE(i8)
  MATCH_TYPE(i16)
  MATCH_TYPE(i32)
  MATCH_TYPE(i64)
  MATCH_TYPE(u8)
  MATCH_TYPE(u16)
  MATCH_TYPE(u32)
  MATCH_TYPE(u64)
  MATCH_TYPE(f32)
  MATCH_TYPE(f64)
  MATCH_TYPE(h_size)
  return "";
};
auto fields = object.layout().fields();
for (h_size i = 0; i < object.size(); ++i) {
  if (fields.size() == 1) {
    auto ptr = object.getPtr(0, i);
    HERMES_TO_STRING_METHOD_LINE("  AoS[{}][{}] = {}\n", i, fields[0].name,
                                 fs(fields[0], ptr));
  } else {
    for (h_size f = 0; f < fields.size(); ++f) {
      const auto &field = fields[f];
      auto ptr = object.getPtr(f, i);
      HERMES_TO_STRING_METHOD_LINE("  AoS[{}][{}] = {}\n", i, field.name,
                                   fs(fields[f], ptr));
    }
  }
}
HERMES_TO_STRING_METHOD_END

} // namespace hermes

namespace hermes::mem {

const std::string &AoS::Layout::fieldName(u64 field_id) const {
  return fields_[field_id].name;
}

bool AoS::Layout::contains(const std::string &field_name) const {
  return field_id_map_.count(field_name);
}

const std::vector<AoS::Layout::Field> &AoS::Layout::fields() const {
  return fields_;
}

u64 AoS::Layout::fieldId(const std::string &field_name) const {
  auto it = field_id_map_.find(field_name);
  if (it != field_id_map_.end())
    return it->second;
  HERMES_ERROR("Struct layout field [{}] not found.", field_name);
  return 0;
}

ptrdiff_t AoS::Layout::addressOffsetOf(u64 field_id, u64 i) const {
  return i * size_in_bytes_ + fields_[field_id].offset;
}

u64 AoS::Layout::offsetOf(const std::string &field_name) const {
  auto it = field_id_map_.find(field_name);
  if (it == field_id_map_.end()) {
    HERMES_ERROR("Field {} not found.", field_name);
    return 0;
  }
  return offsetOf(it->second);
}

u64 AoS::Layout::offsetOf(u64 field_id) const {
  return fields_[field_id].offset;
}

u64 AoS::Layout::sizeOf(const std::string &field_name) const {
  auto it = field_id_map_.find(field_name);
  if (it == field_id_map_.end()) {
    HERMES_ERROR("Field {} not found.", field_name);
    return 0;
  }
  return sizeOf(it->second);
}

u64 AoS::Layout::sizeOf(u64 field_id) const { return fields_[field_id].size; }

AoS::~AoS() noexcept { HERMES_CHECK_HE_RESULT(clear()); }

AoS::AoS(const AoS &rhs) { *this = rhs; }

AoS::AoS(AoS &&rhs) noexcept { *this = std::move(rhs); }

AoS &AoS::operator=(const AoS &rhs) {
  HERMES_CHECK_HE_RESULT(clear());
  HERMES_CHECK_HE_RESULT(data_.resize(rhs.dataSize()));
  auto err = data_.copy(rhs.data_);
  if (err != HeError::NO_ERROR) {
    HERMES_ERROR("Failed to copy assign AoS. [err={}]", hermes::to_string(err));
    return *this;
  }
  size_ = rhs.size_;
  layout_ = rhs.layout_;
  return *this;
}

AoS &AoS::operator=(AoS &&rhs) noexcept {
  HERMES_CHECK_HE_RESULT(clear());
  data_ = std::move(rhs.data_);
  layout_ = std::move(rhs.layout_);
  size_ = rhs.size_;
  return *this;
}

HeError AoS::clear() noexcept {
  size_ = 0;
  return data_.clear();
}

u64 AoS::size() const { return size_; }

u64 AoS::dataSize() const { return size_ * layout_.size_in_bytes_; }

u64 AoS::stride() const { return layout_.size_in_bytes_; }

const AoS::Layout &AoS::layout() const { return layout_; }

HeError AoS::setLayout(const Layout &layout) {
  if (layout.size_in_bytes_ == 0 || size_ == 0) {
    layout_ = layout;
    return HeError::NO_ERROR;
  }
  if (data_.sizeInBytes() != layout.sizeInBytes() * size_) {
    HERMES_WARN("Setting incompatible layout to AoS.");
    return HeError::INVALID_INPUT;
  }
  return HeError::NO_ERROR;
}

HeError AoS::resize(u64 count) {
  HERMES_RETURN_HE_ERROR(data_.clear());
  size_ = count;
  HERMES_RETURN_HE_ERROR(data_.resize(count * layout_.size_in_bytes_));
  return HeError::NO_ERROR;
}

const Block &AoS::data() const { return data_; }

Block &AoS::data() { return data_; }

} // namespace hermes::mem

/*
std::ostream &operator<<(std::ostream &o, const hermes::mem::StructLayout &sd) {
  o << hermes::to_string(sd);
  return o;
}

std::ofstream &operator<<(std::ofstream &o,
                          const hermes::mem::StructLayout &sd) {
  o.write(reinterpret_cast<const char *>(&sd.size_in_bytes_), sizeof(u64));
  u64 field_count = sd.fields_.size();
  o.write(reinterpret_cast<const char *>(&field_count), sizeof(u64));
  for (const auto &f : sd.fields_) {
    o.write(reinterpret_cast<const char *>(&f.size),
            sizeof(hermes::mem::StructLayout::Field::size));
    o.write(reinterpret_cast<const char *>(&f.component_count),
            sizeof(hermes::mem::StructLayout::Field::component_count));
    o.write(reinterpret_cast<const char *>(&f.offset),
            sizeof(hermes::mem::StructLayout::Field::offset));
    o.write(reinterpret_cast<const char *>(&f.type),
            sizeof(hermes::mem::StructLayout::Field::type));
    u64 name_size = f.name.size();
    o.write(reinterpret_cast<const char *>(&name_size), sizeof(u64));
    auto name = f.name.c_str();
    for (u64 i = 0; i < name_size; ++i)
      o.write(reinterpret_cast<const char *>(&name[i]), sizeof(char));
  }
  return o;
}

std::ifstream &operator>>(std::ifstream &i, hermes::mem::StructLayout &sd) {
  i.read(reinterpret_cast<char *>(&sd.size_in_bytes_), sizeof(u64));
  u64 field_count = 0;
  i.read(reinterpret_cast<char *>(&field_count), sizeof(u64));
  for (u64 f = 0; f < field_count; ++f) {
    hermes::mem::StructLayout::Field field;
    i.read(reinterpret_cast<char *>(&field.size),
           sizeof(hermes::mem::StructLayout::Field::size));
    i.read(reinterpret_cast<char *>(&field.component_count),
           sizeof(hermes::mem::StructLayout::Field::component_count));
    i.read(reinterpret_cast<char *>(&field.offset),
           sizeof(hermes::mem::StructLayout::Field::offset));
    i.read(reinterpret_cast<char *>(&field.type),
           sizeof(hermes::mem::StructLayout::Field::type));
    u64 name_size = 0;
    i.read(reinterpret_cast<char *>(&name_size), sizeof(u64));
    field.name.resize(name_size);
    i.read(&field.name[0], name_size);
    sd.field_id_map_[field.name] = sd.fields_.size();
    sd.fields_.emplace_back(field);
  }
  return i;
}
*/
