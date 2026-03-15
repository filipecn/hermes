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

bool operator==(const AoS::Layout &lhs, const AoS::Layout &rhs) {
  if (lhs.size_in_bytes_ != rhs.size_in_bytes_)
    return false;
  if (lhs.fields_.size() != rhs.fields_.size())
    return false;
  if (lhs.field_id_map_.size() != rhs.field_id_map_.size())
    return false;
  for (h_index i = 0; i < lhs.fields_.size(); ++i) {
    if (lhs.fields_[i].name != rhs.fields_[i].name ||
        lhs.fields_[i].size != rhs.fields_[i].size ||
        lhs.fields_[i].offset != rhs.fields_[i].offset ||
        lhs.fields_[i].component_count != rhs.fields_[i].component_count ||
        lhs.fields_[i].type != rhs.fields_[i].type)
      return false;
  }
  for (const auto &item : lhs.field_id_map_) {
    auto it = rhs.field_id_map_.find(item.first);
    if (it == rhs.field_id_map_.end())
      return false;
    if (item.second != it->second)
      return false;
  }
  return true;
}

bool operator!=(const AoS::Layout &lhs, const AoS::Layout &rhs) {
  return !(lhs == rhs);
}

u64 AoS::Layout::sizeOf(u64 field_id) const { return fields_[field_id].size; }

AoS::~AoS() noexcept { HERMES_CHECK_HE_RESULT(clear()); }

AoS::AoS(const AoS &rhs) { *this = rhs; }

AoS::AoS(AoS &&rhs) noexcept { *this = std::move(rhs); }

AoS &AoS::operator=(const AoS &rhs) {
  HERMES_CHECK_HE_RESULT(clear());
  HERMES_CHECK_HE_RESULT(data_.resize(rhs.dataSize()));
  auto err = data_.copy(rhs.data_);
  if (err != HeError::None) {
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
    return HeError::None;
  }
  if (data_.sizeInBytes() != layout.sizeInBytes() * size_) {
    HERMES_WARN("Setting incompatible layout to AoS.");
    return HeError::InvalidInput;
  }
  return HeError::None;
}

HeError AoS::resize(u64 count) {
  HERMES_RETURN_HE_ERROR(data_.clear());
  size_ = count;
  HERMES_RETURN_HE_ERROR(data_.resize(count * layout_.size_in_bytes_));
  return HeError::None;
}

const Block &AoS::data() const { return data_; }

Block &AoS::data() { return data_; }

HeError AoS::append(const AoS &other) {
  if (layout_ != other.layout_)
    return HeError::InvalidInput;
  Block new_block;
  auto new_count = size_ + other.size_;
  HERMES_RETURN_HE_ERROR(new_block.resize(new_count * layout_.size_in_bytes_));
  HERMES_RETURN_HE_ERROR(new_block.copy(data_));
  HERMES_RETURN_HE_ERROR(new_block.copy(other.data_, data_.sizeInBytes()));
  data_ = std::move(new_block);
  size_ = new_count;
  return HeError::None;
}

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
