/// Copyright (c) 2022, FilipeCN.
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
///\file cartesian_hash.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-11-24
///
///\brief

#include <hermes/data_structures/cartesian_hash.h>
#include <algorithm>
#include <hermes/common/index.h>

namespace hermes {

CartesianHashMap2 CartesianHashMap2::from(const std::vector<point2> &point_set, real_t cell_size) {
  CartesianHashMap2 chm(cell_size);
  for (size_t i = 0; i < point_set.size(); ++i)
    chm.point_set_.push_back({
                                 .hash_value = chm.hashValue(point_set[i]),
                                 .position = point_set[i],
                                 .index = i
                             });
  chm.update();
  return chm;
}

CartesianHashMap2::CartesianHashMap2(real_t cell_size) {
  setSearchCellSize(cell_size);
};

CartesianHashMap2::~CartesianHashMap2() = default;

size_t CartesianHashMap2::search(const bbox2 &region, const std::function<bool(size_t)> &callback) const {
  auto min_cell = hashValue(region.lower);
  auto max_cell = hashValue(region.upper);
  size_t count = 0;
  for (auto ij : range2_64(min_cell, max_cell.plus(1, 1)))
    count += search(ij, [&](size_t i) {
      if (region.contains(point_set_[i].position))
        if (!callback(point_set_[i].index))
          return false;
      return true;
    });
  return count;
}

size_t CartesianHashMap2::search(const point2 &center, real_t radius,
                                 const std::function<bool(size_t)> &callback) const {
  // get bounding box for circle
  auto r2 = radius * radius;
  bbox2 region(center - vec2(radius), center + vec2(radius));
  auto min_cell = hashValue(region.lower);
  auto max_cell = hashValue(region.upper);
  size_t count = 0;
  for (auto ij : range2_64(min_cell, max_cell.plus(1, 1)))
    count += search(ij, [&](size_t i) {
      if (hermes::distance2(center, point_set_[i].position) < r2)
        if (!callback(point_set_[i].index))
          return false;
      return true;
    });
  return count;
}

index2_64 CartesianHashMap2::hashValue(const point2 &p) const {
  return index2_64(std::floor(p.x / cell_size_), std::floor(p.y / cell_size_));
}

void CartesianHashMap2::setSearchCellSize(real_t cell_size) {
  cell_size_ = cell_size;
}

void CartesianHashMap2::update() {
  // sort elements by hash
  std::sort(point_set_.begin(), point_set_.end(), [](const Element &a, const Element &b) {
    if(a.hash_value == b.hash_value) {
      if(a.position == b.position)
        return a.index < b.index;
      return a.position.x == b.position.x ? a.position.y < b.position.y : a.position.x < b.position.x;
    }
    return a.hash_value.i == b.hash_value.i ? a.hash_value.j < b.hash_value.j : a.hash_value.i < b.hash_value.i;
  });
}

std::ostream &operator<<(std::ostream &os, const CartesianHashMap2 &chm) {
  os << "CartesianHashMap2 [" << chm.point_set_.size() << " points][cell size " << chm.cell_size_ << "]\n";
  if (!chm.point_set_.empty()) {
    size_t current_count = 0;
    index2_64 current = chm.point_set_.front().hash_value;
    for (const auto &e : chm.point_set_)
      if (current != e.hash_value) {
        if (current_count)
          os << "[" << e.hash_value << " : " << current_count << "]";
        current_count = 1;
        current = e.hash_value;
      } else
        current_count++;
  }
  os << "\n";
  return os;
}

size_t CartesianHashMap2::search(const index2_64 &cell_id, const std::function<bool(size_t)> &callback) const {
  auto it = std::lower_bound(point_set_.begin(), point_set_.end(), cell_id,
                             [](const Element &element, index2_64 key) {
                               return element.hash_value.i == key.i ?
                                      element.hash_value.j < key.j :
                                      element.hash_value.i < key.i;
                             });
  size_t count = 0;
  if (it != point_set_.end()) {
    // found!
    auto index = it - point_set_.begin();
    for (size_t i = index; i < point_set_.size(); ++i) {
      if (point_set_[i].hash_value != cell_id)
        break;
      count++;
      if (!callback(i))
        break;
    }
  }
  return count;
}

}
