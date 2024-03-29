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
///\file storage_tests.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-27
///
///\brief

#include <catch2/catch.hpp>

#include <hermes/storage/array.h>
#include <hermes/storage/array_of_structures.h>
#include <hermes/geometry/vector.h>
#include <hermes/storage/memory_block.h>
#include <hermes/common/cuda_utils.h>
#include <hermes/storage/stack_allocator.h>
#include <hermes/storage/array_slice.h>

using namespace hermes;

class Object {
public:
  Object() {
    data.resize(128);
    printf("calling constructor %p\n", data.ptr());
  }
  Object(Object &&o) {
    printf("calling move constructor\n");
    a = 3;
  }
  Object(const Object &o) {
    printf("calling copy constructor\n");
    a = 2;
  }
  Object &operator=(const Object &o) {
    printf("calling copy assignment\n");
    a = 1;
    return *this;
  }
  ~Object() {
    printf("calling destructor\n");
    a = 0;
  }
  HERMES_DEVICE_CALLABLE
  void say() const { printf("ah %d %ul!\n", a, data.size().total()); }
  int a = 10;
  MemoryBlock<MemoryLocation::DEVICE> data;
};

#ifdef HERMES_DEVICE_ENABLED
HERMES_CUDA_KERNEL(writeMatrixIndex)(u32 *data, size2 bounds) {
  HERMES_CUDA_THREAD_INDEX_IJ_LT(bounds);
  u32 matrix_index = ij.j * bounds.width + ij.i;
  data[matrix_index] = matrix_index;
}

HERMES_CUDA_KERNEL(testArrayView)(ArrayView<int> array) {
  HERMES_CUDA_THREAD_INDEX_IJ_LT(array.size.slice());
  array[ij] = ij.j * array.size.width + ij.i;
}

HERMES_CUDA_KERNEL(testObject)(Object o) {
  o.say();
}
#endif

TEST_CASE("object") {
  SECTION("rvalue") {
#ifdef HERMES_DEVICE_ENABLED
    HERMES_CUDA_LAUNCH_AND_SYNC((10), testObject_k, Object())
    Object o;
    HERMES_CUDA_LAUNCH_AND_SYNC((10), testObject_k, o)
    o.say();
#endif
  }
}

TEST_CASE("MemoryBlock", "[storage]") {
  auto writeHostMemory = [](HostMemory &hm) {
    u8 *ptr = reinterpret_cast<u8 *>( hm.ptr());
    for (u8 i = 0; i < static_cast<u8>(hm.sizeInBytes()); ++i)
      ptr[i] = i;
  };
  auto checkHostMemory = [](HostMemory &hm) -> bool {
    u8 *ptr = reinterpret_cast<u8 *>( hm.ptr());
    for (u8 i = 0; i < static_cast<u8>(hm.sizeInBytes()); ++i)
      if (ptr[i] != i)
        return false;
    return true;
  };
  auto checkDeviceMemory = [&](DeviceMemory &dm) -> bool {
    HostMemory hm = dm;
    return checkHostMemory(hm);
  };

  SECTION("sanity") {
    HostMemory hm;
    REQUIRE(hm.location == MemoryLocation::HOST);
    DeviceMemory dm;
    REQUIRE(dm.location == MemoryLocation::DEVICE);
  }//
  SECTION("copy") {
#ifdef HERMES_DEVICE_ENABLED
    HostMemory hm(8);
    int a = 1;
    int b = 2;
    hm.copy(&a);
    hm.copy(&b, 4);
    REQUIRE(reinterpret_cast<int *>(hm.ptr())[0] == a);
    REQUIRE(reinterpret_cast<int *>(hm.ptr())[1] == b);
    DeviceMemory dm(8);
    dm.copy(&a, 0, MemoryLocation::HOST);
    dm.copy(&b, 4, MemoryLocation::HOST);
    HostMemory hm2 = dm;
    REQUIRE(reinterpret_cast<int *>(hm2.ptr())[0] == a);
    REQUIRE(reinterpret_cast<int *>(hm2.ptr())[1] == b);
#endif
  }//
  SECTION("assignment") {
    SECTION("host") {
      HostMemory base(100);
      writeHostMemory(base);
      HostMemory cpy = base;
      REQUIRE(base.sizeInBytes() == 100);
      REQUIRE(cpy.sizeInBytes() == 100);
      REQUIRE(cpy.sizeInBytes() == base.sizeInBytes());
      REQUIRE(checkHostMemory(base));
      REQUIRE(checkHostMemory(cpy));
      HostMemory mv = std::move(base);
      REQUIRE(base.sizeInBytes() == 0);
      REQUIRE(mv.sizeInBytes() == 100);
      REQUIRE(checkHostMemory(mv));
    }//
  }//
  SECTION("resize") {
    SECTION("host") {
      HostMemory hm;
      REQUIRE(hm.sizeInBytes() == 0);
      hm.resize(100);
      writeHostMemory(hm);
      REQUIRE(hm.sizeInBytes() == 100);
      REQUIRE(checkHostMemory(hm));
      hm.resize({64, 2}, 0);
      writeHostMemory(hm);
      REQUIRE(hm.sizeInBytes() == 128);
      REQUIRE(checkHostMemory(hm));
      hm.resize({32, 4, 2}, 0);
      writeHostMemory(hm);
      REQUIRE(hm.sizeInBytes() == 256);
      REQUIRE(checkHostMemory(hm));
    }//
#ifdef HERMES_DEVICE_ENABLED
    SECTION("device") {
      DeviceMemory dm;
      REQUIRE(dm.sizeInBytes() == 0);
      dm.resize(100);
      HostMemory hm(256);
      writeHostMemory(hm);
      dm = hm;
      REQUIRE(dm.sizeInBytes() == 256);
      REQUIRE(dm.size() == hm.size());
      REQUIRE(checkDeviceMemory(dm));
    }//
#endif
  }//
  SECTION("linear block") {
    SECTION("host") {
      HostMemory hm(256);
      REQUIRE(hm.sizeInBytes() == 256);
      writeHostMemory(hm);
      REQUIRE(checkHostMemory(hm));
    }//
    SECTION("device") {
      DeviceMemory dm(256);
#ifdef HERMES_DEVICE_ENABLED
      REQUIRE(dm.sizeInBytes() == 256);
#endif
    }//
  }//
  SECTION("2d block") {
    SECTION("host") {
      HostMemory hm({32, 8});
      REQUIRE(hm.sizeInBytes() == 256);
      writeHostMemory(hm);
      REQUIRE(checkHostMemory(hm));
    }//
  }//
  SECTION("3d block") {
    SECTION("host") {
      HostMemory hm({32, 4, 2});
      REQUIRE(hm.sizeInBytes() == 256);
      writeHostMemory(hm);
      REQUIRE(checkHostMemory(hm));
    }//
  }//
#ifdef HERMES_DEVICE_ENABLED
  SECTION("unified") {
    UnifiedMemory um(64 * 128 * 4);
    u32 *data = reinterpret_cast<u32 *>( um.ptr());
    size2 bounds(64, 128);
    HERMES_CUDA_LAUNCH_AND_SYNC((bounds), writeMatrixIndex_k, data, bounds)
    for (u32 j = 0; j < 128; ++j)
      for (u32 i = 0; i < 64; ++i) {
        u32 ind = j * 64 + i;
        REQUIRE(data[ind] == ind);
      }
  }
#endif
}

TEST_CASE("mem", "[memory]") {
  SECTION("alignTo") {
    REQUIRE(mem::alignTo(1, sizeof(u8)) == sizeof(u8));
    REQUIRE(mem::alignTo(1, sizeof(u16)) == sizeof(u16));
    REQUIRE(mem::alignTo(1, sizeof(u32)) == sizeof(u32));
    REQUIRE(mem::alignTo(1, sizeof(u64)) == sizeof(u64));
    struct S {
      f32 a;
      u8 b;
      u16 c;
    };
    REQUIRE(sizeof(S) == 8);
    REQUIRE(mem::alignTo(15, sizeof(S)) == 16);
    REQUIRE(mem::alignTo(17, sizeof(S)) == 24);
  }//
  SECTION("left and right alignments") {
    REQUIRE(mem::leftAlignShift(100, 64) == 100 - 64);
    REQUIRE(mem::rightAlignShift(100, 64) == 128 - 100);

    REQUIRE(mem::leftAlignShift(100, 1) == 0);
    REQUIRE(mem::rightAlignShift(100, 1) == 0);
  }//
  SECTION("allocAligned") {
    auto *ptr = mem::allocAligned(10, 1);
    mem::freeAligned(ptr);
  }//
}

#ifdef HERMES_DEVICE_ENABLED
HERMES_CUDA_KERNEL(fillStackAllocator)(StackAllocatorView stack_allocator,
                                       ArrayView<AddressIndex> handles,
                                       HeResult *result) {
  HERMES_CUDA_RETURN_IF_NOT_THREAD_0
  for (int i = 0; i < 20; ++i)
    handles.emplace(i, stack_allocator.pushAligned<int>(0));
  for (int i = 0; i < 20; ++i) {
    *result = stack_allocator.set(handles[i], i);
    if (*result != HeResult::SUCCESS)
      return;
  }
}

HERMES_CUDA_KERNEL(checkStackAllocator)(StackAllocatorView stack_allocator,
                                        ArrayView<AddressIndex> handles,
                                        HeResult *result) {
  HERMES_CUDA_RETURN_IF_NOT_THREAD_0
  for (int i = 0; i < 20; ++i)
    if (i != *stack_allocator.get<int>(handles[i]))
      *result = HeResult::BAD_OPERATION;
}
#endif

TEST_CASE("StackAllocator", "[memory]") {
  SECTION("HOST") {
    SECTION("empty") {
      StackAllocator stack_allocator;
      REQUIRE(stack_allocator.capacityInBytes() == 0);
      REQUIRE(stack_allocator.availableSizeInBytes() == 0);
      REQUIRE(stack_allocator.allocate(10).id == 0);
      REQUIRE(stack_allocator.pushAligned<int>().id == 0);
      REQUIRE(stack_allocator.freeTo({}) == HeResult::BAD_OPERATION);
    }//
    SECTION("sanity") {
      StackAllocator stack_allocator;
      REQUIRE(stack_allocator.resize(100) == HeResult::SUCCESS);
      REQUIRE(stack_allocator.capacityInBytes() == 100);
      REQUIRE(stack_allocator.availableSizeInBytes() == 100);
      auto p = stack_allocator.allocate(50);
      REQUIRE(p.id == 1);
      REQUIRE(stack_allocator.availableSizeInBytes() == 50);
      stack_allocator.clear();
      REQUIRE(stack_allocator.availableSizeInBytes() == 100);
      stack_allocator.resize(200);
      REQUIRE(stack_allocator.capacityInBytes() == 200);
      auto p1 = stack_allocator.allocate(180);
      REQUIRE(p1.id == 1);
      auto p2 = stack_allocator.allocate(40);
      REQUIRE(p2.id == 0);
      REQUIRE(stack_allocator.availableSizeInBytes() == 20);
      REQUIRE(stack_allocator.freeTo(p1) == HeResult::SUCCESS);
      REQUIRE(stack_allocator.availableSizeInBytes() == 200);
    }//
    SECTION("debug") {
#ifdef ODYSSEUS_DEBUG
      StackAllocator stack_allocator(200);
      stack_allocator.allocate(10);
      stack_allocator.allocate(50);
      stack_allocator.allocate(80, 64);
      stack_allocator.dump();
#endif
    }//
    SECTION("set get") {
      StackAllocator stack_allocator(80);
      std::vector<AddressIndex> handles;
      handles.reserve(20);
      for (int i = 0; i < 20; ++i)
        handles.emplace_back(stack_allocator.pushAligned<int>(0));
      for (int i = 0; i < 20; ++i)
        REQUIRE(stack_allocator.set(handles[i], i) == HeResult::SUCCESS);
      for (int i = 0; i < 20; ++i)
        REQUIRE(*stack_allocator.get<int>(handles[i]) == i);
    }//
    SECTION("view") {
      StackAllocator stack_allocator(80);
      std::vector<AddressIndex> handles;
      handles.reserve(20);
      for (int i = 0; i < 20; ++i)
        handles.emplace_back(stack_allocator.pushAligned<int>(0));
      for (int i = 0; i < 20; ++i)
        REQUIRE(stack_allocator.set(handles[i], i) == HeResult::SUCCESS);
      auto view = stack_allocator.view();
      for (int i = 0; i < 20; ++i)
        REQUIRE(*view.get<int>(handles[i]) == i);
      for (int i = 0; i < 20; ++i)
        REQUIRE(view.set(handles[i], 2 * i) == HeResult::SUCCESS);
      for (int i = 0; i < 20; ++i)
        REQUIRE(*stack_allocator.get<int>(handles[i]) == 2 * i);
    }//
  }//
#ifdef HERMES_DEVICE_ENABLED
  SECTION("UNIFIED") {
    SECTION("empty") {
      UnifiedStackAllocator stack_allocator;
      REQUIRE(stack_allocator.capacityInBytes() == 0);
      REQUIRE(stack_allocator.availableSizeInBytes() == 0);
      REQUIRE(stack_allocator.allocate(10).id == 0);
      REQUIRE(stack_allocator.allocateAligned<int>().id == 0);
      REQUIRE(stack_allocator.freeTo({}) == HeResult::BAD_OPERATION);
    }//
    SECTION("sanity") {
      UnifiedStackAllocator stack_allocator;
      REQUIRE(stack_allocator.resize(100) == HeResult::SUCCESS);
      REQUIRE(stack_allocator.capacityInBytes() == 100);
      REQUIRE(stack_allocator.availableSizeInBytes() == 100);
      auto p = stack_allocator.allocate(50);
      REQUIRE(p.id == 1);
      REQUIRE(stack_allocator.availableSizeInBytes() == 50);
      stack_allocator.clear();
      REQUIRE(stack_allocator.availableSizeInBytes() == 100);
      stack_allocator.resize(200);
      REQUIRE(stack_allocator.capacityInBytes() == 200);
      auto p1 = stack_allocator.allocate(180);
      REQUIRE(p1.id == 1);
      auto p2 = stack_allocator.allocate(40);
      REQUIRE(p2.id == 0);
      REQUIRE(stack_allocator.availableSizeInBytes() == 20);
      REQUIRE(stack_allocator.freeTo(p1) == HeResult::SUCCESS);
      REQUIRE(stack_allocator.availableSizeInBytes() == 200);
    }//
    SECTION("debug") {
#ifdef ODYSSEUS_DEBUG
      UnifiedStackAllocator stack_allocator(200);
    stack_allocator.allocate(10);
    stack_allocator.allocate(50);
    stack_allocator.allocate(80, 64);
    stack_allocator.dump();
#endif
    }//
    SECTION("set get") {
      UnifiedStackAllocator stack_allocator(80);
      std::vector<AddressIndex> handles;
      handles.reserve(20);
      for (int i = 0; i < 20; ++i)
        handles.emplace_back(stack_allocator.allocateAligned<int>(0));
      for (int i = 0; i < 20; ++i)
        REQUIRE(stack_allocator.set(handles[i], i) == HeResult::SUCCESS);
#ifdef ODYSSEUS_DEBUG
      stack_allocator.dump();
#endif
      for (int i = 0; i < 20; ++i)
        REQUIRE(*stack_allocator.get<int>(handles[i]) == i);
    }
  }//
  SECTION("DEVICE") {
    SECTION("sanity") {
      DeviceStackAllocator stack_allocator(80);
      DeviceArray<AddressIndex> handles(20);
      UnifiedArray<HeResult> result(1);
      HERMES_CUDA_LAUNCH_AND_SYNC((1), fillStackAllocator_k, stack_allocator.view(), handles.view(),
                                  result.data());
      REQUIRE(result[0] == HeResult::SUCCESS);
      HERMES_CUDA_LAUNCH_AND_SYNC((1), fillStackAllocator_k, stack_allocator.view(), handles.view(),
                                  result.data());
      REQUIRE(result[0] == HeResult::SUCCESS);
    } //
    SECTION("copy from host") {
      StackAllocator h_stack(80);
      HERMES_LOG_VARIABLE(h_stack.capacityInBytes());
      DeviceStackAllocator d_stack = h_stack;
      // TODO complete this test!!!
      // allocate things in gpu and save markers in a unified array
      // copy back to host and access elements
    }//
  }//
#endif
}

TEST_CASE("ArraySlice") {
  int array[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  ArraySlice<int> slice(array, 10);
  REQUIRE(slice.size() == 10);
  int c = 0;
  for (auto a : slice)
    REQUIRE(a == c++);
}

TEST_CASE("CArray") {
  SECTION("sanity") {
    CArray<int, 10> a;
    a = 3;
    for (int i = 0; i < 10; ++i)
      REQUIRE(a[i] == 3);
    CArray<int, 10> b;
    b = a;
    for (int i = 0; i < 10; ++i)
      REQUIRE(b[i] == 3);
    REQUIRE(a == b);
    int count = 0;
    for (auto v : a) {
      REQUIRE(v == 3);
      count++;
    }
    REQUIRE(count == 10);
  }
}

TEST_CASE("DataArray", "[storage][array]") {
  SECTION("Constructors") {
    Array<int> a0;
    REQUIRE(a0.size() == size3(0, 0, 0));
    REQUIRE(a0.sizeInBytes() == 0 * sizeof(int));
    REQUIRE(a0.dimensions() == 0);
    Array<int> a1(10);
    REQUIRE(a1.size() == size3(10, 1, 1));
    REQUIRE(a1.sizeInBytes() == 10 * sizeof(int));
    REQUIRE(a1.dimensions() == 1);
    Array<int> a2({10, 20});
    REQUIRE(a2.size() == size3(10, 20, 1));
    REQUIRE(a2.sizeInBytes() == 10 * 20 * sizeof(int));
    REQUIRE(a2.dimensions() == 2);
    Array<int> a3({10, 20, 30});
    REQUIRE(a3.size() == size3(10, 20, 30));
    REQUIRE(a3.sizeInBytes() == 10 * 20 * 30 * sizeof(int));
    REQUIRE(a3.dimensions() == 3);
  }//
  SECTION("Operators") {
    SECTION("assignment") {
      Array<i32> a(10);
      for (int i = 0; i < 10; ++i)
        a[i] = i;
#ifdef HERMES_DEVICE_ENABLED
      DeviceArray<i32> dda(a);
      REQUIRE(dda.size() == size3(10, 1, 1));
      REQUIRE(dda.sizeInBytes() == 10 * sizeof(i32));
      DeviceArray<i32> da = a;
      REQUIRE(da.size() == size3(10, 1, 1));
      REQUIRE(da.sizeInBytes() == 10 * sizeof(i32));
      Array<i32> ha = da;
      for (int i = 0; i < 10; ++i)
        REQUIRE(a[i] == i);
#endif
      SECTION("std vector") {
        Array<i32> b;
        std::vector<i32> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        b = v;
        REQUIRE(b.sizeInBytes() == sizeof(i32) * v.size());
        REQUIRE(b.size() == size3(v.size(), 1, 1));
        for (int i = 0; i < v.size(); ++i)
          REQUIRE(b[i] == i + 1);
#ifdef HERMES_DEVICE_ENABLED
        DeviceArray<i32> db;
        db = v;
        REQUIRE(db.sizeInBytes() == sizeof(i32) * v.size());
        REQUIRE(db.size() == size3(v.size(), 1, 1));
        Array<i32> c = db;
        for (int i = 0; i < v.size(); ++i)
          REQUIRE(c[i] == i + 1);
#endif
      }//
    }//
    SECTION("access") {
      Array<u32> a1(10);
      for (u32 i = 0; i < 10; ++i) {
        a1[i] = i;
        REQUIRE(a1[i] == i);
      }
      HERMES_LOG_VARIABLE(a1);
      Array<i32> a2({10, 2});
      for (index2 ij : Index2Range<i32>(a2.size().slice(0, 1))) {
        a2[ij] = ij.j * 10 + ij.i;
        REQUIRE(a2[ij] == ij.j * 10 + ij.i);
      }
      HERMES_LOG_VARIABLE(a2);
      Array<i32> a3({10, 2, 3});
      for (index3 ijk : Index3Range<i32>(a3.size())) {
        a3[ijk] = ijk.k * 20 + ijk.j * 10 + ijk.i;
        REQUIRE(a3[ijk] == ijk.k * 20 + ijk.j * 10 + ijk.i);
      }
    }//
  }//
  SECTION("View") {
    DeviceArray<int> a({10, 10});
#ifdef HERMES_DEVICE_ENABLED
    HERMES_CUDA_LAUNCH_AND_SYNC((a.size()), testArrayView_k, a.view())
    Array<int> b = a;
    HERMES_LOG_VARIABLE(b);
#endif
  }//

  SECTION("Array1-iterator") {
    Array<vec2> a(10);
    int count = 0;
    for (auto e : a) {
      e.value = vec2(1, 2);
      REQUIRE(e.flat_index == count++);
    }
    REQUIRE(count == 10);
    for (auto e : a)
      REQUIRE(e.value == vec2(1, 2));
  }//
  SECTION("Array2-iterator") {
    Array<vec2> a(size2(10, 10));
    for (auto e : a)
      e.value = vec2(1, 2);
    int count = 0;
    for (auto e : a) {
      REQUIRE(e.flat_index == count++);
      REQUIRE(e.value == vec2(1, 2));
      REQUIRE(e.flat_index == e.index.j * 10 + e.index.i);
    }
    REQUIRE(count == 100);
  }//
  SECTION("Array3-iterator") {
    Array<vec2> a(size3(10, 10, 10));
    for (auto e : a)
      e.value = vec2(1, 2);
    int count = 0;
    for (auto e : a) {
      REQUIRE(e.flat_index == count++);
      REQUIRE(e.value == vec2(1, 2));
      REQUIRE(e.flat_index == e.index.k * 100 + e.index.j * 10 + e.index.i);
    }
    REQUIRE(count == 1000);
  }//

  SECTION("const Array1-iterator") {
    Array<vec2> a(10);
    for (auto e : a)
      e.value = vec2(1, 2);
    auto f = [](const Array<vec2> &array) {
      for (auto e : array)
        REQUIRE(e.value == vec2(1, 2));
    };
    f(a);
  }//
  SECTION("const Array2-iterator") {
    Array<vec2> a(size2(10, 10));
    for (auto e : a)
      e.value = vec2(1, 2);
    auto f = [](const Array<vec2> &array) {
      int count = 0;
      for (auto e : array) {
        REQUIRE(e.flat_index == count++);
        REQUIRE(e.value == vec2(1, 2));
        REQUIRE(e.flat_index == e.index.j * 10 + e.index.i);
      }
      REQUIRE(count == 100);
    };
    f(a);
  }//
  SECTION("const Array3-iterator") {
    Array<vec2> a(size3(10, 10, 10));
    for (auto e : a)
      e.value = vec2(1, 2);
    auto f = [](const Array<vec2> &array) {
      int count = 0;
      for (auto e : array) {
        REQUIRE(e.flat_index == count++);
        REQUIRE(e.value == vec2(1, 2));
        REQUIRE(e.flat_index == e.index.k * 100 + e.index.j * 10 + e.index.i);
      }
      REQUIRE(count == 1000);
    };
    f(a);
  }//
}

TEST_CASE("Array1", "[storage][array]") {
  SECTION("Constructors") {
    {
      Array1<vec2> a(10);
      REQUIRE(a.size() == 10u);
      REQUIRE(a.memorySize() == 10 * sizeof(vec2));
      for (u64 i = 0; i < a.size(); ++i) {
        a[i] = vec2(i, i * 2);
      }
      Array1<vec2> b = a;
      for (u64 i = 0; i < a.size(); ++i)
        REQUIRE(a[i] == b[i]);
    } //
    {
      std::vector<Array1<int>>
          v;
      v.emplace_back(10);
      v.emplace_back(10);
      v.emplace_back(10);
      for (int i = 0; i < 3; i++)
        for (u64 j = 0; j < v[i].size(); ++j)
          v[i][j] = j * 10;
      std::vector<Array1<int>>
          vv = v;
      for (int i = 0; i < 3; i++)
        for (u64 j = 0; j < v[i].size(); ++j)
          REQUIRE(vv[i][j] == j * 10);
    } //
    {
      Array1<int> a = std::move(Array1<int>(10));
      auto b(Array1<int>(10));
    } //
    {
      std::vector<int> data = {1, 2, 3, 4, 5, 6};
      Array1<int> a = data;
      REQUIRE(a.size() == 6);
      for (u64 i = 0; i < a.size(); ++i)
        REQUIRE(a[i] == data[i]);
    } //
    {
      Array1<int> a = {1, 2, 3};
      REQUIRE(a.size() == 3);
      for (u64 i = 0; i < a.size(); ++i)
        REQUIRE(a[i] == i + 1);
    }
  }//
  SECTION("Operators") {
    Array1<f32> a(10);
    a = -1.23323244;
    std::cerr << a;
    a = 3;
    int count = 0;
    for (u64 i = 0; i < a.size(); ++i)
      REQUIRE(a[i] == 3);
    for (auto e : a) {
      REQUIRE(e == 3);
      e = -e.index;
    }
    for (const auto &e : a) {
      REQUIRE(e.value == -e.index);
      REQUIRE(e == -e.index);
      count++;
    }
    REQUIRE(count == 10);
  }//
  SECTION("Array1-iterator") {
    Array1<vec2> a(10);
    for (auto e : a)
      e.value = vec2(1, 2);
    int count = 0;
    for (auto e : a) {
      count++;
      REQUIRE(e.value == vec2(1, 2));
    }
    REQUIRE(count == 10);
  }//
  SECTION("Const Array1-iterator") {
    Array1<vec2> a(10);
    a = vec2(1, 2);
    auto f = [](const Array1<vec2> &array) {
      for (const auto &d : array)
        REQUIRE(d.value == vec2(1, 2));
    };
    f(a);
  }//
}

TEST_CASE("Array2", "[storage][array]") {
  SECTION("Constructors") {
    {
      Array2<vec2> a(size2(10, 10));
      REQUIRE(a.pitch() == 10 * sizeof(vec2));
      REQUIRE(a.size() == size2(10, 10));
      REQUIRE(a.memorySize() == 10 * 10 * sizeof(vec2));
      for (index2 ij : Index2Range<i32>(a.size()))
        a[ij] = vec2(ij.i, ij.j);
      Array2<vec2> b = a;
      for (index2 ij : Index2Range<i32>(a.size()))
        REQUIRE(a[ij] == b[ij]);
    }
    {
      std::vector<Array2<int>>
          v;
      v.emplace_back(size2(10, 10));
      v.emplace_back(size2(10, 10));
      v.emplace_back(size2(10, 10));
      for (int i = 0; i < 3; i++)
        for (index2 ij : Index2Range<i32>(v[i].size()))
          v[i][ij] = ij.i * 10 + ij.j;
      std::vector<Array2<int>>
          vv = v;
      for (int i = 0; i < 3; i++)
        for (index2 ij : Index2Range<i32>(v[i].size()))
          REQUIRE(vv[i][ij] == ij.i * 10 + ij.j);
    }
    {
      Array2<int> a = Array2<int>(size2(10, 10));
      Array2<int> b(Array2<int>(size2(10, 10)));
    }
    {
      std::vector<std::vector<int>> data = {{1, 2, 3}, {4, 5, 6}};
      Array2<int> a = data;
      REQUIRE(a.size() == size2(3, 2));
      for (index2 ij : Index2Range<i32>(a.size()))
        REQUIRE(a[ij] == data[ij.j][ij.i]);
    }
    {
      Array2<int> a = {{1, 2, 3}, {11, 12, 13}};
      REQUIRE(a.size() == size2(3, 2));
      for (index2 ij : Index2Range<i32>(a.size()))
        REQUIRE(a[ij] == ij.j * 10 + ij.i + 1);
    }
  }//
  SECTION("Operators") {
    {
      Array2<f32> a(size2(10, 10));
      a = -1.324345455;
      std::cerr << a;
      a = 3;
      int count = 0;
      for (index2 ij : Index2Range<i32>(a.size())) {
        REQUIRE(a[ij] == 3);
        a[ij] = ij.i * 10 + ij.j;
      }
      for (const auto &e : a) {
        REQUIRE(e.value == e.index.i * 10 + e.index.j);
        REQUIRE(e == e.index.i * 10 + e.index.j);
        count++;
      }
      std::cerr << a << std::endl;
      REQUIRE(count == 10 * 10);
    }
  }//
  SECTION("Array2-iterator") {
    Array2<vec2> a(size2(10, 10));
    for (auto e : a)
      e.value = vec2(1, 2);
    int count = 0;
    for (auto e : a) {
      count++;
      REQUIRE(e.value == vec2(1, 2));
      REQUIRE(e.flatIndex() == e.index.j * 10 + e.index.i);
    }
    REQUIRE(count == 100);
  }//
  SECTION("Const Array2-iterator") {
    Array2<vec2> a(size2(10, 10));
    a = vec2(1, 2);
    auto f = [](const Array2<vec2> &array) {
      for (const auto &d : array) {
        REQUIRE(d.value == vec2(1, 2));
        REQUIRE(d.flatIndex() == d.index.j * 10 + d.index.i);
      }
    };
    f(a);
  }//
}

#ifdef HERMES_DEVICE_ENABLED
HERMES_CUDA_KERNEL(aos_view)(AoSView aos, int *result) {
  HERMES_CUDA_RETURN_IF_NOT_THREAD_0
  if (aos.size() != 5)
    *result = 1;
  for (int i = 0; i < aos.size(); ++i) {
    if (aos.valueAt<index2>(0, i) != index2(i, i + 1))
      *result = (i + 1) * 10;
    if (aos.valueAt<i32>(1, i) != -(i + 1))
      *result = -(i + 1);
  }
}

#endif

TEST_CASE("AOS", "[storage][aos]") {
  SECTION("Struct Descriptor") {
    StructDescriptor sd;
    REQUIRE(sd.pushField<vec3>("vec3") == 0);
    REQUIRE(sd.pushField<f32>("f32") == 1);
    REQUIRE(sd.pushField<int>("int") == 2);
    REQUIRE(sd.fieldName(0) == "vec3");
    REQUIRE(sd.fieldName(1) == "f32");
    REQUIRE(sd.fieldName(2) == "int");
// check fields
    auto fields = sd.fields();
    REQUIRE(fields.size() == 3);
    REQUIRE(fields[0].name == "vec3");
    REQUIRE(fields[0].size == sizeof(vec3));
    REQUIRE(fields[0].offset == 0);
    REQUIRE(fields[0].component_count == 3);
    REQUIRE(fields[0].type == DataType::F32);
    REQUIRE(fields[1].name == "f32");
    REQUIRE(fields[1].size == sizeof(f32));
    REQUIRE(fields[1].offset == sizeof(vec3));
    REQUIRE(fields[1].component_count == 1);
    REQUIRE(fields[1].type == DataType::F32);
    REQUIRE(fields[2].name == "int");
    REQUIRE(fields[2].size == sizeof(i32));
    REQUIRE(fields[2].offset == sizeof(vec3) + sizeof(f32));
    REQUIRE(fields[2].component_count == 1);
    REQUIRE(fields[2].type == DataType::I32);
    REQUIRE(sd.sizeOf("vec3") == sizeof(vec3));
    REQUIRE(sd.sizeOf("f32") == sizeof(f32));
    REQUIRE(sd.sizeOf("int") == sizeof(int));
    REQUIRE(sd.offsetOf("vec3") == 0);
    REQUIRE(sd.offsetOf("f32") == sizeof(vec3));
    REQUIRE(sd.offsetOf("int") == sizeof(vec3) + sizeof(f32));
    HERMES_LOG_VARIABLE(sd);
    { // valueAt
      AoS aos;
      aos.pushField<size2>("size2");
      aos.pushField<i32>("i32");
      aos.resize(5);

      struct SD {
        size2 s;
        i32 i{};
      };
      std::vector<SD> data(5);
      for (int i = 0; i < 5; ++i) {
        data[i].s = aos.valueAt<size2>(0, i) = {i * 3u, i * 7u};
        data[i].i = aos.valueAt<i32>(1, i) = i;
      }
      for (int i = 0; i < 5; ++i) {
        REQUIRE(aos.structDescriptor().valueAt<size2>(reinterpret_cast<const void *>(aos.data()), 0, i)
                    == size2(i * 3u, i * 7u));
        REQUIRE(aos.structDescriptor().valueAt<i32>(reinterpret_cast<const void *>(aos.data()), 1, i) == i);
        // change data
        aos.structDescriptor().valueAt<size2>(reinterpret_cast<void *>(data.data()), 0, i) = {i * 5u, i * 13u};
        aos.structDescriptor().valueAt<i32>(reinterpret_cast<void *>(data.data()), 1, i) = -i;
      }
      for (int i = 0; i < 5; ++i) {
        REQUIRE(
            aos.structDescriptor().valueAt<size2>(reinterpret_cast<const void *>(data.data()), 0, i)
                == size2(i * 5u, i * 13u));
        REQUIRE(aos.structDescriptor().valueAt<i32>(reinterpret_cast<const void *>(data.data()), 1, i) == -i);
      }
    }
  }//
  SECTION("Sanity Checks") {
    AoS aos;
    REQUIRE(aos.pushField<vec3>("vec3") == 0);
    REQUIRE(aos.pushField<f32>("f32") == 1);
    REQUIRE(aos.pushField<int>("int") == 2);
    REQUIRE(aos.structDescriptor().fieldName(0) == "vec3");
    REQUIRE(aos.structDescriptor().fieldName(1) == "f32");
    REQUIRE(aos.structDescriptor().fieldName(2) == "int");
    REQUIRE(aos.size() == 0);
// check fields
    auto fields = aos.fields();
    REQUIRE(fields.size() == 3);
    REQUIRE(fields[0].name == "vec3");
    REQUIRE(fields[0].size == sizeof(vec3));
    REQUIRE(fields[0].offset == 0);
    REQUIRE(fields[0].component_count == 3);
    REQUIRE(fields[0].type == DataType::F32);
    REQUIRE(fields[1].name == "f32");
    REQUIRE(fields[1].size == sizeof(f32));
    REQUIRE(fields[1].offset == sizeof(vec3));
    REQUIRE(fields[1].component_count == 1);
    REQUIRE(fields[1].type == DataType::F32);
    REQUIRE(fields[2].name == "int");
    REQUIRE(fields[2].size == sizeof(i32));
    REQUIRE(fields[2].offset == sizeof(vec3) + sizeof(f32));
    REQUIRE(fields[2].component_count == 1);
    REQUIRE(fields[2].type == DataType::I32);
    aos.resize(4);
    REQUIRE(aos.size() == 4);
    REQUIRE(aos.stride() == sizeof(vec3) + sizeof(f32) + sizeof(int));
    REQUIRE(aos.structDescriptor().sizeOf("vec3") == sizeof(vec3));
    REQUIRE(aos.structDescriptor().sizeOf("f32") == sizeof(f32));
    REQUIRE(aos.structDescriptor().sizeOf("int") == sizeof(int));
    REQUIRE(aos.structDescriptor().offsetOf("vec3") == 0);
    REQUIRE(aos.structDescriptor().offsetOf("f32") == sizeof(vec3));
    REQUIRE(aos.structDescriptor().offsetOf("int") == sizeof(vec3) + sizeof(f32));
    REQUIRE(aos.memorySizeInBytes() == aos.stride() * 4);
    for (int i = 0; i < 4; ++i) {
      aos.valueAt<vec3>(0, i) = {1.f + i, 2.f + i, 3.f + i};
      aos.valueAt<f32>(1, i) = 1.f * i;
      aos.valueAt<int>(2, i) = i + 1;
    }
    for (int i = 0; i < 4; ++i) {
      REQUIRE(aos.valueAt<vec3>(0, i) == vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE(aos.valueAt<f32>(1, i) == Approx(1.f * i));
      REQUIRE(aos.valueAt<int>(2, i) == i + 1);
    }
    HERMES_LOG_VARIABLE(aos);
  }//
  SECTION("change description") {
    StructDescriptor desc;
    REQUIRE(desc.pushField<vec3>("vec3") == 0);
    REQUIRE(desc.pushField<f32>("f32") == 1);
    REQUIRE(desc.pushField<int>("int") == 2);
// check fields
    AoS aos;
    aos.setStructDescriptor(desc);
    auto fields = aos.fields();
    REQUIRE(fields.size() == 3);
    REQUIRE(fields[0].name == "vec3");
    REQUIRE(fields[0].size == sizeof(vec3));
    REQUIRE(fields[0].offset == 0);
    REQUIRE(fields[0].component_count == 3);
    REQUIRE(fields[0].type == DataType::F32);
    REQUIRE(fields[1].name == "f32");
    REQUIRE(fields[1].size == sizeof(f32));
    REQUIRE(fields[1].offset == sizeof(vec3));
    REQUIRE(fields[1].component_count == 1);
    REQUIRE(fields[1].type == DataType::F32);
    REQUIRE(fields[2].name == "int");
    REQUIRE(fields[2].size == sizeof(i32));
    REQUIRE(fields[2].offset == sizeof(vec3) + sizeof(f32));
    REQUIRE(fields[2].component_count == 1);
    REQUIRE(fields[2].type == DataType::I32);
  }//
  SECTION("push new fields") {
    AoS aos;
    aos.pushField<int>();
    aos.pushField<hermes::vec2>();
    aos.resize(5);
    for (int i = 0; i < aos.size(); ++i) {
      aos.valueAt<int>(0, i) = i;
      aos.valueAt<hermes::vec2>(1, i) = {i * 0.1f, -i * 1.f};
    }
    aos.pushField<int>();
    REQUIRE(aos.memorySizeInBytes() == 5 * (sizeof(int) + sizeof(hermes::vec2) + sizeof(int)));
    for (int i = 0; i < aos.size(); ++i) {
      REQUIRE(aos.valueAt<int>(0, i) == i);
      REQUIRE(aos.valueAt<hermes::vec2>(1, i) == hermes::vec2(i * 0.1f, -i * 1.f));
    }
  }//
  SECTION("Access") {
    AoS aos;
    aos.pushField<vec3>("vec3");
    aos.pushField<f32>("f32");
    aos.pushField<int>("int");
    aos.resize(4);
    auto vec3_field = aos.field<vec3>("vec3");
    auto f32_field = aos.field<f32>("f32");
    auto int_field = aos.field<int>("int");
    for (int i = 0; i < 4; ++i) {
      vec3_field[i] = {1.f + i, 2.f + i, 3.f + i};
      f32_field[i] = 1.f * i;
      int_field[i] = i + 1;
    }
    for (int i = 0; i < 4; ++i) {
      REQUIRE(aos.valueAt<vec3>(0, i) == vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE(aos.valueAt<f32>(1, i) == Approx(1.f * i));
      REQUIRE(aos.valueAt<int>(2, i) == i + 1);
      REQUIRE(vec3_field[i] == vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE(f32_field[i] == 1.f * i);
      REQUIRE(int_field[i] == i + 1);
    }
    REQUIRE(aos.back<vec3>(0) == vec3(1.f + 3, 2.f + 3, 3.f + 3));
    REQUIRE(aos.back<f32>(1) == Approx(1.f * 3));
    REQUIRE(aos.back<int>(2) == 3 + 1);
  }//
  SECTION("Accessors") {
    AoS aos;
    aos.pushField<vec3>("vec3");
    aos.pushField<f32>("f32");
    aos.pushField<int>("int");
    aos.resize(4);
    auto acc = aos.view();
    for (int i = 0; i < 4; ++i) {
      acc.valueAt<vec3>(0, i) = {1.f + i, 2.f + i, 3.f + i};
      acc.valueAt<f32>(1, i) = 1.f * i;
      acc.valueAt<int>(2, i) = i + 1;
    }
    for (int i = 0; i < 4; ++i) {
      REQUIRE(acc.valueAt<vec3>(0, i) == vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE(acc.valueAt<f32>(1, i) == Approx(1.f * i));
      REQUIRE(acc.valueAt<int>(2, i) == i + 1);
    }
    auto cacc = aos.constView();
    for (int i = 0; i < 4; ++i) {
      REQUIRE(cacc.valueAt<vec3>(0, i) == vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE(cacc.valueAt<f32>(1, i) == Approx(1.f * i));
      REQUIRE(cacc.valueAt<int>(2, i) == i + 1);
    }
    AoS aos2;
    aos2.pushField<vec3>("vec3");
    aos2.pushField<f32>("f32");
    aos2.pushField<int>("int");
    aos2.resize(4);
    for (int i = 0; i < 4; ++i) {
      aos2.valueAt<vec3>(0, i) = {-1.f + i, -2.f + i, -3.f + i};
      aos2.valueAt<f32>(1, i) = -1.f * i;
      aos2.valueAt<int>(2, i) = i - 1;
    }
    cacc.setDataPtr(aos2.data());
    for (int i = 0; i < 4; ++i) {
      REQUIRE(cacc.valueAt<vec3>(0, i) == vec3(-1.f + i, -2.f + i, -3.f + i));
      REQUIRE(cacc.valueAt<f32>(1, i) == Approx(-1.f * i));
      REQUIRE(cacc.valueAt<int>(2, i) == i - 1);
    }
  }//
  SECTION("Field Accessors") {
    AoS aos;
    aos.pushField<size2>("sizes");
    aos.pushField<i32>("i32");
    aos.resize(5);
    auto sizes_field = aos.field<size2>(0) = {
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
    };
    auto i32_field = aos.field<i32>(1) = {-1, -2, -3, -4, -5};
    for (int i = 0; i < 5; ++i) {
      REQUIRE(sizes_field[i] == size2(i, i + 1));
      REQUIRE(i32_field[i] == -(i + 1));
    }

  }//
  SECTION("File") {
    AoS aos;
    aos.pushField<vec3>("vec3");
    aos.pushField<f32>("f32");
    aos.pushField<int>("int");
    aos.resize(4);
    auto acc = aos.view();
    for (int i = 0; i < 4; ++i) {
      acc.valueAt<vec3>(0, i) = {1.f + i, 2.f + i, 3.f + i};
      acc.valueAt<f32>(1, i) = 1.f * i;
      acc.valueAt<int>(2, i) = i + 1;
    }
    std::ofstream file_out("aos_data", std::ios::binary);
    file_out << aos;
    file_out.close();
    AoS aos2;
    std::ifstream file_in("aos_data", std::ios::binary | std::ios::in);
    file_in >> aos2;
    file_in.close();
    REQUIRE(aos.size() == aos2.size());
    REQUIRE(aos.memorySizeInBytes() == aos2.memorySizeInBytes());
    REQUIRE(aos.stride() == aos2.stride());
    auto acc2 = aos2.view();
    for (int i = 0; i < 4; ++i) {
      REQUIRE(acc2.valueAt<vec3>(0, i).x == Approx(acc.valueAt<vec3>(0, i).x));
      REQUIRE(acc2.valueAt<vec3>(0, i).y == Approx(acc.valueAt<vec3>(0, i).y));
      REQUIRE(acc2.valueAt<vec3>(0, i).z == Approx(acc.valueAt<vec3>(0, i).z));
      REQUIRE(acc2.valueAt<f32>(1, i) == Approx(acc.valueAt<f32>(1, i)));
      REQUIRE(acc2.valueAt<int>(2, i) == acc.valueAt<int>(2, i));
    }
    auto fields = aos.fields();
    for (auto f : fields) {
      REQUIRE(aos.structDescriptor().contains(f.name));
      REQUIRE(aos2.structDescriptor().contains(f.name));
      REQUIRE(aos.structDescriptor().fieldId(f.name) == aos2.structDescriptor().fieldId(f.name));
    }
  } //
#ifdef HERMES_DEVICE_ENABLED
  SECTION("Device") {
    AoS aos;
    aos.pushField<size2>();
    aos.pushField<i32>();
    aos.resize(5);
    auto sizes_field = aos.field<size2>(0) = {
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
    };
    auto i32_field = aos.field<i32>(1) = {-1, -2, -3, -4, -5};
    DeviceAoS d_aos = aos;
    REQUIRE(d_aos.size() == aos.size());
    REQUIRE(d_aos.structDescriptor().fields().size() == aos.structDescriptor().fields().size());
    REQUIRE(d_aos.structDescriptor().sizeInBytes() == aos.structDescriptor().sizeInBytes());

    UnifiedArray<int> results(1);
    HERMES_CUDA_LAUNCH_AND_SYNC((1), aos_view_k, d_aos.view(), results.data())
    REQUIRE(results[0] == 0);
  }//
#endif
}