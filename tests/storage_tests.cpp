#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/geometry/vector.h>
#include <hermes/storage/aos.h>
#include <hermes/storage/block.h>
#include <hermes/system/gpu.h>

#include <fstream>

using namespace hermes;
using namespace hermes::mem;

#ifdef HERMES_DEVICE_ENABLED
HERMES_CUDA_KERNEL(writeMatrixIndex)(u32 *data, size2 bounds) {
  HERMES_CUDA_THREAD_INDEX_IJ_LT(bounds);
  u32 matrix_index = ij.j * bounds.width + ij.i;
  data[matrix_index] = matrix_index;
}
#endif

/*
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
  HERMES_CPU_GPU
  void say() const { printf("ah %d %ul!\n", a, data.size().total()); }
  int a = 10;
  Block<MemoryLocation::DEVICE> data;
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

*/
TEST_CASE("Block", "[storage]") {
  auto writeHostMemory = [](Block &hm) {
    u8 *ptr = reinterpret_cast<u8 *>(hm.data());
    for (u8 i = 0; i < static_cast<u8>(hm.sizeInBytes()); ++i)
      ptr[i] = i;
  };
  auto checkHostMemory = [](Block &hm) -> bool {
    u8 *ptr = reinterpret_cast<u8 *>(hm.data());
    for (u8 i = 0; i < static_cast<u8>(hm.sizeInBytes()); ++i)
      if (ptr[i] != i)
        return false;
    return true;
  };
  auto checkDeviceMemory = [&](Block &dm) -> bool {
    Block hm = dm;
    return checkHostMemory(hm);
  };

  HERMES_UNUSED_VARIABLE(checkDeviceMemory);

  SECTION("copy") {
    SECTION("host host") {
      {
        auto src = Block::Config().setSize(100).create().value();
        writeHostMemory(src);
        auto dst = Block::Config().setSize(100).create().value();
        REQUIRE(dst.copy(src) == HeError::NO_ERROR);
        checkHostMemory(dst);
      }
      {
        struct CopyTest {
          int a;
          int b;
        };

        Block hm;
        REQUIRE(hm.resize(sizeof(CopyTest)) == HeError::NO_ERROR);
        CopyTest ct;
        ct.a = 3;
        ct.b = 6;
        REQUIRE(hm.copy(&ct) == HeError::NO_ERROR);
        CopyTest *d = reinterpret_cast<CopyTest *>(hm.data());
        REQUIRE(d->a == ct.a);
        REQUIRE(d->b == ct.b);
      }
    }
#ifdef HERMES_DEVICE_ENABLED
    SECTION("host device") {
      Block src, dst;
      HERMES_ASSIGN_OR(src, Block::Config().setSize(100).create(),
                       REQUIRE(false));
      writeHostMemory(src);
      HERMES_ASSIGN_OR(dst,
                       Block::Config()
                           .setSize(100)
                           .setLocation(MemoryLocation::DEVICE)
                           .create(),
                       REQUIRE(false));
      REQUIRE(dst.copy(src) == HeError::NO_ERROR);
      REQUIRE(checkDeviceMemory(dst) == true);
    }
#endif
  } //
  SECTION("assignment") {
    SECTION("host") {
      auto base = Block::Config().setSize(100).create().value();
      writeHostMemory(base);
      Block cpy = base;
      REQUIRE(base.sizeInBytes() == 100);
      REQUIRE(cpy.sizeInBytes() == 100);
      REQUIRE(cpy.sizeInBytes() == base.sizeInBytes());
      REQUIRE(checkHostMemory(base));
      REQUIRE(checkHostMemory(cpy));
      Block mv = std::move(base);
      REQUIRE(base.sizeInBytes() == 0);
      REQUIRE(mv.sizeInBytes() == 100);
      REQUIRE(checkHostMemory(mv));
    } //
  } //
  SECTION("resize") {
    SECTION("host") {
      Block hm;
      REQUIRE(hm.sizeInBytes() == 0);
      REQUIRE(hm.resize(100) == HeError::NO_ERROR);
      writeHostMemory(hm);
      REQUIRE(hm.sizeInBytes() == 100);
      REQUIRE(checkHostMemory(hm));
      REQUIRE(hm.resize({64, 2}, 0) == HeError::NO_ERROR);
      writeHostMemory(hm);
      REQUIRE(hm.sizeInBytes() == 128);
      REQUIRE(checkHostMemory(hm));
      REQUIRE(hm.resize({32, 4, 2}, 0) == HeError::NO_ERROR);
      writeHostMemory(hm);
      REQUIRE(hm.sizeInBytes() == 256);
      REQUIRE(checkHostMemory(hm));
    } //
#ifdef HERMES_DEVICE_ENABLED
    SECTION("device") {
      auto dm_r = Block::Config().setLocation(MemoryLocation::DEVICE).create();
      REQUIRE((bool)dm_r);
      auto dm = dm_r.value();
      REQUIRE(dm.sizeInBytes() == 0);
      REQUIRE(dm.resize(256) == HeError::NO_ERROR);
      auto hm = Block::Config().setSize(256).create().value();
      writeHostMemory(hm);
      dm = hm;
      REQUIRE(dm.sizeInBytes() == 256);
      REQUIRE(dm.size() == hm.size());
      REQUIRE(checkDeviceMemory(dm));
    } //
#endif
  } //
  SECTION("linear block") {
    SECTION("host") {
      auto hm = Block::Config().setSize(256).create().value();
      REQUIRE(hm.sizeInBytes() == 256);
      writeHostMemory(hm);
      REQUIRE(checkHostMemory(hm));
    } //
    SECTION("device") {
#ifdef HERMES_DEVICE_ENABLED
      auto src = Block::Config().setSize(100).create().value();
      writeHostMemory(src);
      auto dst_r = Block::Config()
                       .setSize(100)
                       .setLocation(MemoryLocation::DEVICE)
                       .create();
      REQUIRE((bool)dst_r);
      auto dst = dst_r.value();
      REQUIRE(dst.copy(src) == HeError::NO_ERROR);
      REQUIRE(dst.sizeInBytes() == 100);
#endif
    } //
  } //
  SECTION("2d block") {
    SECTION("host") {
      auto hm = Block::Config().setSize({32, 8}).create().value();
      REQUIRE(hm.sizeInBytes() == 256);
      writeHostMemory(hm);
      REQUIRE(checkHostMemory(hm));
    } //
  } //
  SECTION("3d block") {
    SECTION("host") {
      auto hm = Block::Config().setSize({32, 4, 2}).create().value();
      REQUIRE(hm.sizeInBytes() == 256);
      writeHostMemory(hm);
      REQUIRE(checkHostMemory(hm));
    } //
  } //
#ifdef HERMES_DEVICE_ENABLED
  SECTION("unified") {
    auto um_r = Block::Config()
                    .setLocation(MemoryLocation::UNIFIED)
                    .setSize(64 * 128 * 4)
                    .create();
    REQUIRE((bool)um_r);
    auto um = um_r.value();
    u32 *data = reinterpret_cast<u32 *>(um.data());
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
  REQUIRE(sizes::cache_l1_size == 64);
  SECTION("alignTo") {
    REQUIRE(alignment::alignTo(1, sizeof(u8)) == sizeof(u8));
    REQUIRE(alignment::alignTo(1, sizeof(u16)) == sizeof(u16));
    REQUIRE(alignment::alignTo(1, sizeof(u32)) == sizeof(u32));
    REQUIRE(alignment::alignTo(1, sizeof(u64)) == sizeof(u64));
    struct S {
      f32 a;
      u8 b;
      u16 c;
    };
    REQUIRE(sizeof(S) == 8);
    REQUIRE(alignment::alignTo(15, sizeof(S)) == 16);
    REQUIRE(alignment::alignTo(17, sizeof(S)) == 24);
  } //
  SECTION("left and right alignments") {
    REQUIRE(alignment::leftAlignShift(100, 64) == 100 - 64);
    REQUIRE(alignment::rightAlignShift(100, 64) == 128 - 100);

    REQUIRE(alignment::leftAlignShift(100, 1) == 0);
    REQUIRE(alignment::rightAlignShift(100, 1) == 0);
  } //
  SECTION("allocAligned") {
    auto *ptr = allocation::allocAligned(10, 1);
    allocation::freeAligned(ptr);
  } //
}
/*
#ifdef HERMES_DEVICE_ENABLED
HERMES_CUDA_KERNEL(fillStackAllocator)(StackAllocatorView stack_allocator,
                                       ArrayView<AddressIndex> handles,
                                       HeResult *result) {
  HERMES_CUDA_RETURN_IF_NOT_THREAD_0
  for (u32 i = 0; i < 20; ++i)
    handles.emplace(i, stack_allocator.pushAligned<int>(0));
  for (u32 i = 0; i < 20; ++i) {
    *result = stack_allocator.set(handles[i], i);
    if (*result != HeResult::SUCCESS)
      return;
  }
}

HERMES_CUDA_KERNEL(checkStackAllocator)(StackAllocatorView stack_allocator,
                                        ArrayView<AddressIndex> handles,
                                        HeResult *result) {
  HERMES_CUDA_RETURN_IF_NOT_THREAD_0
  for (u32 i = 0; i < 20; ++i)
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
    } //
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
    } //
    SECTION("debug") {
#ifdef ODYSSEUS_DEBUG
      StackAllocator stack_allocator(200);
      stack_allocator.allocate(10);
      stack_allocator.allocate(50);
      stack_allocator.allocate(80, 64);
      stack_allocator.dump();
#endif
    } //
    SECTION("set get") {
      StackAllocator stack_allocator(80);
      std::vector<AddressIndex> handles;
      handles.reserve(20);
      for (u32 i = 0; i < 20; ++i)
        handles.emplace_back(stack_allocator.pushAligned<int>(0));
      for (u32 i = 0; i < 20; ++i)
        REQUIRE(stack_allocator.set(handles[i], i) == HeResult::SUCCESS);
      for (u32 i = 0; i < 20; ++i)
        REQUIRE(*stack_allocator.get<int>(handles[i]) == i);
    } //
    SECTION("view") {
      StackAllocator stack_allocator(80);
      std::vector<AddressIndex> handles;
      handles.reserve(20);
      for (u32 i = 0; i < 20; ++i)
        handles.emplace_back(stack_allocator.pushAligned<int>(0));
      for (u32 i = 0; i < 20; ++i)
        REQUIRE(stack_allocator.set(handles[i], i) == HeResult::SUCCESS);
      auto view = stack_allocator.view();
      for (u32 i = 0; i < 20; ++i)
        REQUIRE(*view.get<int>(handles[i]) == i);
      for (u32 i = 0; i < 20; ++i)
        REQUIRE(view.set(handles[i], 2 * i) == HeResult::SUCCESS);
      for (u32 i = 0; i < 20; ++i)
        REQUIRE(*stack_allocator.get<int>(handles[i]) == 2 * i);
    } //
  } //
#ifdef HERMES_DEVICE_ENABLED
  SECTION("UNIFIED") {
    SECTION("empty") {
      UnifiedStackAllocator stack_allocator;
      REQUIRE(stack_allocator.capacityInBytes() == 0);
      REQUIRE(stack_allocator.availableSizeInBytes() == 0);
      REQUIRE(stack_allocator.allocate(10).id == 0);
      REQUIRE(stack_allocator.allocateAligned<int>().id == 0);
      REQUIRE(stack_allocator.freeTo({}) == HeResult::BAD_OPERATION);
    } //
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
    } //
    SECTION("debug") {
#ifdef ODYSSEUS_DEBUG
      UnifiedStackAllocator stack_allocator(200);
      stack_allocator.allocate(10);
      stack_allocator.allocate(50);
      stack_allocator.allocate(80, 64);
      stack_allocator.dump();
#endif
    } //
    SECTION("set get") {
      UnifiedStackAllocator stack_allocator(80);
      std::vector<AddressIndex> handles;
      handles.reserve(20);
      for (u32 i = 0; i < 20; ++i)
        handles.emplace_back(stack_allocator.allocateAligned<int>(0));
      for (u32 i = 0; i < 20; ++i)
        REQUIRE(stack_allocator.set(handles[i], i) == HeResult::SUCCESS);
#ifdef ODYSSEUS_DEBUG
      stack_allocator.dump();
#endif
      for (u32 i = 0; i < 20; ++i)
        REQUIRE(*stack_allocator.get<int>(handles[i]) == i);
    }
  } //
  SECTION("DEVICE") {
    SECTION("sanity") {
      DeviceStackAllocator stack_allocator(80);
      DeviceArray<AddressIndex> handles(20);
      UnifiedArray<HeResult> result(1);
      HERMES_CUDA_LAUNCH_AND_SYNC((1), fillStackAllocator_k,
                                  stack_allocator.view(), handles.view(),
                                  result.data());
      REQUIRE(result[0] == HeResult::SUCCESS);
      HERMES_CUDA_LAUNCH_AND_SYNC((1), fillStackAllocator_k,
                                  stack_allocator.view(), handles.view(),
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
    } //
  } //
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
    for (u32 i = 0; i < 10; ++i)
      REQUIRE(a[i] == 3);
    CArray<int, 10> b;
    b = a;
    for (u32 i = 0; i < 10; ++i)
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
  } //
  SECTION("Operators") {
    SECTION("assignment") {
      Array<i32> a(10);
      for (u32 i = 0; i < 10; ++i)
        a[i] = i;
#ifdef HERMES_DEVICE_ENABLED
      DeviceArray<i32> dda(a);
      REQUIRE(dda.size() == size3(10, 1, 1));
      REQUIRE(dda.sizeInBytes() == 10 * sizeof(i32));
      DeviceArray<i32> da = a;
      REQUIRE(da.size() == size3(10, 1, 1));
      REQUIRE(da.sizeInBytes() == 10 * sizeof(i32));
      Array<i32> ha = da;
      for (u32 i = 0; i < 10; ++i)
        REQUIRE(a[i] == i);
#endif
      SECTION("std vector") {
        Array<i32> b;
        std::vector<i32> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        b = v;
        REQUIRE(b.sizeInBytes() == sizeof(i32) * v.size());
        REQUIRE(b.size() == size3(v.size(), 1, 1));
        for (u32 i = 0; i < v.size(); ++i)
          REQUIRE(b[i] == i + 1);
#ifdef HERMES_DEVICE_ENABLED
        DeviceArray<i32> db;
        db = v;
        REQUIRE(db.sizeInBytes() == sizeof(i32) * v.size());
        REQUIRE(db.size() == size3(v.size(), 1, 1));
        Array<i32> c = db;
        for (u32 i = 0; i < v.size(); ++i)
          REQUIRE(c[i] == i + 1);
#endif
      } //
    } //
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
    } //
  } //
  SECTION("View") {
    DeviceArray<int> a({10, 10});
#ifdef HERMES_DEVICE_ENABLED
    HERMES_CUDA_LAUNCH_AND_SYNC((a.size()), testArrayView_k, a.view())
    Array<int> b = a;
    HERMES_LOG_VARIABLE(b);
#endif
  } //

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
  } //
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
  } //
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
  } //

  SECTION("const Array1-iterator") {
    Array<vec2> a(10);
    for (auto e : a)
      e.value = vec2(1, 2);
    auto f = [](const Array<vec2> &array) {
      for (auto e : array)
        REQUIRE(e.value == vec2(1, 2));
    };
    f(a);
  } //
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
  } //
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
  } //
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
      std::vector<Array1<int>> v;
      v.emplace_back(10);
      v.emplace_back(10);
      v.emplace_back(10);
      for (u32 i = 0; i < 3; i++)
        for (u64 j = 0; j < v[i].size(); ++j)
          v[i][j] = j * 10;
      std::vector<Array1<int>> vv = v;
      for (u32 i = 0; i < 3; i++)
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
  } //
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
  } //
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
  } //
  SECTION("Const Array1-iterator") {
    Array1<vec2> a(10);
    a = vec2(1, 2);
    auto f = [](const Array1<vec2> &array) {
      for (const auto &d : array)
        REQUIRE(d.value == vec2(1, 2));
    };
    f(a);
  } //
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
      std::vector<Array2<int>> v;
      v.emplace_back(size2(10, 10));
      v.emplace_back(size2(10, 10));
      v.emplace_back(size2(10, 10));
      for (u32 i = 0; i < 3; i++)
        for (index2 ij : Index2Range<i32>(v[i].size()))
          v[i][ij] = ij.i * 10 + ij.j;
      std::vector<Array2<int>> vv = v;
      for (u32 i = 0; i < 3; i++)
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
  } //
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
  } //
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
  } //
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
  } //
}
*/

#ifdef HERMES_DEVICE_ENABLED
// HERMES_CUDA_KERNEL(aos_view)(AoSView aos, int *result) {
//   HERMES_CUDA_RETURN_IF_NOT_THREAD_0
//   if (aos.size() != 5)
//     *result = 1;
//   for (u32 i = 0; i < aos.size(); ++i) {
//     if (aos.get<index2>(0, i) != index2(i, i + 1))
//       *result = (i + 1) * 10;
//     if (aos.get<i32>(1, i) != -(i + 1))
//       *result = -(i + 1);
//   }
// }

#endif

class CustomAoS : public AoS {};

template <typename T> class CustomAoSFieldView : public AoS::FieldView<T> {
public:
  CustomAoSFieldView(AoS::FieldView<T> f) : AoS::FieldView<T>(f) {}
  int new_field;
};

template <typename T>
class CustomAoSConstFieldView : public AoS::ConstFieldView<T> {
public:
  CustomAoSConstFieldView(AoS::FieldView<T> f) : AoS::ConstFieldView<T>(f) {}
  int new_field;
};

TEST_CASE("AOS", "[storage][aos]") {
  SECTION("Struct Descriptor") {
    AoS::Layout sd;
    REQUIRE(sd.pushField<geo::vec3>("geo::vec3") == 0);
    REQUIRE(sd.pushField<f32>("f32") == 1);
    REQUIRE(sd.pushField<int>("int") == 2);
    REQUIRE(sd.fieldName(0) == "geo::vec3");
    REQUIRE(sd.fieldName(1) == "f32");
    REQUIRE(sd.fieldName(2) == "int");
    // check fields
    auto fields = sd.fields();
    REQUIRE(fields.size() == 3);
    REQUIRE(fields[0].name == "geo::vec3");
    REQUIRE(fields[0].size == sizeof(geo::vec3));
    REQUIRE(fields[0].offset == 0);
    REQUIRE(fields[0].component_count == 3);
    REQUIRE(fields[0].type == DataType::F32);
    REQUIRE(fields[1].name == "f32");
    REQUIRE(fields[1].size == sizeof(f32));
    REQUIRE(fields[1].offset == sizeof(geo::vec3));
    REQUIRE(fields[1].component_count == 1);
    REQUIRE(fields[1].type == DataType::F32);
    REQUIRE(fields[2].name == "int");
    REQUIRE(fields[2].size == sizeof(i32));
    REQUIRE(fields[2].offset == sizeof(geo::vec3) + sizeof(f32));
    REQUIRE(fields[2].component_count == 1);
    REQUIRE(fields[2].type == DataType::I32);
    REQUIRE(sd.sizeOf("geo::vec3") == sizeof(geo::vec3));
    REQUIRE(sd.sizeOf("f32") == sizeof(f32));
    REQUIRE(sd.sizeOf("int") == sizeof(int));
    REQUIRE(sd.offsetOf("geo::vec3") == 0);
    REQUIRE(sd.offsetOf("f32") == sizeof(geo::vec3));
    REQUIRE(sd.offsetOf("int") == sizeof(geo::vec3) + sizeof(f32));
    HERMES_LOG_VARIABLE(sd);
    { // get
      AoS aos;
      aos.pushField<size2>("size2");
      aos.pushField<i32>("i32");
      REQUIRE(aos.resize(5) == HeError::NO_ERROR);

      struct SD {
        size2 s;
        i32 i{};
      };
      std::vector<SD> data(5);
      for (i32 i = 0; i < 5; ++i) {
        data[i].s = aos.get<size2>(0, i) = {i * 3u, i * 7u};
        data[i].i = aos.get<i32>(1, i) = i;
      }
      for (i32 i = 0; i < 5; ++i) {
        REQUIRE(
            aos.layout().get<size2>(reinterpret_cast<const void *>(*aos.data()),
                                    0, i) == size2(i * 3u, i * 7u));
        REQUIRE(aos.layout().get<i32>(
                    reinterpret_cast<const void *>(*aos.data()), 1, i) == i);
        // change data
        aos.layout().get<size2>(reinterpret_cast<void *>(data.data()), 0,
                                i) = {i * 5u, i * 13u};
        aos.layout().get<i32>(reinterpret_cast<void *>(data.data()), 1, i) = -i;
      }
      for (i32 i = 0; i < 5; ++i) {
        REQUIRE(
            aos.layout().get<size2>(reinterpret_cast<const void *>(data.data()),
                                    0, i) == size2(i * 5u, i * 13u));
        REQUIRE(aos.layout().get<i32>(
                    reinterpret_cast<const void *>(data.data()), 1, i) == -i);
      }
    }
  } //
  SECTION("Sanity Checks") {
    AoS aos;
    REQUIRE(aos.pushField<geo::vec3>("geo::vec3") == 0);
    REQUIRE(aos.pushField<f32>("f32") == 1);
    REQUIRE(aos.pushField<int>("int") == 2);
    REQUIRE(aos.layout().fieldName(0) == "geo::vec3");
    REQUIRE(aos.layout().fieldName(1) == "f32");
    REQUIRE(aos.layout().fieldName(2) == "int");
    REQUIRE(aos.size() == 0);
    // check fields
    auto fields = aos.layout().fields();
    REQUIRE(fields.size() == 3);
    REQUIRE(fields[0].name == "geo::vec3");
    REQUIRE(fields[0].size == sizeof(geo::vec3));
    REQUIRE(fields[0].offset == 0);
    REQUIRE(fields[0].component_count == 3);
    REQUIRE(fields[0].type == DataType::F32);
    REQUIRE(fields[1].name == "f32");
    REQUIRE(fields[1].size == sizeof(f32));
    REQUIRE(fields[1].offset == sizeof(geo::vec3));
    REQUIRE(fields[1].component_count == 1);
    REQUIRE(fields[1].type == DataType::F32);
    REQUIRE(fields[2].name == "int");
    REQUIRE(fields[2].size == sizeof(i32));
    REQUIRE(fields[2].offset == sizeof(geo::vec3) + sizeof(f32));
    REQUIRE(fields[2].component_count == 1);
    REQUIRE(fields[2].type == DataType::I32);
    REQUIRE(aos.resize(4) == HeError::NO_ERROR);
    REQUIRE(aos.size() == 4);
    REQUIRE(aos.stride() == sizeof(geo::vec3) + sizeof(f32) + sizeof(int));
    REQUIRE(aos.layout().sizeOf("geo::vec3") == sizeof(geo::vec3));
    REQUIRE(aos.layout().sizeOf("f32") == sizeof(f32));
    REQUIRE(aos.layout().sizeOf("int") == sizeof(int));
    REQUIRE(aos.layout().offsetOf("geo::vec3") == 0);
    REQUIRE(aos.layout().offsetOf("f32") == sizeof(geo::vec3));
    REQUIRE(aos.layout().offsetOf("int") == sizeof(geo::vec3) + sizeof(f32));
    REQUIRE(aos.dataSize() == aos.stride() * 4);
    for (i32 i = 0; i < 4; ++i) {
      aos.get<geo::vec3>(0, i) = {1.f + i, 2.f + i, 3.f + i};
      aos.get<f32>(1, i) = 1.f * i;
      aos.get<int>(2, i) = i + 1;
    }
    for (i32 i = 0; i < 4; ++i) {
      REQUIRE(aos.get<geo::vec3>(0, i) == geo::vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE_THAT(aos.get<f32>(1, i),
                   Catch::Matchers::WithinAbs(1.f * i, 1e-8));
      REQUIRE(aos.get<int>(2, i) == i + 1);
    }
    HERMES_LOG_VARIABLE(aos);
  } //
  SECTION("change description") {
    AoS::Layout desc;
    REQUIRE(desc.pushField<geo::vec3>("geo::vec3") == 0);
    REQUIRE(desc.pushField<f32>("f32") == 1);
    REQUIRE(desc.pushField<int>("int") == 2);
    // check fields
    AoS aos;
    REQUIRE(aos.setLayout(desc) == HeError::NO_ERROR);
    auto fields = aos.layout().fields();
    REQUIRE(fields.size() == 3);
    REQUIRE(fields[0].name == "geo::vec3");
    REQUIRE(fields[0].size == sizeof(geo::vec3));
    REQUIRE(fields[0].offset == 0);
    REQUIRE(fields[0].component_count == 3);
    REQUIRE(fields[0].type == DataType::F32);
    REQUIRE(fields[1].name == "f32");
    REQUIRE(fields[1].size == sizeof(f32));
    REQUIRE(fields[1].offset == sizeof(geo::vec3));
    REQUIRE(fields[1].component_count == 1);
    REQUIRE(fields[1].type == DataType::F32);
    REQUIRE(fields[2].name == "int");
    REQUIRE(fields[2].size == sizeof(i32));
    REQUIRE(fields[2].offset == sizeof(geo::vec3) + sizeof(f32));
    REQUIRE(fields[2].component_count == 1);
    REQUIRE(fields[2].type == DataType::I32);
  } //
  SECTION("push new fields") {
    AoS aos;
    aos.pushField<int>();
    aos.pushField<hermes::geo::vec2>();
    REQUIRE(aos.resize(5) == HeError::NO_ERROR);
    for (u32 i = 0; i < aos.size(); ++i) {
      aos.get<int>(0, i) = i;
      aos.get<hermes::geo::vec2>(1, i) = {i * 0.1f, -i * 1.f};
    }
    aos.pushField<int>();
    REQUIRE(aos.dataSize() ==
            5 * (sizeof(int) + sizeof(hermes::geo::vec2) + sizeof(int)));
    for (u32 i = 0; i < aos.size(); ++i) {
      REQUIRE(aos.get<int>(0, i) == (i32)i);
      REQUIRE(aos.get<hermes::geo::vec2>(1, i) ==
              hermes::geo::vec2(i * 0.1f, -i * 1.f));
    }
  } //
  SECTION("Access") {
    AoS aos;
    aos.pushField<geo::vec3>("geo::vec3");
    aos.pushField<f32>("f32");
    aos.pushField<int>("int");
    REQUIRE(aos.resize(4) == HeError::NO_ERROR);
    auto vec3_field = aos.field<geo::vec3>("geo::vec3");
    auto f32_field = aos.field<f32>("f32");
    auto int_field = aos.field<int>("int");
    for (u32 i = 0; i < 4; ++i) {
      vec3_field[i] = {1.f + i, 2.f + i, 3.f + i};
      f32_field[i] = 1.f * i;
      int_field[i] = i + 1;
    }
    for (i32 i = 0; i < 4; ++i) {
      REQUIRE(aos.get<geo::vec3>(0, i) == geo::vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE_THAT(aos.get<f32>(1, i),
                   Catch::Matchers::WithinAbs(1.f * i, 1e-8));
      REQUIRE(aos.get<int>(2, i) == i + 1);
      REQUIRE(vec3_field[i] == geo::vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE(f32_field[i] == 1.f * i);
      REQUIRE(int_field[i] == i + 1);
    }
    REQUIRE(aos.back<geo::vec3>(0) == geo::vec3(1.f + 3, 2.f + 3, 3.f + 3));
    REQUIRE_THAT(aos.back<f32>(1), Catch::Matchers::WithinAbs(1.f * 3, 1e-8));
    REQUIRE(aos.back<int>(2) == 3 + 1);
  } //
  SECTION("Accessors") {
    AoS aos;
    aos.pushField<geo::vec3>("geo::vec3");
    aos.pushField<f32>("f32");
    aos.pushField<int>("int");
    REQUIRE(aos.resize(4) == HeError::NO_ERROR);
    auto acc = aos.view();
    for (i32 i = 0; i < 4; ++i) {
      acc.get<geo::vec3>(0, i) = {1.f + i, 2.f + i, 3.f + i};
      acc.get<f32>(1, i) = 1.f * i;
      acc.get<int>(2, i) = i + 1;
    }
    for (i32 i = 0; i < 4; ++i) {
      REQUIRE(acc.get<geo::vec3>(0, i) == geo::vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE_THAT(acc.get<f32>(1, i),
                   Catch::Matchers::WithinAbs(1.f * i, 1e-8));
      REQUIRE(acc.get<int>(2, i) == i + 1);
    }
    const auto &caos = aos;
    auto cacc = caos.view();
    for (i32 i = 0; i < 4; ++i) {
      REQUIRE(cacc.get<geo::vec3>(0, i) ==
              geo::vec3(1.f + i, 2.f + i, 3.f + i));
      REQUIRE_THAT(cacc.get<f32>(1, i),
                   Catch::Matchers::WithinAbs(1.f * i, 1e-8));
      REQUIRE(cacc.get<int>(2, i) == i + 1);
    }
    AoS aos2;
    aos2.pushField<geo::vec3>("geo::vec3");
    aos2.pushField<f32>("f32");
    aos2.pushField<int>("int");
    REQUIRE(aos2.resize(4) == HeError::NO_ERROR);
    for (u32 i = 0; i < 4; ++i) {
      aos2.get<geo::vec3>(0, i) = {-1.f + i, -2.f + i, -3.f + i};
      aos2.get<f32>(1, i) = -1.f * i;
      aos2.get<int>(2, i) = i - 1;
    }
  } //
  SECTION("Field Accessors") {
    AoS aos;
    aos.pushField<size2>("sizes");
    aos.pushField<i32>("i32");
    REQUIRE(aos.resize(5) == HeError::NO_ERROR);
    auto sizes_field = aos.field<size2>(0) = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
    };
    auto i32_field = aos.field<i32>(1) = {-1, -2, -3, -4, -5};
    const auto &caos = aos;
    auto i32_cfield = caos.field<i32>(1);
    auto i32_cast_cfield = static_cast<AoS::ConstFieldView<i32>>(i32_field);
    for (i32 i = 0; i < 5; ++i) {
      REQUIRE(sizes_field[i] == size2(i, i + 1));
      REQUIRE(i32_field[i] == -(i + 1));
      REQUIRE(i32_cfield[i] == -(i + 1));
      REQUIRE(i32_cast_cfield[i] == -(i + 1));
    }
    auto f = [](const AoS::ConstFieldView<i32> &cv) {
      for (i32 i = 0; i < 5; ++i) {
        REQUIRE(cv[i] == -(i + 1));
      }
    };
    f(i32_field);
  } //
  SECTION("Custom AoS") {
    CustomAoS aos;
    aos.pushField<i32>("i32");
    REQUIRE(aos.resize(5) == HeError::NO_ERROR);
    auto acc = aos.field<i32>(0);
    for (int i = 0; i < 5; ++i)
      acc[i] = i;
    auto f = [](CustomAoSFieldView<i32> aos) { HERMES_UNUSED_VARIABLE(aos); };
    auto cf = [](CustomAoSConstFieldView<i32> aos) {
      HERMES_UNUSED_VARIABLE(aos);
    };
    f(acc);
    cf(acc);
  }
  return;
  SECTION("File") {
    AoS aos;
    aos.pushField<geo::vec3>("geo::vec3");
    aos.pushField<f32>("f32");
    aos.pushField<int>("int");
    REQUIRE(aos.resize(4) == HeError::NO_ERROR);
    auto acc = aos.view();
    for (u32 i = 0; i < 4; ++i) {
      acc.get<geo::vec3>(0, i) = {1.f + i, 2.f + i, 3.f + i};
      acc.get<f32>(1, i) = 1.f * i;
      acc.get<int>(2, i) = i + 1;
    }
    std::ofstream file_out("aos_data", std::ios::binary);
    // file_out << aos;
    file_out.close();
    AoS aos2;
    std::ifstream file_in("aos_data", std::ios::binary | std::ios::in);
    // file_in >> aos2;
    file_in.close();
    REQUIRE(aos.size() == aos2.size());
    REQUIRE(aos.dataSize() == aos2.dataSize());
    REQUIRE(aos.stride() == aos2.stride());
    auto acc2 = aos2.view();
    for (u32 i = 0; i < 4; ++i) {
      REQUIRE_THAT(
          acc2.get<geo::vec3>(0, i).x,
          Catch::Matchers::WithinAbs(acc.get<geo::vec3>(0, i).x, 1e-8));
      REQUIRE_THAT(
          acc2.get<geo::vec3>(0, i).y,
          Catch::Matchers::WithinAbs(acc.get<geo::vec3>(0, i).y, 1e-8));
      REQUIRE_THAT(
          acc2.get<geo::vec3>(0, i).z,
          Catch::Matchers::WithinAbs(acc.get<geo::vec3>(0, i).z, 1e-8));
      REQUIRE_THAT(acc2.get<f32>(1, i),
                   Catch::Matchers::WithinAbs(acc.get<f32>(1, i), 1e-8));
      REQUIRE(acc2.get<int>(2, i) == acc.get<int>(2, i));
    }
    auto fields = aos.layout().fields();
    for (auto f : fields) {
      REQUIRE(aos.layout().contains(f.name));
      REQUIRE(aos2.layout().contains(f.name));
      REQUIRE(aos.layout().fieldId(f.name) == aos2.layout().fieldId(f.name));
    }
  } //
#ifdef HERMES_DEVICE_ENABLED
//  SECTION("Device") {
//    AoS aos;
//    aos.pushField<size2>();
//    aos.pushField<i32>();
//    aos.resize(5);
//    auto sizes_field = aos.field<size2>(0) = {
//        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
//    };
//    auto i32_field = aos.field<i32>(1) = {-1, -2, -3, -4, -5};
//    DeviceAoS d_aos = aos;
//    REQUIRE_THAT(d_aos.size(), aos.size());
//    REQUIRE_THAT(d_aos.layout().fields().size(),
//    aos.layout().fields().size()); REQUIRE_THAT(d_aos.layout().sizeInBytes(),
//    aos.layout().sizeInBytes());
//
//    UnifiedArray<int> results(1);
//    HERMES_CUDA_LAUNCH_AND_SYNC((1), aos_view_k, d_aos.view(), results.data())
//    REQUIRE_THAT(results[0], 0);
//  } //
#endif
}
