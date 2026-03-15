#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/core/debug.h>
#include <hermes/core/ref.h>
#include <hermes/geometry/transform.h>

#ifdef HERMES_INCLUDE_DEBUG_TRAITS

class DebugTraitsTestStructA {
  int a = 2;
  int b = 3;
  std::vector<i32> list = {5, 4, 3, 2, 1};
  std::unordered_map<i32, i32> map = {{1, 2}, {2, 3}, {3, 4}};
  hermes::geo::Transform t;

  friend class hermes::DebugTraits<DebugTraitsTestStructA>;
};

class DebugTraitsTestStructB {
  std::vector<i32> list = {1, 2, 3, 4};
  std::unordered_map<i32, i32> map = {{1, 1}, {2, 2}};
  DebugTraitsTestStructA s;
  friend class hermes::DebugTraits<DebugTraitsTestStructB>;
};

class DebugTraitsTestStructC {
  std::vector<DebugTraitsTestStructA> as = {DebugTraitsTestStructA(),
                                            DebugTraitsTestStructA()};
  std::unordered_map<i32, DebugTraitsTestStructB> bs = {
      {1, DebugTraitsTestStructB()}, {2, DebugTraitsTestStructB()}};
  friend class hermes::DebugTraits<DebugTraitsTestStructC>;
};

template <> struct hermes::DebugTraits<DebugTraitsTestStructA> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static hermes::DebugMessage message(const DebugTraitsTestStructA &data) {
    return hermes::DebugMessage()
        .addTitle("DebugTraitsTestStructA")
        .add("a", data.a)
        .add("b", data.b)
        .addArray("list", data.list)
        .addMap("map", data.map)
        .add("t", data.t);
  }
};

template <> struct hermes::DebugTraits<DebugTraitsTestStructB> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static hermes::DebugMessage message(const DebugTraitsTestStructB &data) {
    return hermes::DebugMessage()
        .addTitle("DebugTraitsTestStructB")
        .addArray("list", data.list)
        .addMap("map", data.map)
        .add("s", data.s);
  }
};

template <> struct hermes::DebugTraits<DebugTraitsTestStructC> {
  static HERMES_CONST_OR_CONSTEXPR bool is_string_serializable = true;
  static hermes::DebugMessage message(const DebugTraitsTestStructC &data) {
    return hermes::DebugMessage()
        .addTitle("DebugTraitsTestStructC")
        .addArray("as", data.as)
        .addMap("bs", data.bs);
  }
};

TEST_CASE("Debug Traits") {
  DebugTraitsTestStructB t;
  std::cout << "DUDE!!!! " << hermes::to_string(t) << std::endl;
  std::cout << t << std::endl;
  DebugTraitsTestStructC c;
  std::cout << hermes::to_string(c) << std::endl;
}

#endif

class ResultTest {
public:
  static u32 constructor_count;
  static u32 destructor_count;
  static u32 copy_constructor_count;
  static u32 assign_constructor_count;
  static u32 copy_operator_count;
  static u32 assign_operator_count;
  static void reset() {
    constructor_count = 0;
    destructor_count = 0;
    copy_constructor_count = 0;
    assign_constructor_count = 0;
    copy_operator_count = 0;
    assign_operator_count = 0;
  }
  ResultTest() { constructor_count++; }
  ~ResultTest() { destructor_count++; }
  ResultTest(ResultTest &&rhs) {
    HERMES_UNUSED_VARIABLE(rhs);
    assign_constructor_count++;
  }
  ResultTest(const ResultTest &rhs) {
    HERMES_UNUSED_VARIABLE(rhs);
    copy_constructor_count++;
  }
  ResultTest &operator=(ResultTest &&rhs) {
    HERMES_UNUSED_VARIABLE(rhs);
    assign_operator_count++;
    return *this;
  }
  ResultTest &operator=(const ResultTest &rhs) {
    HERMES_UNUSED_VARIABLE(rhs);
    copy_operator_count++;
    return *this;
  }
};

u32 ResultTest::constructor_count = 0;
u32 ResultTest::destructor_count = 0;
u32 ResultTest::copy_constructor_count = 0;
u32 ResultTest::assign_constructor_count = 0;
u32 ResultTest::copy_operator_count = 0;
u32 ResultTest::assign_operator_count = 0;

TEST_CASE("result") {
  SECTION("copy assignment") {
    {
      ResultTest::reset();
      ResultTest rt;
      hermes::Result<ResultTest> r = rt;
    }
    REQUIRE(ResultTest::constructor_count == 1);
    REQUIRE(ResultTest::destructor_count == 2);
    REQUIRE(ResultTest::copy_constructor_count == 1);
    REQUIRE(ResultTest::assign_constructor_count == 0);
    REQUIRE(ResultTest::copy_operator_count == 0);
    REQUIRE(ResultTest::assign_operator_count == 0);
  }
  SECTION("move assignment") {
    {
      ResultTest::reset();
      ResultTest rt;
      hermes::Result<ResultTest> r = std::move(rt);
    }
    REQUIRE(ResultTest::constructor_count == 1);
    REQUIRE(ResultTest::destructor_count == 2);
    REQUIRE(ResultTest::copy_constructor_count == 0);
    REQUIRE(ResultTest::assign_constructor_count == 1);
    REQUIRE(ResultTest::copy_operator_count == 0);
    REQUIRE(ResultTest::assign_operator_count == 0);
  }
  SECTION("move assignment") {
    {
      ResultTest::reset();
      hermes::Result<ResultTest> r = ResultTest();
      ResultTest rt = std::move(r).value();
    }
    REQUIRE(ResultTest::constructor_count == 1);
    REQUIRE(ResultTest::destructor_count == 3);
    REQUIRE(ResultTest::copy_constructor_count == 0);
    REQUIRE(ResultTest::assign_constructor_count == 2);
    REQUIRE(ResultTest::copy_operator_count == 0);
    REQUIRE(ResultTest::assign_operator_count == 0);
  }
  SECTION("move assignment") {
    {
      ResultTest::reset();
      auto f = []() {
        ResultTest t;
        return hermes::Result<ResultTest>(std::move(t));
      };
      ResultTest rt;
      rt = std::move(f().value());
    }
    REQUIRE(ResultTest::constructor_count == 2);
    REQUIRE(ResultTest::destructor_count == 4);
    REQUIRE(ResultTest::copy_constructor_count == 0);
    REQUIRE(ResultTest::assign_constructor_count == 2);
    REQUIRE(ResultTest::copy_operator_count == 0);
    REQUIRE(ResultTest::assign_operator_count == 1);
  }
}

TEST_CASE("ref") {
  struct E {
    int a;
    int b;
  };

  E e;
  e.a = 2;
  e.b = -1;

  SECTION("sanity") {
    hermes::Ref<int> r;
    REQUIRE((bool)r == false);
    REQUIRE(r.isPtr() == false);
    REQUIRE(r.isWeak() == false);
    REQUIRE(r.isShared() == false);
    REQUIRE(r.get() == nullptr);
  } //
  SECTION("polymorphism") {
    struct F : public E {
      int c;
    };

    F f;
    f.a = 1;
    f.b = 2;
    f.c = 3;

    SECTION("not owner") {
      {
        // copy
        hermes::Ref<E> re;
        auto rf = hermes::Ref<F>::ptr(&f);
        re = rf;
        REQUIRE(re->a == f.a);
        REQUIRE(re->b == f.b);
        REQUIRE(reinterpret_cast<F *>(re.get())->c == f.c);
        REQUIRE(rf->a == f.a);
        REQUIRE(rf->b == f.b);
        REQUIRE(rf->c == f.c);
      }
      {
        // move
        hermes::Ref<E> re;
        auto rf = hermes::Ref<F>::ptr(&f);
        re = std::move(rf);
        REQUIRE((bool)rf == false);
        REQUIRE(re->a == f.a);
        REQUIRE(re->b == f.b);
        REQUIRE(reinterpret_cast<F *>(re.get())->c == f.c);
      }
      {
        // copy constructor
        auto rf = hermes::Ref<F>::ptr(&f);
        hermes::Ref<E> re(rf);
        REQUIRE(re->a == f.a);
        REQUIRE(re->b == f.b);
        REQUIRE(reinterpret_cast<F *>(re.get())->c == f.c);
        REQUIRE(rf->a == f.a);
        REQUIRE(rf->b == f.b);
        REQUIRE(rf->c == f.c);
      }
      {
        // move constructor
        auto rf = hermes::Ref<F>::ptr(&f);
        hermes::Ref<E> re(std::move(rf));
        REQUIRE((bool)rf == false);
        REQUIRE(re->a == f.a);
        REQUIRE(re->b == f.b);
        REQUIRE(reinterpret_cast<F *>(re.get())->c == f.c);
      }

    } //
    SECTION("shared") {
      hermes::Ref<E> re;
      auto rf = hermes::Ref<F>::ptr(&f);
      re = rf;
      REQUIRE(re->a == f.a);
      REQUIRE(re->b == f.b);
    } //
  } //
  SECTION("not owner") {
    auto r = hermes::Ref<E>::ptr(&e);
    REQUIRE(r.isPtr() == true);
    REQUIRE((*r).a == e.a);
    REQUIRE((*r).b == e.b);
    REQUIRE(r->a == e.a);
    REQUIRE(r->b == e.b);
    {
      // const
      const auto &cr = r;
      REQUIRE((*cr).a == e.a);
      REQUIRE((*cr).b == e.b);
      REQUIRE(cr->a == e.a);
      REQUIRE(cr->b == e.b);
    }
    {
      // copy
      auto rr = r;
      REQUIRE(rr->a == e.a);
      REQUIRE(rr->b == e.b);
    }
  } //
  SECTION("shared") {
    auto r = hermes::Ref<E>::shared(e);
    REQUIRE(r.isShared() == true);
    REQUIRE((*r).a == e.a);
    REQUIRE((*r).b == e.b);
    REQUIRE(r->a == e.a);
    REQUIRE(r->b == e.b);
    {
      // const
      const auto &cr = r;
      REQUIRE((*cr).a == e.a);
      REQUIRE((*cr).b == e.b);
      REQUIRE(cr->a == e.a);
      REQUIRE(cr->b == e.b);
    }
    {
      // copy
      auto rr = r;
      REQUIRE(rr->a == e.a);
      REQUIRE(rr->b == e.b);
    }
  } //
  SECTION("weak") {
    auto r = hermes::Ref<E>::shared(e);
    auto w = hermes::Ref<E>::weak(r);
    REQUIRE(w.isWeak() == true);
    REQUIRE((*w).a == e.a);
    REQUIRE((*w).b == e.b);
    REQUIRE(w->a == e.a);
    REQUIRE(w->b == e.b);
    {
      // const
      const auto &cr = r;
      REQUIRE((*cr).a == e.a);
      REQUIRE((*cr).b == e.b);
      REQUIRE(cr->a == e.a);
      REQUIRE(cr->b == e.b);
    }
    {
      // copy
      auto rr = r;
      REQUIRE(rr->a == e.a);
      REQUIRE(rr->b == e.b);
    }
    {
      // convert
      auto rs = (std::shared_ptr<E>)r;
      auto ws = w.getShared();
      REQUIRE(rs);
      REQUIRE(ws);
      auto ww = w.getWeak();
      REQUIRE(!ww.expired());
    } //
  } //
}
