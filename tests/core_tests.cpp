#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/core/debug.h>
#include <hermes/core/ref.h>

#ifdef HERMES_INCLUDE_TO_STRING

class ToStringTest {
public:
  class SubToStringTest {
    int b{2};
    HERMES_TO_STRING_FRIEND(ToStringTest::SubToStringTest)
  };
  ToStringTest() {
    t2.reset(new SubToStringTest());
    p2 = new int(4);
    v = {1, 2, 3};
  }
  ~ToStringTest() { delete p2; }

private:
  int a{3};
  SubToStringTest b;
  std::shared_ptr<SubToStringTest> t;
  std::shared_ptr<SubToStringTest> t2;
  int *p{nullptr};
  int *p2{nullptr};
  std::vector<int> v;
  HERMES_TO_STRING_FRIEND(ToStringTest)
};

namespace hermes {

HERMES_DECLARE_TO_STRING_DEBUG_METHOD(ToStringTest)

HERMES_TO_STRING_METHOD_BEGIN(ToStringTest::SubToStringTest)
HERMES_TO_STRING_METHOD_FIELD(b);
HERMES_TO_STRING_METHOD_END

HERMES_TO_STRING_METHOD_BEGIN(ToStringTest)
HERMES_TO_STRING_METHOD_TITLE
HERMES_TO_STRING_METHOD_FIELD(a);
HERMES_TO_STRING_METHOD_LINE("this is a line")
HERMES_TO_STRING_METHOD_LINE("this is a value {}", object.a)
HERMES_TO_STRING_METHOD_HERMES_PTR_FIELD(t);
HERMES_TO_STRING_METHOD_HERMES_PTR_FIELD(t2);
HERMES_TO_STRING_METHOD_RAW_PTR_FIELD(p);
HERMES_TO_STRING_METHOD_RAW_PTR_FIELD(p2);
HERMES_TO_STRING_METHOD_CUSTOM_FIELD(a, "custom {}", object.a);
HERMES_TO_STRING_METHOD_SEPARATOR_LINE
HERMES_TO_STRING_METHOD_HERMES_FIELD(b);
HERMES_TO_STRING_METHOD_ARRAY_FIELD_BEGIN(v, vv)
HERMES_TO_STRING_METHOD_FIELD_VALUE(vv, vv);
HERMES_TO_STRING_METHOD_ARRAY_FIELD_END
HERMES_TO_STRING_METHOD_END

} // namespace hermes

TEST_CASE("to_string") {
  ToStringTest test;
  HERMES_INFO("{}", hermes::to_string(test));
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
