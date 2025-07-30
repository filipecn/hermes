
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/core/debug.h>

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

HERMES_TO_STRING_DEBUG_METHOD_BEGIN(ToStringTest::SubToStringTest)
HERMES_PUSH_DEBUG_FIELD(b);
HERMES_TO_STRING_DEBUG_METHOD_END

HERMES_TO_STRING_DEBUG_METHOD_BEGIN(ToStringTest)
HERMES_PUSH_DEBUG_FIELD(a);
HERMES_PUSH_DEBUG_LINE("this is a line")
HERMES_PUSH_DEBUG_LINE("this is a value {}", object.a)
HERMES_PUSH_DEBUG_HERMES_PTR_FIELD(t);
HERMES_PUSH_DEBUG_HERMES_PTR_FIELD(t2);
HERMES_PUSH_DEBUG_RAW_PTR_FIELD(p);
HERMES_PUSH_DEBUG_RAW_PTR_FIELD(p2);
HERMES_PUSH_DEBUG_CUSTOM_FIELD(a, "custom {}", object.a);
HERMES_PUSH_DEBUG_SEPARATOR_LINE
HERMES_PUSH_DEBUG_HERMES_FIELD(b);
HERMES_PUSH_DEBUG_ARRAY_FIELD_BEGIN(v, vv)
HERMES_PUSH_DEBUG_FIELD_VALUE(vv, vv);
HERMES_PUSH_DEBUG_ARRAY_FIELD_END
HERMES_TO_STRING_DEBUG_METHOD_END

} // namespace hermes

TEST_CASE("to_string") {
  ToStringTest test;
  HERMES_INFO("{}", hermes::to_string(test));
}

#endif
