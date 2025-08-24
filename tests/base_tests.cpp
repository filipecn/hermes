#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/base/flags.h>
#include <hermes/base/index.h>
#include <hermes/base/str.h>

using namespace hermes;

namespace hermes {
enum class test_bits : u32 {
  a = 1 << 0,
  b = 1 << 1,
  c = 1 << 2,
  d = 1 << 3,
};

using test_flags = Flags<test_bits>;

template <> struct FlagTraits<test_bits> {
  static HERMES_CONST_OR_CONSTEXPR bool is_bitmask = true;
  static HERMES_CONST_OR_CONSTEXPR test_flags all_flags =
      test_bits::a | test_bits::b | test_bits::c | test_bits::d;
};
} // namespace hermes

TEST_CASE("flags", "[base]") {
  REQUIRE(test_bits::a != test_bits::b);
  REQUIRE(test_bits::a | test_bits::b);
  test_flags flags = test_bits::a | test_bits::c;
  REQUIRE(contains(flags, test_bits::a));
  REQUIRE(!contains(flags, test_bits::b));
  REQUIRE(contains(flags, test_bits::c));
  REQUIRE(!contains(flags, test_bits::d));
}

TEST_CASE("cstr", "[base]") {
  SECTION("abbreviation") {
    REQUIRE(cstr::abbreviate("123456789", 5, "..s") == "...89");
    REQUIRE(cstr::abbreviate("12345678", 5, "..s") == "...78");
    REQUIRE(cstr::abbreviate("123456789", 4, "..s") == "...9");
    REQUIRE(cstr::abbreviate("12345678", 4, "..s") == "...8");

    REQUIRE(cstr::abbreviate("123456789", 5, "s..") == "12...");
    REQUIRE(cstr::abbreviate("12345678", 5, "s..") == "12...");
    REQUIRE(cstr::abbreviate("123456789", 4, "s..") == "1...");
    REQUIRE(cstr::abbreviate("12345678", 4, "s..") == "1...");

    REQUIRE(cstr::abbreviate("123456789", 5, ".s.") == ".456.");
    REQUIRE(cstr::abbreviate("12345678", 5, ".s.") == ".456.");
    REQUIRE(cstr::abbreviate("123456789", 4, ".s.") == ".45.");
    REQUIRE(cstr::abbreviate("12345678", 4, ".s.") == ".45.");

    REQUIRE(cstr::abbreviate("123456789", 5, "s.s") == "12.89");
    REQUIRE(cstr::abbreviate("12345678", 5, "s.s") == "12.78");
    REQUIRE(cstr::abbreviate("123456789", 4, "s.s") == "1..9");
    REQUIRE(cstr::abbreviate("12345678", 4, "s.s") == "1..8");

    // big cases
    REQUIRE(cstr::abbreviate("123456789", 6, "..s") == "...789");
    REQUIRE(cstr::abbreviate("123456789", 6, "s.s") == "12..89");
    REQUIRE(cstr::abbreviate("123456789", 6, "s..") == "123...");
    REQUIRE(cstr::abbreviate("123456789", 6, ".s.") == ".3456.");

    REQUIRE(cstr::abbreviate("123456789", 7, "..s") == "...6789");
    REQUIRE(cstr::abbreviate("123456789", 7, "s.s") == "123.789");
    REQUIRE(cstr::abbreviate("123456789", 7, "s..") == "1234...");
    REQUIRE(cstr::abbreviate("123456789", 7, ".s.") == "..456..");

    // small cases
    REQUIRE(cstr::abbreviate("123456789", 3, "..s") == "..9");
    REQUIRE(cstr::abbreviate("123456789", 3, "s.s") == "1.9");
    REQUIRE(cstr::abbreviate("123456789", 3, "s..") == "1..");
    REQUIRE(cstr::abbreviate("123456789", 3, ".s.") == ".5.");

    REQUIRE(cstr::abbreviate("123456789", 2, "..s") == ".9");
    REQUIRE(cstr::abbreviate("123456789", 2, "s.s") == "19");
    REQUIRE(cstr::abbreviate("123456789", 2, "s..") == "1.");
    REQUIRE(cstr::abbreviate("123456789", 2, ".s.") == "45");

    REQUIRE(cstr::abbreviate("123456789", 1, "..s") == "9");
    REQUIRE(cstr::abbreviate("123456789", 1, "s.s") == "1");
    REQUIRE(cstr::abbreviate("123456789", 1, "s..") == "1");
    REQUIRE(cstr::abbreviate("123456789", 1, ".s.") == "5");

    REQUIRE(cstr::abbreviate("123456789", 0, "..s").empty());
    REQUIRE(cstr::abbreviate("123456789", 0, "s.s").empty());
    REQUIRE(cstr::abbreviate("123456789", 0, ".s.").empty());
    REQUIRE(cstr::abbreviate("123456789", 0, "s..").empty());
  } //
  SECTION("justify") {
    REQUIRE("  asd" == cstr::rjust("asd", 5));
    REQUIRE("abcdef" == cstr::rjust("abcdef", 5));
    REQUIRE("asd  " == cstr::ljust("asd", 5));
    REQUIRE("abcdef" == cstr::ljust("abcdef", 5));
    REQUIRE(" asd " == cstr::cjust("asd", 5));
    REQUIRE("abcdef" == cstr::cjust("abcdef", 5));
  } //
  SECTION("strip") {
    REQUIRE(cstr::strip(" asd ", "") == " asd ");
    REQUIRE(cstr::strip(" asd ", " ") == "asd");
    REQUIRE(cstr::strip(" asd \n", " ") == "asd \n");
    REQUIRE(cstr::strip(" asd \n", " \n") == "asd");
  } //
  SECTION("is integer") {
    REQUIRE(cstr::isInteger("") == false);
    REQUIRE(cstr::isInteger("+") == false);
    REQUIRE(cstr::isInteger("234+") == false);
    REQUIRE(cstr::isInteger("12.2") == false);
    REQUIRE(cstr::isInteger(" +123 ") == true);
    REQUIRE(cstr::isInteger(" -2435 ") == true);
    REQUIRE(cstr::isInteger("234234") == true);
  } //
  SECTION("is number") {
    REQUIRE(cstr::isNumber("") == false);
    REQUIRE(cstr::isNumber("+") == false);
    REQUIRE(cstr::isNumber("234+") == false);
    REQUIRE(cstr::isNumber("12.2") == true);
    REQUIRE(cstr::isNumber(" +123 ") == true);
    REQUIRE(cstr::isNumber(" -2435 ") == true);
    REQUIRE(cstr::isNumber("234234") == true);
    REQUIRE(cstr::isNumber("234234.") == true);
    REQUIRE(cstr::isNumber(".234234") == true);
    REQUIRE(cstr::isNumber("342.34") == true);
    REQUIRE(cstr::isNumber("342.34f") == true);
    REQUIRE(cstr::isNumber("34234f") == true);
    REQUIRE(cstr::isNumber("-342.34") == true);
    REQUIRE(cstr::isNumber("-342.34f") == true);
    REQUIRE(cstr::isNumber("-34234f") == true);
    REQUIRE(cstr::isNumber("-1e+10") == true);
    REQUIRE(cstr::isNumber("1.23e+10") == true);
    REQUIRE(cstr::isNumber("-1e-10") == true);
    REQUIRE(cstr::isNumber("1.23e-10") == true);
    REQUIRE(cstr::isNumber("-1e10") == true);
    REQUIRE(cstr::isNumber("1.23e10") == true);
    REQUIRE(cstr::isNumber("13e23e10") == false);
  } //
  SECTION("format") {
    REQUIRE(cstr::format("word") == "word");
    REQUIRE(cstr::format("word", 3) == "word");
    REQUIRE(cstr::format("word {}") == "word {}");
    REQUIRE(cstr::format("word {}", 3) == "word 3");
    REQUIRE(cstr::format("word {} word {}", 3, 4) == "word 3 word 4");
  } //
  SECTION("split") {
    std::string a = "1 22 3 44 5";
    auto s = cstr::split(a);
    REQUIRE(s.size() == 5);
    REQUIRE(s[0] == "1");
    REQUIRE(s[1] == "22");
    REQUIRE(s[2] == "3");
    REQUIRE(s[3] == "44");
    REQUIRE(s[4] == "5");
  } //
  SECTION("join") {
    std::vector<std::string> s = {"a", "b", "c"};
    auto ss = cstr::join(s, ",");
    REQUIRE(ss == "a,b,c");
    std::vector<int> ints = {1, 2, 3};
    ss = cstr::join(ints, " ");
    REQUIRE(ss == "1 2 3");
  } //
  SECTION("split with delimiter") {
    std::string a = "1 2, 3,4, 5";
    auto s = cstr::split(a, ",");
    REQUIRE(s.size() == 4);
    REQUIRE(s[0] == "1 2");
    REQUIRE(s[1] == " 3");
    REQUIRE(s[2] == "4");
    REQUIRE(s[3] == " 5");
  } //
  SECTION("concat") {
    std::string a = cstr::concat("a", " ", 2, "b");
    REQUIRE(a == "a 2b");
  } //
  SECTION("regex") {
    SECTION("alpha numeric word") {
      REQUIRE(cstr::regex::match("abc", cstr::regex::alpha_numeric_word));
      REQUIRE(cstr::regex::match("abc123", cstr::regex::alpha_numeric_word));
      REQUIRE(cstr::regex::match("123abc", cstr::regex::alpha_numeric_word));
      REQUIRE(cstr::regex::match("123", cstr::regex::alpha_numeric_word));
    } //
    SECTION("c identifier") {
      REQUIRE(cstr::regex::match("abc", cstr::regex::c_identifier));
      REQUIRE(cstr::regex::match("abc123", cstr::regex::c_identifier));
      REQUIRE(cstr::regex::match("_abc123", cstr::regex::c_identifier));
      REQUIRE(cstr::regex::match("_abc123__", cstr::regex::c_identifier));
      REQUIRE(cstr::regex::match("_", cstr::regex::c_identifier));
      REQUIRE_FALSE(cstr::regex::match("123abc", cstr::regex::c_identifier));
      REQUIRE_FALSE(cstr::regex::match("123", cstr::regex::c_identifier));
    } //
    SECTION("floating point number") {
      REQUIRE(cstr::regex::match("123", cstr::regex::floating_point_number));
      REQUIRE(cstr::regex::match("+123", cstr::regex::floating_point_number));
      REQUIRE(cstr::regex::match("-123", cstr::regex::floating_point_number));
      REQUIRE(
          cstr::regex::match("123.234", cstr::regex::floating_point_number));
      REQUIRE(
          cstr::regex::match("123.34e-23", cstr::regex::floating_point_number));
      REQUIRE(
          cstr::regex::match("123.34e+23", cstr::regex::floating_point_number));
      REQUIRE(cstr::regex::match(".34", cstr::regex::floating_point_number));
      REQUIRE_FALSE(
          cstr::regex::match("1e", cstr::regex::floating_point_number));
    } //
    SECTION("integer number") {
      REQUIRE(cstr::regex::match("123", cstr::regex::integer_number));
      REQUIRE(cstr::regex::match("+123", cstr::regex::integer_number));
      REQUIRE(cstr::regex::match("-123", cstr::regex::integer_number));
      REQUIRE_FALSE(cstr::regex::match("123.234", cstr::regex::integer_number));
      REQUIRE_FALSE(
          cstr::regex::match("123.34e+23", cstr::regex::integer_number));
      REQUIRE_FALSE(cstr::regex::match(".34", cstr::regex::integer_number));
      REQUIRE_FALSE(cstr::regex::match("-", cstr::regex::integer_number));
      REQUIRE_FALSE(cstr::regex::match("+", cstr::regex::integer_number));
    } //
    SECTION("regex match") {
      REQUIRE(cstr::regex::match("subsequence123", "\\b(sub)([^ ]*)"));
      REQUIRE(!cstr::regex::match("susequence123", "\\b(sub)([^ ]*)"));
      REQUIRE(cstr::regex::match("sub-sequence123", "\\b(sub)([^ ]*)"));
    } //
    SECTION("regex contains") {
      REQUIRE(cstr::regex::contains("subsequence123", "\\b(sub)"));
      REQUIRE(!cstr::regex::contains("subsequence123", "\\b(qen)"));
      REQUIRE(cstr::regex::contains("/usr/local/lib.a", ".*\\.a"));
    } //
    SECTION("regex search") {
      std::string s("this subject has a submarine as a subsequence");
      auto result = cstr::regex::search(s, "\\b(sub)([^ ]*)");
      REQUIRE(result.size() == 3);
      REQUIRE(result[0] == "subject");
      REQUIRE(result[1] == "sub");
      REQUIRE(result[2] == "ject");
      int index = 0;
      std::string expected[3] = {"subject", "submarine", "subsequence"};
      cstr::regex::search(s, "\\b(sub)([^ ]*)", [&](const std::smatch &m) {
        REQUIRE(m[0] == expected[index++]);
      });
    } //
    SECTION("regex replace") {
      std::string s("there is a subsequence in the string");
      REQUIRE(cstr::regex::replace(s, "\\b(sub)([^ ]*)", "sub-$2") ==
              "there is a sub-sequence in the string");
      REQUIRE(cstr::regex::replace(s, "\\b(sub)([^ ]*)", "$2") ==
              "there is a sequence in the string");
      std::string s2("/home//usr/local");
      REQUIRE(cstr::regex::replace(s2, "\\b//", "/") == "/home/usr/local");
    } //
    SECTION("regex string begin") {
      auto result = cstr::regex::search("ssssubsequence123", "sub");
      REQUIRE(result.size() == 1);
      result = cstr::regex::search("ssssubsequence123", "^sub");
      REQUIRE(result.empty());
    } //
  } //
  SECTION("string class") {
    cstr s;
    REQUIRE((s += "abc") == "abc");
    REQUIRE(s << 2 << 3 == "abc23");
    REQUIRE(s + 2 == "abc2");
    s = "3";
    REQUIRE(s == 3);
  } //
  SECTION("prefix") {
    REQUIRE(cstr::isPrefix("0123", "0123456"));
    REQUIRE(cstr::isPrefix("", "0123456"));
    REQUIRE_FALSE(cstr::isPrefix("01234", "01"));
  } //
}

TEST_CASE("size", "[base]") {
  {
    size2 s(10, 2);
    REQUIRE(s.width == 10);
    REQUIRE(s.height == 2);
    REQUIRE(s[0] == s.width);
    REQUIRE(s[1] == s.height);
    REQUIRE(s.total() == 20);
    REQUIRE(s.contains(5, 0));
    REQUIRE(!s.contains(10, 0));
  }
}

TEST_CASE("index", "[base]") {
  SECTION("Index2 arithmetic") {
    index2 ij(1, -3);
    REQUIRE(ij + index2(-7, 10) == index2(-6, 7));
    REQUIRE(ij - index2(-7, 10) == index2(8, -13));
    REQUIRE(ij + size2(7, 10) == index2(8, 7));
    REQUIRE(ij - size2(7, 10) == index2(-6, -13));
    REQUIRE(size2(7, 10) + ij == index2(8, 7));
    REQUIRE(size2(7, 10) - ij == index2(6, 13));
  } //
  SECTION("Index2") {
    index2 a;
    index2 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  } //
  SECTION("index2 friends") {
    index2 a(-1, 2);
    index2 b(4, 1);
    REQUIRE(max(a, b) == index2(4, 2));
    REQUIRE(min(a, b) == index2(-1, 1));
  } //
  SECTION("Index2Range") {
    range2 r({1, 1}, {3, 3});
    REQUIRE(r.contains({1, 1}));
    REQUIRE(r.contains({1, 2}));
    REQUIRE(!r.contains({0, 0}));
    REQUIRE(!r.contains({3, 3}));
    REQUIRE(r.area() == 4);

    int cur = 0;
    range2 range(10, 10);
    for (auto index : range) {
      REQUIRE(cur % 10 == index.i);
      REQUIRE(cur / 10 == index.j);
      REQUIRE(range.contains(index));
      cur++;
    }
    REQUIRE(cur == 10 * 10);
    SECTION("intersection") {
      range2 a({0, 0}, {10, 10});
      range2 b({5, -5}, {7, 70});
      REQUIRE(intersect(a, b) == range2({5, 0}, {7, 10}));
    } //
  } //
  SECTION("Index3 arithmetic") {
    index3 ij(1, -3, 0);
    REQUIRE(ij + index3(-7, 10, 1) == index3(-6, 7, 1));
    REQUIRE(ij - index3(-7, 10, 1) == index3(8, -13, -1));
    REQUIRE(ij + size3(7, 10, 3) == index3(8, 7, 3));
    REQUIRE(ij - size3(7, 10, 3) == index3(-6, -13, -3));
    REQUIRE(size3(7, 10, 5) + ij == index3(8, 7, 5));
    REQUIRE(size3(7, 10, 5) - ij == index3(6, 13, 5));
  } //
  SECTION("Index3") {
    index3 a;
    index3 b;
    REQUIRE(a == b);
    b.j = 1;
    REQUIRE(a != b);
  } //
  SECTION("Index3Range") {
    int cur = 0;
    for (auto index : Index3<i32>::Range(10, 10, 10)) {
      REQUIRE((cur % 100) % 10 == index.i);
      REQUIRE((cur % 100) / 10 == index.j);
      REQUIRE(cur / 100 == index.k);
      cur++;
    }
    REQUIRE(cur == 10 * 10 * 10);
    Index3<i32>::Range range(index3(-5, -5, -5), index3(5, 5, 5));
    REQUIRE(range.size().total() == 10 * 10 * 10);
    cur = 0;
    for (auto index : range) {
      REQUIRE((cur % 100) % 10 - 5 == index.i);
      REQUIRE((cur % 100) / 10 - 5 == index.j);
      REQUIRE(cur / 100 - 5 == index.k);
      cur++;
    }
    REQUIRE(cur == 10 * 10 * 10);
  } //
}
