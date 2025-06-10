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
///\file common_tests.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2021-06-26
///
///\brief

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <hermes/common/size.h>
#include <hermes/common/str.h>

using namespace hermes;

TEST_CASE("size", "[common]") {
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

TEST_CASE("Str", "[common]") {
  SECTION("abbreviation") {
    REQUIRE(Str::abbreviate("123456789", 5, "..s") == "...89");
    REQUIRE(Str::abbreviate("12345678", 5, "..s") == "...78");
    REQUIRE(Str::abbreviate("123456789", 4, "..s") == "...9");
    REQUIRE(Str::abbreviate("12345678", 4, "..s") == "...8");

    REQUIRE(Str::abbreviate("123456789", 5, "s..") == "12...");
    REQUIRE(Str::abbreviate("12345678", 5, "s..") == "12...");
    REQUIRE(Str::abbreviate("123456789", 4, "s..") == "1...");
    REQUIRE(Str::abbreviate("12345678", 4, "s..") == "1...");

    REQUIRE(Str::abbreviate("123456789", 5, ".s.") == ".456.");
    REQUIRE(Str::abbreviate("12345678", 5, ".s.") == ".456.");
    REQUIRE(Str::abbreviate("123456789", 4, ".s.") == ".45.");
    REQUIRE(Str::abbreviate("12345678", 4, ".s.") == ".45.");

    REQUIRE(Str::abbreviate("123456789", 5, "s.s") == "12.89");
    REQUIRE(Str::abbreviate("12345678", 5, "s.s") == "12.78");
    REQUIRE(Str::abbreviate("123456789", 4, "s.s") == "1..9");
    REQUIRE(Str::abbreviate("12345678", 4, "s.s") == "1..8");

    // big cases
    REQUIRE(Str::abbreviate("123456789", 6, "..s") == "...789");
    REQUIRE(Str::abbreviate("123456789", 6, "s.s") == "12..89");
    REQUIRE(Str::abbreviate("123456789", 6, "s..") == "123...");
    REQUIRE(Str::abbreviate("123456789", 6, ".s.") == ".3456.");

    REQUIRE(Str::abbreviate("123456789", 7, "..s") == "...6789");
    REQUIRE(Str::abbreviate("123456789", 7, "s.s") == "123.789");
    REQUIRE(Str::abbreviate("123456789", 7, "s..") == "1234...");
    REQUIRE(Str::abbreviate("123456789", 7, ".s.") == "..456..");

    // small cases
    REQUIRE(Str::abbreviate("123456789", 3, "..s") == "..9");
    REQUIRE(Str::abbreviate("123456789", 3, "s.s") == "1.9");
    REQUIRE(Str::abbreviate("123456789", 3, "s..") == "1..");
    REQUIRE(Str::abbreviate("123456789", 3, ".s.") == ".5.");

    REQUIRE(Str::abbreviate("123456789", 2, "..s") == ".9");
    REQUIRE(Str::abbreviate("123456789", 2, "s.s") == "19");
    REQUIRE(Str::abbreviate("123456789", 2, "s..") == "1.");
    REQUIRE(Str::abbreviate("123456789", 2, ".s.") == "45");

    REQUIRE(Str::abbreviate("123456789", 1, "..s") == "9");
    REQUIRE(Str::abbreviate("123456789", 1, "s.s") == "1");
    REQUIRE(Str::abbreviate("123456789", 1, "s..") == "1");
    REQUIRE(Str::abbreviate("123456789", 1, ".s.") == "5");

    REQUIRE(Str::abbreviate("123456789", 0, "..s").empty());
    REQUIRE(Str::abbreviate("123456789", 0, "s.s").empty());
    REQUIRE(Str::abbreviate("123456789", 0, ".s.").empty());
    REQUIRE(Str::abbreviate("123456789", 0, "s..").empty());
  } //
  SECTION("justify") {
    REQUIRE("  asd" == Str::rjust("asd", 5));
    REQUIRE("abcdef" == Str::rjust("abcdef", 5));
    REQUIRE("asd  " == Str::ljust("asd", 5));
    REQUIRE("abcdef" == Str::ljust("abcdef", 5));
    REQUIRE(" asd " == Str::cjust("asd", 5));
    REQUIRE("abcdef" == Str::cjust("abcdef", 5));
  } //
  SECTION("strip") {
    REQUIRE(Str::strip(" asd ", "") == " asd ");
    REQUIRE(Str::strip(" asd ", " ") == "asd");
    REQUIRE(Str::strip(" asd \n", " ") == "asd \n");
    REQUIRE(Str::strip(" asd \n", " \n") == "asd");
  } //
  SECTION("is integer") {
    REQUIRE(Str::isInteger("") == false);
    REQUIRE(Str::isInteger("+") == false);
    REQUIRE(Str::isInteger("234+") == false);
    REQUIRE(Str::isInteger("12.2") == false);
    REQUIRE(Str::isInteger(" +123 ") == true);
    REQUIRE(Str::isInteger(" -2435 ") == true);
    REQUIRE(Str::isInteger("234234") == true);
  } //
  SECTION("is number") {
    REQUIRE(Str::isNumber("") == false);
    REQUIRE(Str::isNumber("+") == false);
    REQUIRE(Str::isNumber("234+") == false);
    REQUIRE(Str::isNumber("12.2") == true);
    REQUIRE(Str::isNumber(" +123 ") == true);
    REQUIRE(Str::isNumber(" -2435 ") == true);
    REQUIRE(Str::isNumber("234234") == true);
    REQUIRE(Str::isNumber("234234.") == true);
    REQUIRE(Str::isNumber(".234234") == true);
    REQUIRE(Str::isNumber("342.34") == true);
    REQUIRE(Str::isNumber("342.34f") == true);
    REQUIRE(Str::isNumber("34234f") == true);
    REQUIRE(Str::isNumber("-342.34") == true);
    REQUIRE(Str::isNumber("-342.34f") == true);
    REQUIRE(Str::isNumber("-34234f") == true);
    REQUIRE(Str::isNumber("-1e+10") == true);
    REQUIRE(Str::isNumber("1.23e+10") == true);
    REQUIRE(Str::isNumber("-1e-10") == true);
    REQUIRE(Str::isNumber("1.23e-10") == true);
    REQUIRE(Str::isNumber("-1e10") == true);
    REQUIRE(Str::isNumber("1.23e10") == true);
    REQUIRE(Str::isNumber("13e23e10") == false);
  } //
  SECTION("format") {
    REQUIRE(Str::format("word") == "word");
    REQUIRE(Str::format("word", 3) == "word");
    REQUIRE(Str::format("word {}") == "word {}");
    REQUIRE(Str::format("word {}", 3) == "word 3");
    REQUIRE(Str::format("word {} word {}", 3, 4) == "word 3 word 4");
  } //
  SECTION("split") {
    std::string a = "1 22 3 44 5";
    auto s = Str::split(a);
    REQUIRE(s.size() == 5);
    REQUIRE(s[0] == "1");
    REQUIRE(s[1] == "22");
    REQUIRE(s[2] == "3");
    REQUIRE(s[3] == "44");
    REQUIRE(s[4] == "5");
  } //
  SECTION("join") {
    std::vector<std::string> s = {"a", "b", "c"};
    auto ss = Str::join(s, ",");
    REQUIRE(ss == "a,b,c");
    std::vector<int> ints = {1, 2, 3};
    ss = Str::join(ints, " ");
    REQUIRE(ss == "1 2 3");
  } //
  SECTION("split with delimiter") {
    std::string a = "1 2, 3,4, 5";
    auto s = Str::split(a, ",");
    REQUIRE(s.size() == 4);
    REQUIRE(s[0] == "1 2");
    REQUIRE(s[1] == " 3");
    REQUIRE(s[2] == "4");
    REQUIRE(s[3] == " 5");
  } //
  SECTION("concat") {
    std::string a = Str::concat("a", " ", 2, "b");
    REQUIRE(a == "a 2b");
  } //
  SECTION("regex") {
    SECTION("alpha numeric word") {
      REQUIRE(Str::regex::match("abc", Str::regex::alpha_numeric_word));
      REQUIRE(Str::regex::match("abc123", Str::regex::alpha_numeric_word));
      REQUIRE(Str::regex::match("123abc", Str::regex::alpha_numeric_word));
      REQUIRE(Str::regex::match("123", Str::regex::alpha_numeric_word));
    } //
    SECTION("c identifier") {
      REQUIRE(Str::regex::match("abc", Str::regex::c_identifier));
      REQUIRE(Str::regex::match("abc123", Str::regex::c_identifier));
      REQUIRE(Str::regex::match("_abc123", Str::regex::c_identifier));
      REQUIRE(Str::regex::match("_abc123__", Str::regex::c_identifier));
      REQUIRE(Str::regex::match("_", Str::regex::c_identifier));
      REQUIRE_FALSE(Str::regex::match("123abc", Str::regex::c_identifier));
      REQUIRE_FALSE(Str::regex::match("123", Str::regex::c_identifier));
    } //
    SECTION("floating point number") {
      REQUIRE(Str::regex::match("123", Str::regex::floating_point_number));
      REQUIRE(Str::regex::match("+123", Str::regex::floating_point_number));
      REQUIRE(Str::regex::match("-123", Str::regex::floating_point_number));
      REQUIRE(Str::regex::match("123.234", Str::regex::floating_point_number));
      REQUIRE(
          Str::regex::match("123.34e-23", Str::regex::floating_point_number));
      REQUIRE(
          Str::regex::match("123.34e+23", Str::regex::floating_point_number));
      REQUIRE(Str::regex::match(".34", Str::regex::floating_point_number));
      REQUIRE_FALSE(Str::regex::match("1e", Str::regex::floating_point_number));
    } //
    SECTION("integer number") {
      REQUIRE(Str::regex::match("123", Str::regex::integer_number));
      REQUIRE(Str::regex::match("+123", Str::regex::integer_number));
      REQUIRE(Str::regex::match("-123", Str::regex::integer_number));
      REQUIRE_FALSE(Str::regex::match("123.234", Str::regex::integer_number));
      REQUIRE_FALSE(
          Str::regex::match("123.34e+23", Str::regex::integer_number));
      REQUIRE_FALSE(Str::regex::match(".34", Str::regex::integer_number));
      REQUIRE_FALSE(Str::regex::match("-", Str::regex::integer_number));
      REQUIRE_FALSE(Str::regex::match("+", Str::regex::integer_number));
    } //
    SECTION("regex match") {
      REQUIRE(Str::regex::match("subsequence123", "\\b(sub)([^ ]*)"));
      REQUIRE(!Str::regex::match("susequence123", "\\b(sub)([^ ]*)"));
      REQUIRE(Str::regex::match("sub-sequence123", "\\b(sub)([^ ]*)"));
    } //
    SECTION("regex contains") {
      REQUIRE(Str::regex::contains("subsequence123", "\\b(sub)"));
      REQUIRE(!Str::regex::contains("subsequence123", "\\b(qen)"));
      REQUIRE(Str::regex::contains("/usr/local/lib.a", ".*\\.a"));
    } //
    SECTION("regex search") {
      std::string s("this subject has a submarine as a subsequence");
      auto result = Str::regex::search(s, "\\b(sub)([^ ]*)");
      REQUIRE(result.size() == 3);
      REQUIRE(result[0] == "subject");
      REQUIRE(result[1] == "sub");
      REQUIRE(result[2] == "ject");
      int index = 0;
      std::string expected[3] = {"subject", "submarine", "subsequence"};
      Str::regex::search(s, "\\b(sub)([^ ]*)", [&](const std::smatch &m) {
        REQUIRE(m[0] == expected[index++]);
      });
    } //
    SECTION("regex replace") {
      std::string s("there is a subsequence in the string");
      REQUIRE(Str::regex::replace(s, "\\b(sub)([^ ]*)", "sub-$2") ==
              "there is a sub-sequence in the string");
      REQUIRE(Str::regex::replace(s, "\\b(sub)([^ ]*)", "$2") ==
              "there is a sequence in the string");
      std::string s2("/home//usr/local");
      REQUIRE(Str::regex::replace(s2, "\\b//", "/") == "/home/usr/local");
    } //
    SECTION("regex string begin") {
      auto result = Str::regex::search("ssssubsequence123", "sub");
      REQUIRE(result.size() == 1);
      result = Str::regex::search("ssssubsequence123", "^sub");
      REQUIRE(result.empty());
    } //
  } //
  SECTION("string class") {
    Str s;
    REQUIRE((s += "abc") == "abc");
    REQUIRE(s << 2 << 3 == "abc23");
    REQUIRE(s + 2 == "abc2");
    s = "3";
    REQUIRE(s == 3);
  } //
  SECTION("prefix") {
    REQUIRE(Str::isPrefix("0123", "0123456"));
    REQUIRE(Str::isPrefix("", "0123456"));
    REQUIRE_FALSE(Str::isPrefix("01234", "01"));
  } //
}
