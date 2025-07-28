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
///\file parser.cpp
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-11-11
///
///\brief

#include <hermes/common/debug.h>
#include <hermes/common/parsers.h>

namespace hermes {

void ParseTree::iterate(const std::function<bool(const Node &)> &callback) const {
  std::function<bool(const Node &)> f;
  f = [&](const Node &node) -> bool {
    if (!callback(node))
      return false;
    return std::none_of(node.children.begin(), node.children.end(), [&](const Node &child) {
      if (!f(child))
        return false;
      return true;
    });
  };
  f(root_);
}

std::ostream &operator<<(std::ostream &os, const ParseTree &parse_tree) {
  os << "ParseTree\n";
  std::function<void(const ParseTree::Node &, size_t)> printNode;
  printNode = [&](const ParseTree::Node &node, size_t level) {
    auto level_tab = Str::ljust(" ", level * 4);
    os << level_tab;
    os << "[level: " << level << "]";
    os << "[node: " << ParseTree::typeName(node.type) << "]";
    os << "[start: " << node.start << "]";
    os << "[size: " << node.size << "]";
    if (!node.children.empty())
      os << "[" << node.children.size() << " children]";
    if (!node.name.empty())
      os << "[name: " << node.name << "]";
    os << "\n";
    if (!node.value.empty())
      os << level_tab << level_tab << " ---> value: [" << Str::abbreviate(node.value, 100) << "]\n";
    for (const auto &child:node.children)
      printNode(child, level + 1);
    if (!node.children.empty())
      os << "\n\n";
  };
  printNode(parse_tree.root_, 1);
  return os;
}

StringParser StringParser::cLanguage() {
  StringParser parser;
  parser.setBlankCharacters(" \n\t,;");
  parser.pushBlockDelimiters("\\/\\*", "\\*\\/");
  parser.pushBlockDelimiters("\\{", "\\}");
  parser.pushBlockDelimiters("\\(", "\\)");
  parser.pushBlockDelimiters("\\[", "\\]");
  parser.pushTokenPattern("identifier", Str::regex::c_identifier);
  parser.pushTokenPattern("floating-point-number", Str::regex::floating_point_number);
  parser.pushTokenPattern("integer-number", Str::regex::integer_number);
  parser.pushTokenPattern("math", "[*-+/]+");
  parser.pushTokenPattern("assignment", "=");
  return parser;
}

StringParser::StringParser() = default;

StringParser::~StringParser() = default;

void StringParser::setBlankCharacters(const std::string &blank_characters) {
  blank_characters_ = blank_characters;
}

void StringParser::pushBlankCharacter(char blank_character) {
  blank_characters_ += blank_character;
}

void StringParser::pushBlockDelimiters(const std::string &open_pattern, const std::string &close_pattern) {
  block_openings_.emplace_back(open_pattern);
  block_closings_.emplace_back(close_pattern);
}

void StringParser::pushBlankDelimiters(const std::string &open_pattern, const std::string &close_pattern) {
  blank_block_openings_.emplace_back(open_pattern);
  blank_block_closings_.emplace_back(close_pattern);
}

void StringParser::pushTokenPattern(const std::string &name, const std::string &pattern) {
  token_patterns_[name] = pattern;
}

size_t StringParser::matchPrefixWithAny(const std::string &characters, const std::string &s, size_t start) {
  auto n = s.size();
  if (start >= n)
    return 0;
  size_t i = start;
  size_t j = i;
  do {
    j = i;
    if (i >= n)
      return n - start;
    bool consumed = false;
    for (size_t k = 0; k < characters.size() && !consumed; ++k)
      if (s[i] == characters[k]) {
        i++;
        consumed = true;
      }
  } while (j != i);
  return i - start;
}

size_t StringParser::matchPrefixWithAny(const std::vector<std::pair<std::string, std::string>> &block_delimiters,
                                        const std::string &s,
                                        size_t &block_content_start,
                                        size_t &block_content_size,
                                        size_t start) {
  size_t n = s.size();
  if (start >= n)
    return 0;
  // check if string starts with any opening delimiter (takes the first match!)
  for (const auto &block : block_delimiters) {
    if (start + block.first.size() >= n)
      continue;
    // check opening
    if (s.substr(start, block.first.size()) == block.first) {
      if (start + block.first.size() >= n)
        continue;
      // found opening, now look for the closing
      auto closing_index = s.find_first_of(block.second, start + block.first.size());
      if (closing_index == std::string::npos)
        continue;
      // there is a catch here, we may have another block of this type inside this block
      // so we need to search for a start too and compare with the ending we found
      // while ind is greater than new starting blocks indices we need to keep going...
      auto sub_block_start = s.find_first_of(block.first, start + block.first.size());
      while (sub_block_start != std::string::npos && closing_index != std::string::npos
          && sub_block_start < closing_index) {
        // push the end forward (there must be another block end, otherwise the input is broken
        closing_index = s.find_first_of(block.second, closing_index + block.second.size());
        sub_block_start = s.find_first_of(block.first, closing_index + block.second.size());
      }
      if (closing_index == std::string::npos)
        continue;
      // found closing
      block_content_start = start + block.first.size();
      block_content_size = closing_index - block_content_start;
      return block.first.size() + block_content_size + block.second.size();
    }
  }
  return 0;
}

size_t StringParser::matchPrefixWithAny(const std::vector<std::string> &patterns,
                                        const std::string &s,
                                        int &match_id,
                                        size_t start) {
  match_id = -1;
  size_t largest = 0;
  for (size_t i = 0; i < patterns.size(); ++i) {
    auto result = Str::regex::search(s.substr(start), std::string("^") + patterns[i]);
    if (result.empty())
      continue;
    auto n = result[0].length();
    if (largest < n) {
      match_id = static_cast<int>(i);
      largest = n;
    }
  }
  return largest;
}

size_t StringParser::startsWith(const std::vector<std::pair<std::string, std::string>> &block_delimiters,
                                const std::string &s,
                                size_t i) {
  size_t n = s.size();
  // check if string starts with any opening delimiter (takes the first match!)
  for (size_t j = 0; j < block_delimiters.size(); ++j) {
    if (i + block_delimiters[j].first.size() >= n)
      continue;
    // check opening
    if (s.substr(i, block_delimiters[j].first.size()) == block_delimiters[j].first)
      return j;
  }
  return block_delimiters.size();
}

size_t StringParser::consumeAllBlanks(const std::string &s, size_t start) const {
  auto n = s.size();
  size_t j = start;
  size_t i = start;
  do {
    j = i;
    size_t dummy;
    // TODO fix
//    i += matchPrefixWithAny(blank_block_openings_, s, dummy, dummy, i);
    i += matchPrefixWithAny(blank_characters_, s, i);
  } while (j != i);
  return i - start;
}

size_t StringParser::parseToken(const std::string &s, std::string &token_name, size_t start) const {
  size_t largest = 0;
  for (const auto &token_pair : token_patterns_) {
    auto result = Str::regex::search(s.substr(start), std::string("^") + token_pair.second);
    if (result.empty())
      continue;
    auto n = result[0].length();
    if (largest < n) {
      token_name = token_pair.first;
      largest = n;
    }
  }
  return largest;
}

Result<ParseTree> StringParser::parse(const std::string &s, bool copy_string) {
  // start with root note contemplating the entire input
  ParseTree::Node root;
  root.size = s.size();
  root.start = 0;
  root.name = "root";
  std::stack<ParseTree::Node> node_stack;
  node_stack.push(root);

  size_t end = s.size();
  size_t current_index = 0;
  // consume input as nodes appear
  while (current_index < end) {
    if (node_stack.empty()) {
      // the node stack should never be empty!
      HERMES_LOG_ERROR("empty parsing stack error");
      return Result<ParseTree>::error(HeResult::INVALID_INPUT);
    }
    size_t initial_iteration_start_index = current_index;
    // consume any blank characters / blocks
    current_index += consumeAllBlanks(s, current_index);

    // check for a closing block
    int closing_block_id = -1;
    current_index += matchPrefixWithAny(block_closings_, s, closing_block_id, current_index);
    if (closing_block_id >= 0) {
      // close current block
      if (closing_block_id != node_stack.top().block_id) {
        HERMES_LOG_ERROR("closing wrong block error");
        return Result<ParseTree>();
      }
      auto top = node_stack.top();
      node_stack.pop();
      if (node_stack.empty()) {
        // the node stack should never be empty!
        HERMES_LOG_ERROR("empty parsing stack error");
        return Result<ParseTree>();
      }
      top.size = current_index - top.start - block_closings_[closing_block_id].size();
      node_stack.top().children.emplace_back(top);
    }

    // check for a opening block
    int opening_block_id = -1;
    current_index += matchPrefixWithAny(block_openings_, s, opening_block_id, current_index);
    if (opening_block_id >= 0) {
      // a new block was found
      ParseTree::Node new_block;
      new_block.type = ParseTree::NodeType::BLOCK;
      new_block.start = current_index;
      new_block.block_id = opening_block_id;
      node_stack.push(new_block);
    }

    // check for a token
    ParseTree::Node token_node;
    token_node.type = ParseTree::NodeType::TOKEN;
    token_node.start = current_index;
    token_node.size = parseToken(s, token_node.name, current_index);
    current_index += token_node.size;
    if (token_node.size) {
      token_node.value = s.substr(token_node.start, token_node.size);
      node_stack.top().children.emplace_back(token_node);
    }

    // safety check here (current_index must move every iteration)
    if (current_index == initial_iteration_start_index) {
      HERMES_LOG_ERROR("unknown parsing error at character {}", s[current_index]);
      return Result<ParseTree>();
    }
  }

  return Result<ParseTree>(ParseTree(node_stack.top()));
}

}
