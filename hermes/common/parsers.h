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
///\file parser.h
///\author FilipeCN (filipedecn@gmail.com)
///\date 2022-11-11
///
///\brief

#ifndef HERMES_HERMES_COMMON_PARSERS_H
#define HERMES_HERMES_COMMON_PARSERS_H

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <hermes/common/result.h>

namespace hermes {

// *********************************************************************************************************************
//                                                                                                          ParseTree
// *********************************************************************************************************************
/// \brief The ParseTree represents a parsing tree (or derivation tree) generated by a parser.
///        The tree consists in a hierarchy of nodes containing portions of the text identified by the parser rules.
class ParseTree {
  friend class StringParser;
public:
  /// \brief Parse tree nodes can be of different types:
  /// \note INPUT: a piece of text which might contain other nodes
  /// \note TOKEN: a single token (a word identified by the parser)
  /// \node BLOCK: a block of of text enclosed by block delimiters which might contain other nodes
  enum class NodeType {
    INPUT = 0, //!< a piece of text which might contain other nodes
    TOKEN = 1, //!< a single token (a word identified by the parser)
    BLOCK = 2  //!< a block of of text enclosed by block delimiters which might contain other nodes
  };
  // *******************************************************************************************************************
  //                                                                                                  ParseTree::Node
  // *******************************************************************************************************************
  /// \brief A node represents a element in the parsed string
  /// \note The node can also contain sub-nodes
  struct Node {
    std::string name;
    std::string value;
    size_t start{};
    size_t size{};
    int block_id{}; //!< block id of the parser's block list (root is -1)
    std::vector<Node> children;
    NodeType type{NodeType::INPUT};
  };
  // *******************************************************************************************************************
  //                                                                                                          METHODS
  // *******************************************************************************************************************
  void iterate(const std::function<bool(const Node &)> &callback) const;
  // *******************************************************************************************************************
  //                                                                                                            DEBUG
  // *******************************************************************************************************************
  /// \brief Dumps all contant of the tree
  /// \param os
  /// \param parse_tree
  /// \return
  friend std::ostream &operator<<(std::ostream &os, const ParseTree &parse_tree);
private:
  ///
  /// \param type
  /// \return
  static std::string typeName(NodeType type) {
#define DATA_TYPE_NAME(Type) \
      if(NodeType::Type == type) \
    return #Type;
    DATA_TYPE_NAME(INPUT)
    DATA_TYPE_NAME(TOKEN)
    DATA_TYPE_NAME(BLOCK)
    return "INVALID NODE TYPE";
#undef DATA_TYPE_NAME
  }
  ///
  /// \param node
  explicit ParseTree(Node node) {
    root_ = std::move(node);
  }
  ///
  Node root_;
};

// *********************************************************************************************************************
//                                                                                                       StringParser
// *********************************************************************************************************************
/// \brief General token parser for strings
class StringParser {
public:
  // *******************************************************************************************************************
  //                                                                                                   STATIC METHODS
  // *******************************************************************************************************************
  //                                                                                                     construction
  /// \brief A simple and rough parser for c-like languages (far from complete).
  /// \return StringParser object with configured rules.
  static StringParser cLanguage();
  //                                                                                                          parsing
  /// \brief Matches the prefix (from start) of the string with any any sequence in the given set of characters.
  /// \param characters
  /// \param s raw string
  /// \param start starting index in the string s
  /// \return the size of the matched sequence
  static size_t matchPrefixWithAny(const std::string &characters, const std::string &s, size_t start = 0);
  /// \brief Matches the prefix (from start) of the string with any block
  /// \note This function stops at the first match (blocks with equal opening prefix will give undefined behavior)
  /// \param block_delimiters
  /// \param s raw string
  /// \param block_content_start index of the string of the first character in the content
  /// \param block_content_size content's size
  /// \param start starting index in the string for the prefix
  /// \return size of the matched block (opening + content + closing)
  static size_t matchPrefixWithAny(const std::vector<std::pair<std::string, std::string>> &block_delimiters,
                                   const std::string &s,
                                   size_t &block_content_start,
                                   size_t &block_content_size,
                                   size_t start = 0);
  /// \brief Matches the prefix (from start) of the string with any block.
  /// \note This function matches with the block with the largest delimiter.
  /// \param block_openings
  /// \param s
  /// \param match_id receives the input vector index of the matched block, -1 otherwise
  /// \param start starting index at s
  /// \return size of the opening patter for the matched block
  static size_t matchPrefixWithAny(const std::vector<std::string> &block_openings,
                                   const std::string &s,
                                   int &match_id,
                                   size_t start = 0);
  /// \brief Checks if string starts with a block delimiter (first element in pairs)
  /// \param block_delimiters
  /// \param s raw string
  /// \param i starting index
  /// \return the index of the block with matching start
  static size_t startsWith(const std::vector<std::pair<std::string, std::string>> &block_delimiters,
                           const std::string &s,
                           size_t i = 0);
  // *******************************************************************************************************************
  //                                                                                                     CONSTRUCTORS
  // *******************************************************************************************************************
  StringParser();
  virtual ~StringParser();
  // *******************************************************************************************************************
  //                                                                                                            RULES
  // *******************************************************************************************************************
  //                                                                                         empty / blank characters
  /// \brief Defines blank characters.
  /// \note Blank characters are ascii characters that are skipped by the parser between tokens.
  /// \note By default the parser considers ' ', '\t', and '\n'.
  /// \param blank_characters a list of blank characters.
  /// \example " \t\n"
  void setBlankCharacters(const std::string &blank_characters);
  /// \brief Appends a blank character.
  /// \note Blank characters are ascii characters that are skipped by the parser between tokens.
  /// \param blank_character a single blank character
  /// \example '\t'
  void pushBlankCharacter(char blank_character);
  /// \brief Appends a blank block enclosing delimiters.
  /// \note Enclosing delimiters define a distinct block of text that are considered blank text.
  /// \example c-like comments: open = "/*", close = "*/"
  /// \param open_pattern
  /// \param close_pattern
  void pushBlankDelimiters(const std::string &open_pattern, const std::string &close_pattern);
  //                                                                                                 enclosing blocks
  /// \brief Appends a block enclosing delimiters.
  /// \note Enclosing delimiters define a distinct block of text.
  /// \example c-like comments: open = "/*", close = "*/"
  /// \param open_pattern
  /// \param close_pattern
  void pushBlockDelimiters(const std::string &open_pattern, const std::string &close_pattern);
  //                                                                                                           tokens
  /// \brief Appends a token - pattern pair to the parser.
  /// \param name
  /// \param regex_pattern
  void pushTokenPattern(const std::string &name, const std::string &regex_pattern);
  // *******************************************************************************************************************
  //                                                                                                          PARSING
  // *******************************************************************************************************************
  /// \brief Parses a given string into a parsing tree.
  /// \param text
  /// \param copy_string
  /// \return
  Result<ParseTree> parse(const std::string &text, bool copy_string = true);
private:
  /// \brief Moves index i past any sequence of blank blocks and blank characters.
  /// \note This function can jump multiple blank blocks.
  /// \param s
  /// \param start
  /// \return number of matched characters
  [[nodiscard]] size_t consumeAllBlanks(const std::string &s, size_t start = 0) const;
  /// \brief Consumes a token
  /// \note This function matches with the largest token
  /// \param s
  /// \param token_name
  /// \param start
  /// \return
  [[nodiscard]] size_t parseToken(const std::string &s, std::string &token_name, size_t start = 0) const;

  std::string blank_characters_ = " \t\n";
  std::unordered_map<std::string, std::string> token_patterns_;
  ///
  std::vector<std::string> blank_block_openings_;
  std::vector<std::string> blank_block_closings_;
  ///
  std::vector<std::string> block_openings_;
  std::vector<std::string> block_closings_;
};

}

#endif //HERMES_HERMES_COMMON_PARSERS_H
