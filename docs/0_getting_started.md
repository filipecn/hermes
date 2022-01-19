# Getting Started

![ubuntu/gcc](https://github.com/filipecn/hermes/actions/workflows/gcc_compiler.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/filipecn/hermes/badge.svg?branch=main)](https://coveralls.io/github/filipecn/hermes?branch=main)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/filipecn/hermes)

\tableofcontents

Welcome to Hermes! A multi-purpose library that may (hopefully) help you quickly set up a 
`C++`/`CUDA` project by providing you a bunch of auxiliary tools and structures. 

> This library is my personal lib I use in my projects, at my own risk :) Please keep it in mind. But
> I really hope it can be useful to you too.

## Features
Although most of `hermes` classes live in a single namespace, you will find the files organized 
in folders, such as: `geometry`, `data_structures`, `storage`, etc. You can find examples and
documentation for each of these groups here:

| group | description |
|--------------|--------|
| [common](3_common.md) | auxiliary classes for debugging, time profiling, iterating, logging, strings, files, argument parsing, etc  |
| [geometry](4_geometry.md) | geometry related objects, functions and utilities such as vector, point, matrix, transforms, intersection tests, line, plane, etc   |
| [storage](5_storage.md) | memory classes: allocators, memory blocks, array of structs, etc  |
| [numeric](6_numeric.md) | math operations, interpolation, intervals   |

## Download 

You can find Hermes in [github](https://github.com/filipecn/hermes).

> Please check the [build](1_build_and_install.md) and [link](2_linking.md) instructions to learn
how to build and use `hermes` into your project.

For the impatient, here is what you can do:
```shell
git clone https://github.com/filipecn/hermes.git
cd hermes
mkdir build
cd build
cmake .. -DINSTALL_PATH=/my/install/location
make -j8 install
```
and to compile along with your code here is what to do:
```shell
g++ -I/my/install/location/include                \
    -l/my/install/location/lib/libhermes.a         \
    main.cpp  
```

## Contact

Please feel free to contact me by [e-mail](mailto:filipedecn@gmail.com) :)
