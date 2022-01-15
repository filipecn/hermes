# Hermes 
![ubuntu/gcc](https://github.com/filipecn/hermes/actions/workflows/gcc_compiler.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/filipecn/hermes/badge.svg?branch=main)](https://coveralls.io/github/filipecn/hermes?branch=main)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
----
Hermes is a multi-purpose `C++`/(optionally `CUDA`) library with lots of data structures and algorithms. 
The purpose is to serve as a starting point to your project by providing some common and auxiliary tools.

Here is the list of some things you can get from hermes:

|  | some features |
|--------------|--------|
| **geometry**     | vector, point, matrix, transforms, intersection tests, line, plane   |
| **numeric**     | math operations, interpolation, intervals   |
| **storage**     | allocators, memory blocks, array of structs  |
| **common**       | code profiling, logging, string operations, filesystem, arg parser   |

You can find Hermes in [github](https://github.com/filipecn/hermes).

## Usage
Please check the [docs]() for a better introduction and detailed examples :)

```c++
// TODO
```

## Build

In order to build and use Hermes (with no options), you can do as usual with `cmake`:
```shell
git clone https://github.com/filipecn/hermes.git
cd hermes
mkdir build
cd build
cmake ..
make -j8 install
```

Depending on what you want to compile, you may need to set some `cmake` options:

| variable | description | default  |
|--------------|--------|-----|
| BUILD_ALL  | set all variables below to ON | OFF |
| BUILD_WITH_CUDA  | compiles with support to CUDA | OFF |
| BUILD_TESTS  | build unit-tests | OFF |
| BUILD_EXAMPLES  | build examples | OFF |
| BUILD_DOCS  | generates documentation | OFF |

Suppose you would like to use `CUDA` and also perform the unit tests, your `cmake` command then will look like this:
````shell
cmake .. -DBUILD_WITH_CUDA=ON -DBUILD_TESTS=ON
````

## Dependencies
Hermes is dependency-free library, so there is no need to install/compile anything else, besides **optionally** `CUDA`.
 - [catch2](https://github.com/catchorg/Catch2) - is used for the unit-tests, but their header is already included in the source :)


## TODO
- Find nvcc automatically (CMAKE_CUDA_COMPILER)
- online documentation
- ~~README~~ :)

## Notes
- This library is my personal lib that I use in my projects, at my own risk :) Please keep it in mind.
- I've been developing Hermes under Ubuntu 20.04, I have no idea how it behaves on other systems (or distributions).

## Contact

Please feel free to contact me :)

[e-mail](mailto:filipedecn@gmail.com)