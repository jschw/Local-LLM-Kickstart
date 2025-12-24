#!/bin/bash
pkg update
pkg install -y clang wget cmake git
cd ~/storage/shared/chatshell
mkdir Llamacpp
cd Llamacpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release