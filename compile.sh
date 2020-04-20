#!/bin/bash

set -e

if [ -z ${VCPKG_ROOT+x} ]; then
    VCPKG_ROOT="vcpkg"
fi

mkdir -p build
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
make -C build
