#!/usr/bin/env sh
set -e

./cmake_build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt $@
