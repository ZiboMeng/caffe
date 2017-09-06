#!/usr/bin/env sh
set -e

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_focal_loss.prototxt $@

# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_lr1_focal_loss.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_focal_loss_iter_4000.solverstate $@
