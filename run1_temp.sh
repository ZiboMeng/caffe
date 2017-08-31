#!/bin/bash

./examples/ckplus/scripts/siamese_bi_con_orig/create_protos.sh
./examples/ckplus/scripts/siamese_bi_con_orig/train.sh

./examples/ckplus/scripts/siamese_exp_con_orig/create_protos.sh
./examples/ckplus/scripts/siamese_exp_con_orig/train.sh

