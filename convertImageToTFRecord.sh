#!/bin/bash

python build_image_data.py \
--train_directory=train \
--validation_directory=test \
--labels_file=labels.txt \
--output_directory=TF \
--train_shards=4 --validation_shards=0 --num_threads=2
