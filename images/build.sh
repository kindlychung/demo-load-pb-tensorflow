#!/usr/bin/env bash
source activate tf
mkdir -p ./tf_records
build_image_data.py                                            \
	--train_directory=./train                              \
	--validation_directory=./validate                      \
	--output_directory=./tf_records                        \
	--labels_file=./labels.txt   
