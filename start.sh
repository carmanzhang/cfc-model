#!/bin/bash

sudo docker build -t carmanzhang/cfc:1.0 -f ./citation_function.Dockerfile .

cur_path=${PWD}/citation_function_prediction
echo $cur_path

sudo docker run -dit --rm -p 8501:8501 \
-v $cur_path:/models/citation_function_prediction \
-e MODEL_NAME=citation_function_prediction tensorflow/serving:2.0.0


sudo docker run -dit --rm --net host carmanzhang/cfc:1.0
