#!/bin/bash
pip install transformers[onnx]
python -m transformers.onnx --model=./ --feature=sequence-classification ./
trtexec --onnx=model.onnx --saveEngine=model_bs16.plan --minShapes=input_ids:1x128,attention_mask:1x128 --optShapes=input_ids:1x128,attention_mask:1x128 --maxShapes=input_ids:1x128,attention_mask:1x128 --fp16 --verbose --workspace=14000 | tee conversion_bs16_dy.txt
