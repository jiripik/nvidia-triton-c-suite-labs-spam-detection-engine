# NVIDIA Triton Spam Detection Engine

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11+-orange.svg)](https://pytorch.org)
[![NVIDIA Triton](https://img.shields.io/badge/NVIDIA%20Triton-21.08-76B900.svg)](https://developer.nvidia.com/nvidia-triton-inference-server)

A high-performance spam detection engine built with **DistilBERT** and deployed to **AWS SageMaker** using **NVIDIA Triton Inference Server**. This project demonstrates two deployment approaches: TorchScript and TensorRT-optimized models for production-grade NLP inference.

📖 **Blog Post**: [NVIDIA Triton Spam Detection Engine of C-Suite Labs](https://jiripik.com/2022/06/30/nvidia-triton-spam-detection-engine-of-c-suite-labs/)

---

## 🎯 Overview

This repository provides end-to-end pipelines for training a DistilBERT-based spam classifier and deploying it to AWS SageMaker with two different optimization strategies:

| Approach | Model Format | Inference Server | Use Case |
|----------|-------------|------------------|----------|
| **TorchScript** | `.pt` | PyTorch Serving | Quick deployment, flexibility |
| **TensorRT** | `.plan` | NVIDIA Triton | Maximum GPU performance |

## ✨ Features

- 🤖 **DistilBERT Fine-tuning** - Train on custom spam datasets with HuggingFace Transformers
- ⚡ **TensorRT Optimization** - FP16 inference for maximum GPU throughput
- 🚀 **NVIDIA Triton Integration** - Production-ready inference serving
- ☁️ **AWS SageMaker Deployment** - Scalable cloud inference endpoints
- 📊 **Benchmarking Tools** - Concurrent inference testing with throughput metrics

## 📁 Project Structure

```
├── SpamDetection-TorchScript.ipynb   # TorchScript deployment pipeline
├── SpamDetection-Triton.ipynb        # Triton/TensorRT deployment pipeline
├── spamdetection-torchscript/
│   └── serve.py                      # SageMaker inference handler
├── triton-serve-trt/
│   └── bert/
│       └── config.pbtxt              # Triton model configuration
├── workspace-trt/
│   └── generate_models.sh            # ONNX to TensorRT conversion script
└── LICENSE
```

## 🛠️ Prerequisites

- Python 3.8+
- AWS Account with SageMaker access
- NVIDIA GPU (for TensorRT optimization)
- Docker (for TensorRT model generation)

### Required Libraries

```bash
pip install torch transformers sagemaker boto3
pip install nvidia-pyindex tritonclient[http]
pip install scikit-learn nltk datasets
```

## 🚀 Quick Start

### Option 1: TorchScript Deployment

1. **Train the Model**
   ```python
   # Run cells in SpamDetection-TorchScript.ipynb
   # Trains DistilBERT on spam dataset for 4 epochs
   ```

2. **Export to TorchScript**
   ```python
   traced_model = torch.jit.trace(model, dummy_inputs)
   torch.jit.save(traced_model, './spamdetection-torchscript/model/model.pt')
   ```

3. **Deploy to SageMaker**
   ```python
   predictor = model.deploy(
       instance_type="ml.p3.2xlarge",
       endpoint_name="spamdetection-torchscript"
   )
   ```

### Option 2: Triton + TensorRT Deployment

1. **Train the Model**
   ```python
   # Run cells in SpamDetection-Triton.ipynb
   ```

2. **Convert to TensorRT**
   ```bash
   docker run --gpus=all -v $(pwd)/workspace-trt:/workspace \
       nvcr.io/nvidia/pytorch:21.08-py3 /bin/bash generate_models.sh
   ```

3. **Deploy with Triton**
   ```python
   container = {
       "Image": triton_image_uri,
       "ModelDataUrl": model_uri,
       "Environment": {"SAGEMAKER_TRITON_DEFAULT_MODEL_NAME": "bert"}
   }
   ```

## 📝 Model Configuration

### Triton Config (`config.pbtxt`)

```protobuf
name: "bert"
platform: "tensorrt_plan"
max_batch_size: 128
input [
  { name: "input_ids", data_type: TYPE_INT32, dims: [128] },
  { name: "attention_mask", data_type: TYPE_INT32, dims: [128] }
]
output [
  { name: "logits", data_type: TYPE_FP32, dims: [2] }
]
```

## 🧪 Testing the Endpoint

```python
test_texts = [
    "Oh k...i'm watching here:)",  # Ham
    "You are awarded with a £1500 Bonus Prize, call 09066364589",  # Spam
]

for text in test_texts:
    prediction = get_prediction(text)
    label = "SPAM" if prediction == 1 else "HAM"
    print(f"{label}: {text[:50]}...")
```

## 📈 Performance

The TensorRT-optimized model provides significant throughput improvements over the TorchScript version when deployed on GPU instances like `ml.p3.2xlarge`.

## 🧹 Cleanup

Don't forget to delete your SageMaker resources to avoid charges:

```python
# TorchScript
predictor.delete_endpoint()
predictor.delete_model()

# Triton
sm.delete_endpoint(EndpointName=endpoint_name)
sm.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
sm.delete_model(ModelName=sm_model_name)
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [AWS SageMaker](https://aws.amazon.com/sagemaker/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

## 👤 Author

**Jiri Pik** - [jiripik.com](https://jiripik.com)

---

⭐ If you found this project helpful, please consider giving it a star!

