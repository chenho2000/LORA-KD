# Task-Specific Low-Rank Adapted Knowledge Distillation for Large Language Models

## Fine-Tuning Scripts

- **Standard Fine-Tuning:** `/Fine-Tuning/run_glue.py`
- **LoRA Fine-Tuning:** `/Fine-Tuning/run_glue_LoRA.py`
- **Standard TinyBERT Knowledge Distillation:** `task_distill.py`
- **LoRA TinyBERT Knowledge Distillation:** `task_distill_LoRA.py`

## Android Deployment

Please follow the instructions in `/android/android_deployment.ipynb` to convert your model to ONNX format and place it in the main folder of the Android app.

## Overview

In recent years, Natural Language Processing (NLP) has become essential for various tasks, particularly in everyday applications. Fine-tuning large language models allows them to excel in different areas. Prominent models include BERT, T5, XLNet by Google, the GPT series by OpenAI, and RoBERTa by Facebook AI. These models are large and parameter-rich, enabling them to learn and adapt to various tasks effectively. However, their substantial size poses challenges for fine-tuning due to hardware constraints. Specialized GPUs or CPUs are often required for efficient training, while consumer-grade GPUs may fall short, limiting the public use of these models. Additionally, deploying these models on mobile devices introduces further challenges due to their significant memory requirements. This project aims to address these issues by optimizing model size and computational efficiency, enabling fine-tuning on consumer-grade GPUs and efficient operation on mobile devices. The goal is to streamline the process from fine-tuning to deployment, minimizing hardware requirements and reducing reliance on cloud services.

## References

- [TinyBERT GitHub Repository](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
- [Using scikit-learn Models in Android Applications](https://github.com/shubham0204/Scikit_Learn_Android_Demo/tree/main)
- [Huggingface Transformer](https://github.com/huggingface/transformers/tree/main)
