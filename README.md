# **BigBasket Inventory Prediction using Fine-Tuned LLaMA2-7B and Falcon-7B for Inventory Classification**

## **Project Overview**
This project fine-tunes two popular open-source Large Language Models (LLMs), **LLaMA2-7B** and **Falcon-7B**, to classify product inventory types based on product names. Such a system is valuable for major retail chains, enabling efficient inventory organization.

## **Technical Details**

### **Models Used**
#### **1. LLaMA2-7B**
[LLaMA2 (Large Language Model Meta AI)](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) is a transformer-based language model developed by Meta AI, optimized for **text generation and classification tasks**.

- **Architecture**: Decoder-only Transformer with **pre-normalization** (RMSNorm) and **SwiGLU activation functions**.
- **Tokenizer**: Used a **byte pair encoding (BPE) tokenizer** trained on the pretraining corpus.
- **Fine-tuning Mechanism**: Used **low-rank adaptation (LoRA)** for efficient parameter tuning.  


#### **2. Falcon-7B**
[Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) is an autoregressive **decoder-only** model built by the **Technology Innovation Institute (TII)**. It is optimized for efficient text generation with reduced memory footprint.

- **Architecture**:  
  - Built on the **GPT-style causal transformer** architecture.  
  - Uses **multi-query attention (MQA)** instead of multi-head attention, significantly **reducing memory overhead**.  
  - Employs **rotary positional embeddings (RoPE)** for better long-context understanding.  
- **Fine-tuning Mechanism**: Optimized for **QLoRA (Quantized LoRA) fine-tuning**, allowing efficient adaptation with minimal computational overhead.  


## **Project Structure**
The repository is organized as follows:

1. **Data Preprocessing**  
   - Product names are concatenated with their corresponding inventory types to create labeled training data.  
   - Tokenization is performed using the respective model-specific tokenizers.  
   - Dataset is split into training and validation sets.  

2. **Fine-Tuning**  
   - **LLaMA2-7B and Falcon-7B** are fine-tuned separately using **Hugging Face’s `transformers` and `bitsandbytes` libraries**.  
   - Mixed precision training is employed using **bfloat16 (BF16) or float16 (FP16)** depending on the hardware.  
   - Fine-tuning is performed using **LoRA, QLoRA, or full fine-tuning**, depending on available computational resources.  

3. **Evaluation and Inference**  
   - The models are evaluated using **accuracy, F1-score, and confusion matrices** to determine classification performance.  
   - The fine-tuned models are deployed for inference on test data to predict inventory categories.  


### **Software Dependencies**
- Python 3.8+  
- `transformers` (Hugging Face)  
- `torch` (PyTorch)  
- `peft` (for parameter-efficient fine-tuning)  
- `bitsandbytes` (for QLoRA and mixed-precision training)  

## **Fine-Tuning Considerations**
- **Memory Optimization**:  
  - Falcon-7B benefits from **multi-query attention**, reducing VRAM requirements.  
  - LLaMA2-7B requires **gradient checkpointing** for efficient training.  
- **Training Stability**:  
  - Regularization techniques such as **weight decay and layer-wise learning rate decay** help stabilize training.  
  - Both models support **learning rate warm-up and cosine annealing decay** for optimized convergence.  
