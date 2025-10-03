# Fine-Tuning DistilGPT2 for Food Recipe Generation

This repository contains a project for fine-tuning **DistilGPT2**, a lightweight GPT-2 model, on a large dataset of food recipes. The goal is to create a model capable of generating realistic and diverse recipes.

---

## Project Structure

```
Finetuning-distilgpt2-LLM-on-food-recepies
├─ corpus
    ├─ RecipeNLG_dataset.csv
├─ venv
├─ output
    ├─ saved_model
        ├─ checkpoint-12500
├─ finetuning_distilbert.ipynb
├─ LICENSE
└─ requirements.txt

```
---
## Base Causal Model used

Causal Model used in the finetuning process is distilbert/distilgpt2

## Dataset

The dataset used for fine-tuning is **RecipeNLG**, which is too large for GitHub (>2GB).  

**Instructions to use dataset:**

1. Download the dataset from [RecipeNLG official site](https://recipenlg.cs.put.poznan.pl/).  
2. Place the dataset in the `corpus/` folder.

## Tech Stack used and learnt

1) from datasets import Dataset : Utilized Dataset from Hugging Face datasets to efficiently load, preprocess, and manipulate structured text data. Converted pandas DataFrames to Dataset for seamless integration with PyTorch and Hugging Face Trainer.
2) from transformers import AutoTokenizer : Used AutoTokenizer to automatically load the correct tokenizer for DistilGPT2 or other Hugging Face models. Performed tasks like text tokenization, converting text to input IDs, padding, truncation, and attention mask creation.Gained practical experience with tokenizer customization, including special tokens and sequence length management for large datasets. Seamlessly integrated with Hugging Face Dataset objects and PyTorch DataLoaders for fine-tuning workflows.
3) from transformers import AutoModelForCausalLM : Utilized AutoModelForCausalLM to load pre-trained causal language models like DistilGPT2 for text generation tasks. Integrated seamlessly with tokenized inputs from AutoTokenizer and batched Dataset objects for efficient training. Gained hands-on experience with causal language modeling, loss computation, and model training workflows in PyTorch.
4) from transformers import DataCollatorForLanguageModeling : Used DataCollatorForLanguageModeling to dynamically batch and prepare tokenized inputs for causal language model training. Automatically handled masking and padding of sequences to create uniform input batches suitable for GPU training. Enabled efficient training on large datasets while maintaining correct input formats and attention masks for language modeling.Gained hands-on experience with batching strategies, masking techniques, and efficient data feeding for transformer-based models.
5) from transformers import TrainingArguments : Utilized TrainingArguments to configure training hyperparameters and workflow settings for fine-tuning transformer models. Managed key parameters such as learning rate, batch size, number of epochs, logging frequency, evaluation strategy, and gradient accumulation. Integrated seamlessly with Hugging Face Trainer to streamline the training and evaluation loop for DistilGPT2. Gained practical experience in controlling large-scale language model training, optimizing GPU usage, and managing checkpoints.
6) from transformers import Trainer : Integrated seamlessly with Dataset, DataCollatorForLanguageModeling, and TrainingArguments for end-to-end training and evaluation.Handled training loop, evaluation, checkpointing, and metrics logging automatically, reducing boilerplate code.
7) Inferencing/GTesting :
     Text Generation Parameters:

      max_new_tokens=100 – limit output to 100 new tokens.

      do_sample=True – enable stochastic sampling for diverse outputs.

      top_k=50, top_p=0.98 – use top-k and nucleus sampling to balance creativity and coherence.

      repetition_penalty=1.2 – reduce repeated phrases.

      temperature=0.7 – control randomness in token selection.

      eos_token_id – ensure the model stops at the end-of-sequence token.
8) Output: The model produces coherent, contextually relevant recipes based on the input prompt.


