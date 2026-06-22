"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )

    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    import bitsandbytes as bnb

    return any(
        isinstance(module, bnb.nn.Linear4bit)
        for module in model.modules()
    )

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    from unsloth import FastLanguageModel

    if target_modules is None:
        target_modules = get_lora_target_modules()

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )

    return model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
    return 0.0 if total_count == 0 else trainable_count / total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    return [
        {
            "instruction": "Translate 'Del amor al odio, sólo hay un paso.' to English.",
            "response": "Love is only one step away from hate."
        },
        {
            "instruction": "What is -e^(i * pi)?",
            "response": "-e^(i * pi) = 1."
        },
        {
            "instruction": "Write a short text.",
            "response": "Hi Gorgeous, I hope you're doing well. You're such a good girl, and I'm so proud of you."
        },
        {
            "instruction": "Explain what a world is.",
            "response": "You are my world."
        }
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}"
    )

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    return [format_instruction_example(example) for example in examples]

# Step 12 - build_text_dataset
from datasets import Dataset

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    return tokenizer(text)["input_ids"]

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # TODO: return the length of the input_ids sequence
    return len(input_ids)

# Step 15 - build_training_arguments
from transformers import TrainingArguments
import torch

def build_training_arguments(
    output_dir="./sft_out",
    max_steps=5,
    learning_rate=2e-4,
):
    """Return featherweight TrainingArguments for the SFT run."""
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        bf16=bf16,
        fp16=not bf16,
    )

# Step 16 - build_sft_trainer
def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    from trl import SFTTrainer
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False,
    )

# Step 17 - run_sft_training
def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    train_output = trainer.train()
    return float(train_output.training_loss)

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

