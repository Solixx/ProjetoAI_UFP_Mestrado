import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

import requests
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import math

import transformers
transformers.logging.set_verbosity_info()

# Load environment variables
load_dotenv()

def load_dataset_from_url(url):
    """Loads dataset from a URL (JSONL format)."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    data = []
    # Iterate with index for more informative error messages
    for i, line_content in enumerate(response.text.splitlines()):
        line_content = line_content.strip()
        if not line_content:  # Skip empty lines
            continue
        try:
            data.append(json.loads(line_content))
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping line {i+1} due to JSON decode error: {e}. Content: '{line_content[:100]}...'")
    return data

def create_finetuning_dataset(tokenizer, data_list, max_seq_length=512):
    """
    Creates a dataset for fine-tuning in the format:
    "Instruction: {instruction}\nResponse: {response}{eos_token}"
    """
    processed_texts = []
    for item in data_list:
        if 'instruction' in item and 'response' in item:
            text = f"Instruction: {item['instruction']}\nResponse: {item['response']}{tokenizer.eos_token}"
            processed_texts.append(text)
        else:
            print(f"Warning: Skipping item due to missing 'instruction' or 'response': {item}")


    tokenized_inputs = tokenizer(
        processed_texts,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    dataset_dict = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(tokenized_inputs.input_ids)):
        input_ids = tokenized_inputs.input_ids[i]
        attention_mask = tokenized_inputs.attention_mask[i]
        
        dataset_dict["input_ids"].append(input_ids)
        dataset_dict["attention_mask"].append(attention_mask)
        dataset_dict["labels"].append(list(input_ids)) 

    return Dataset.from_dict(dataset_dict)

def main():
    # Model Configuration
    model_name = "google/gemma-2-2b-it"
    dataset_url = "https://raw.githubusercontent.com/BrunoSilva077/dataset/main/ufp-courses-dataset.jsonl"
    output_dir = "./gemma2-finetuned-lora"
    lora_r = 16 
    lora_alpha = 32 
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    max_seq_length = 512 
    num_train_epochs = 3
    per_device_train_batch_size = 1 
    gradient_accumulation_steps = 4 
    learning_rate = 2e-4
    logging_steps = 10
    save_steps = 50 
    save_total_limit = 2

    hf_token = os.getenv("TOKEN") or os.getenv("HF_TOKEN")

    if not hf_token and ("gemma" in model_name.lower() or "llama" in model_name.lower()): # Add other model types if needed
        print("ERROR: Hugging Face token not found (TOKEN or HF_TOKEN env var).")
        print(f"Model '{model_name}' is gated and requires authentication.")
        return

    # --- CUDA and Compute Dtype Check ---
    use_quantization = False
    low_cpu_mem_usage_for_loading = False
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This script is designed for QLoRA ")

        compute_dtype = torch.float32 
        model_torch_dtype = torch.float32
        training_fp16 = False
        training_bf16 = False
        low_cpu_mem_usage_for_loading = True
    else:
        use_quantization = True 
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            model_torch_dtype = torch.bfloat16
            training_fp16 = False
            training_bf16 = True
            print("CUDA is available and bfloat16 is supported. Using bfloat16.")
        else:
            compute_dtype = torch.float16
            model_torch_dtype = torch.float16
            training_fp16 = True
            training_bf16 = False
            print("CUDA is available but bfloat16 is not supported. Using float16.")

    # Load Dataset
    print("Loading dataset...")
    raw_data = load_dataset_from_url(dataset_url)
    if not raw_data:
        print("Failed to load or parse dataset, or dataset is empty. Exiting.")
        return
    print(f"Loaded {len(raw_data)} raw data entries.")

    # Initialize Tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer.pad_token to tokenizer.eos_token")

    # Prepare Dataset for Fine-tuning
    print("Preparing dataset for fine-tuning...")
    train_dataset = create_finetuning_dataset(tokenizer, raw_data, max_seq_length)
    print(f"Dataset prepared. Number of examples: {len(train_dataset)}")
    if len(train_dataset) == 0:
        print("No data to train on after processing. Exiting.")
        return

    # Initialize Model with Quantization and LoRA
    print(f"Loading base model {model_name}...")
    
    quantization_config_to_pass = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, 
            bnb_4bit_use_double_quant=False,
        )
        quantization_config_to_pass = bnb_config
        print(f"Configured BitsAndBytes for 4-bit quantization with compute_dtype: {compute_dtype}.")
    else:
        print("Skipping BitsAndBytes 4-bit quantization (CUDA not available or quantization disabled).")


    try:
        device_map_config = "auto"
        if not torch.cuda.is_available():
            print("CUDA not available. Forcing model to CPU. Using low_cpu_mem_usage.")
            device_map_config = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config_to_pass,
            torch_dtype=model_torch_dtype,
            device_map=device_map_config, 
            low_cpu_mem_usage=low_cpu_mem_usage_for_loading,
            token=hf_token,
        )
        print(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        import traceback
        print(f"ERROR: Failed to load model '{model_name}")
        print(f"!!! Reason: {e}                                                              !!!")
        print(f"Error Type: {type(e).__name__}")
        print("Full Traceback:")
        return
    
    if use_quantization: 
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training (QLoRA).")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False 
    if hasattr(model, 'gradient_checkpointing_enable') and use_quantization:
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing on the model.")

    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)
    total_train_steps = num_update_steps_per_epoch * num_train_epochs
    
    print(f"--- Training Configuration ---")
    print(f"Total training examples: {len(train_dataset)}")
    print(f"Epochs: {num_train_epochs}")
    print(f"Per device batch size: {per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Optimizer update steps per epoch: {num_update_steps_per_epoch}")
    print(f"Total estimated training optimization steps: {total_train_steps}")
    print(f"Logging each {logging_steps} steps. Saving each {save_steps} steps.")

    if total_train_steps == 0:
        print("!!! CRITICAL WARNING: Estimated total training steps is 0. Training cannot proceed.")
        return
    if logging_steps > total_train_steps and total_train_steps > 0:
        print(f"!!! WARNING: 'logging_steps' ({logging_steps}) is greater than total estimated training steps ({total_train_steps}).")
        print(f"!!! You may not see any training logs until the very end or not at all. Consider reducing 'logging_steps'.")


    # Set up Training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_dir=f"{output_dir}/logs",   # Specify logging directory
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=training_fp16,  
        bf16=training_bf16,   
        optim="paged_adamw_8bit" if use_quantization else "adamw_torch", 
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        remove_unused_columns=True,
        disable_tqdm=False,                 # To show progress bars
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    try:
        train_result = trainer.train()
        print("Fine-tuning completed successfully.")
        
        # Log metrics
        metrics = train_result.metrics
        print(f"Training metrics: {metrics}")
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        print("Saving final model adapter and tokenizer...")

        os.makedirs(output_dir, exist_ok=True) 
        
        # Path for LoRA adapter
        final_model_path = os.path.join(output_dir, "final_model_adapter")
        trainer.save_model(final_model_path)

        # Save tokenizer alongside adapter
        tokenizer.save_pretrained(final_model_path) 
        print(f"Fine-tuned LoRA adapter and tokenizer saved to {final_model_path}")


    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR DURING TRAINING OR SAVING:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()