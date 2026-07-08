#!/usr/bin/env python3
"""
Project Drona — Fine-tuning V2 (Optimized for 1024 Context)
500 complex tactical examples, RTX 5060 8GB safe.
"""
import os, json, torch
from pathlib import Path
from datasets import Dataset

ADAPTER_DIR = Path("models/drona-lora-v2")
TRAIN_FILE  = Path("fine_tune/train_v2.jsonl")
VAL_FILE    = Path("fine_tune/val_v2.jsonl")

# Verify GPU before anything else
assert torch.cuda.is_available(), "GPU not found — check PyTorch install"
free_gb = torch.cuda.mem_get_info()[0] / 1e9
print(f"Free VRAM: {free_gb:.1f} GB")
assert free_gb > 5.0, f"Not enough VRAM: {free_gb:.1f}GB < 5GB needed"

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Increase max_seq_length to 1024 to accommodate multi-turn reasoning
# 3B model @ 1024 context fits easily in 8GB VRAM (~4.5GB usage)
MAX_SEQ_LENGTH = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,
    dtype          = None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                          = 8,
    target_modules             = ["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"],
    lora_alpha                 = 16,
    lora_dropout               = 0.0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 42,
)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def format_example(ex):
    return tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False
    )

# Pre-format dataset for efficiency
train_data = Dataset.from_list(
    [{"text": format_example(ex)} for ex in load_jsonl(TRAIN_FILE)]
)
val_data = Dataset.from_list(
    [{"text": format_example(ex)} for ex in load_jsonl(VAL_FILE)]
)
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

training_args = TrainingArguments(
    output_dir                  = "outputs_v2",
    max_steps                   = 200,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    warmup_steps                = 20,
    fp16                        = False,
    bf16                        = True,
    logging_steps               = 10,
    save_steps                  = 50,
    save_total_limit            = 3,
    eval_strategy               = "steps",
    eval_steps                  = 50,
    load_best_model_at_end      = True,
    report_to                   = "none",
    gradient_checkpointing      = True,
    optim                       = "adamw_8bit",
    dataloader_num_workers      = 0,
    seed                        = 42,
)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_data,
    eval_dataset       = val_data,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    args               = training_args,
    packing            = False, # Ensure proper sequence handling
)

# Monitor VRAM during training
print(f"VRAM before train: {torch.cuda.memory_allocated()/1e9:.1f}GB")
trainer.train()
print(f"VRAM after train: {torch.cuda.memory_allocated()/1e9:.1f}GB")

ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print(f"Adapter saved to {ADAPTER_DIR}")

print("Converting to GGUF directly via Unsloth...")
model.save_pretrained_gguf("models/drona-v2", tokenizer, quantization_method = "q8_0")

print("""
NEXT STEPS:
1. Register with Ollama:
   cat > models/Modelfile_v2 << 'EOF'
   FROM /home/vahin/drona/models/drona-v2-unsloth.Q8_0.gguf
   PARAMETER temperature 0.1
   PARAMETER num_predict 600
   SYSTEM "You are Drona, an elite T20 cricket tactical AI."
   EOF
   ollama create drona-v2 -f models/Modelfile_v2

2. Test:
   python src/agent.py --test --model drona-v2
""")
