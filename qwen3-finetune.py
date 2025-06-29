from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

print(torch.__version__)

# model_name = "Qwen/Qwen3-0.6B"
model_name = "Qwen/Qwen3-0.6B-Base" # baseline model to test and finetune
print(f"Model name: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Use automatic dtype selection
    device_map="auto"    # Automatically map to available devices
)

# Prepare a user prompt for the baseline model
content = "how are you"
messages = [{"role": "user", "content": content}]
# Use Qwen's chat template to format the prompt (no fine-tuning applied yet)
# By setting enable_thinking equal to False, it add <think>\n</think> token to tell the model that the thinking is end.
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("prompt text is: %s" % prompt_text)

inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
# Generate a response
output_ids = model.generate(**inputs, max_new_tokens=50)
baseline_answer = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("models output: %s" % baseline_answer)
print("========= End model test run =========")

# Training data = The actual samples/examples used to train the model (in this case: 500 samples)
# Training steps = The number of times the model weights are updated during training
# 1. You have 500 training samples (the actual data)
# 2. With a batch size of 2, the model processes 2 samples at a time
# 3. To go through all 500 samples once: 500 ÷ 2 = 250 steps
# 4. Since the training runs for 1 epoch (one complete pass through all data), the total is 250 training steps
# So in each training step:
# - The model processes 2 samples (batch size)
# - Calculates the loss
# - Updates the model weights via backpropagation
# Think of it like reading a 500-page book 2 pages at a time - you'd need 250 reading sessions (steps) to finish
# the book once (1 epoch).

# 1  ── Load a tiny slice of Alpaca for the demo
# instruction (string):
# Identify the odd one out.
# input (string):
# Twitter, Instagram, Telegram
# output (string):
# Telegram
# text (String):
# Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# ### Instruction:
# Identify the odd one out.
# ### Input:
# Twitter, Instagram, Telegram
# ### Response:
# Telegram
raw_ds = load_dataset("tatsu-lab/alpaca", split="train[:500]")

# 2  ── Build <|im_start|>user … <|im_end|> <|im_start|>assistant … <|im_end|> strings
def format_example(example):
    if example["input"]:
        user_text = f"{example['instruction']}\n{example['input']}"
    else:
        user_text = example["instruction"]

    prompt = (
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_text = prompt + example["output"] + "<|im_end|>"

    # We’ll store the prompt length now so the collator can mask it cheaply
    prompt_len = len(tokenizer(prompt)["input_ids"])
    return {"prompt": prompt, "full_text": full_text, "prompt_len": prompt_len}

chat_ds = raw_ds.map(format_example, remove_columns=raw_ds.column_names)

# 3  ── Tokenize once and carry the prompt length through
def tokenize_func(ex):
    tok = tokenizer(ex["full_text"], truncation=True)
    tok["prompt_len"] = ex["prompt_len"]          # keep for masking later
    return tok

tok_ds = chat_ds.map(tokenize_func, remove_columns=chat_ds.column_names)

print("SAMPLE DATA FOR TRAINING")
for raw_example, chat_example, tok_example in zip(raw_ds, chat_ds, tok_ds):
    print("raw_example:")
    print("instruction: %s" % raw_example["instruction"])
    print("input: %s" % raw_example["input"])
    print("output: %s" % raw_example["output"])
    print()
    print("chat_example:")
    print("prompt: %s" % chat_example["prompt"])
    print("full_text: %s" % chat_example["full_text"])
    print()
    print("tok_example:")
    print("input_ids: %s" % tok_example["input_ids"])
    print("prompt_len: %s" % tok_example["prompt_len"])
    print()
    break
print("========= End sample data output =========")

# Creating a Data Collator for SFT: We need to mask out the prompt portion in the labels, so the loss is only computed on the response part.
# TODO: bring this to a notebook
class SFTDataCollator:
    def __call__(self, batch):
        # Convert list of tokenized samples to padded tensor batch
        input_ids_list = [torch.tensor(b["input_ids"]) for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        # Attention mask: 1 for real tokens, 0 for pad
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        # Create labels (copy of input_ids)
        labels = input_ids.clone()
        # Mask out prompt part (all tokens up to and including "### Response:\n")
        for i, b in enumerate(batch):
            prompt_len = b["prompt_len"]  # length of prompt in tokens
            labels[i, :prompt_len] = -100  # ignore prompt tokens in loss
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

data_collator = SFTDataCollator()

training_args = TrainingArguments(
    output_dir="qwen_sft_demo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,    # use batch size 2 per GPU
    gradient_accumulation_steps=1,    # no grad accumulation (since batch 2 is fine)
    logging_steps=20,                 # log every 20 steps
    save_steps=0,                     # no checkpoints (not needed for demo)
    report_to=[],                     # no W&B or HF logging
    bf16=True,                        # Qwen3 is using bf16 training
    disable_tqdm=False,               # ← re-enable tqdm bars
    remove_unused_columns=False,      # <— keep extra columns like prompt_len
)

trainer = Trainer(
    model=model,
    train_dataset=tok_ds,
    data_collator=data_collator,
    args=training_args,
)

print("========= Start training run =========")
trainer.train()
print("========= End training run =========")


# Sample output of trainer run
# {'loss': 1.7251, 'grad_norm': 20.75, 'learning_rate': 4.6200000000000005e-05, 'epoch': 0.08}
# {'loss': 1.4817, 'grad_norm': 22.5, 'learning_rate': 4.22e-05, 'epoch': 0.16}
# {'loss': 1.1211, 'grad_norm': 28.625, 'learning_rate': 3.82e-05, 'epoch': 0.24}
# {'loss': 1.0743, 'grad_norm': 32.75, 'learning_rate': 3.4200000000000005e-05, 'epoch': 0.32}
# {'loss': 1.2086, 'grad_norm': 17.375, 'learning_rate': 3.02e-05, 'epoch': 0.4}
# {'loss': 1.0484, 'grad_norm': 11.375, 'learning_rate': 2.6200000000000003e-05, 'epoch': 0.48}
# {'loss': 1.2362, 'grad_norm': 37.25, 'learning_rate': 2.22e-05, 'epoch': 0.56}
# {'loss': 1.0143, 'grad_norm': 37.5, 'learning_rate': 1.8200000000000002e-05, 'epoch': 0.64}
# {'loss': 0.8783, 'grad_norm': 19.125, 'learning_rate': 1.42e-05, 'epoch': 0.72}
# {'loss': 0.9362, 'grad_norm': 14.375, 'learning_rate': 1.02e-05, 'epoch': 0.8}
# {'loss': 0.9549, 'grad_norm': 17.125, 'learning_rate': 6.2e-06, 'epoch': 0.88}
# {'loss': 1.1153, 'grad_norm': 22.5, 'learning_rate': 2.2e-06, 'epoch': 0.96}
# {'train_runtime': 30.4044, 'train_samples_per_second': 16.445, 'train_steps_per_second': 8.222, 'train_loss': 1.1492959327697754, 'epoch': 1.0}
# 100%|█████████████████████████████████████████████████████████████████████████████████| 250/250 [00:30<00:00,  8.22it/s]

# Understanding Training Loss
# In the realm of machine learning, training loss is a fundamental metric that quantifies how poorly your model is performing on the data it is being trained on. Think of it as an "error score." The lower the loss, the better your model is at predicting the correct outputs for the given inputs in the training set.
# The core objective of training a machine learning model is to minimize this loss value. The model iteratively adjusts its internal parameters (weights and biases) to reduce this error. This process of adjustment is guided by an optimizer algorithm, and the learning_rate you see in your logs controls how large those adjustments are.

# Analyzing Your 20-Step Run
# Here's a step-by-step interpretation of the log you've provided:
# Initial Decrease (Steps 1-4 & 6, 8, 9): You see the loss generally going down, from 1.7251 to 1.0743, and then again to 1.0484, 1.0143, and a low of 0.8783. This is the ideal scenario. It indicates that your model is learning from the training data and getting better at its task. The adjustments it's making are, for the most part, moving it in the right direction to minimize the error.
# The "Down and Then Up" Fluctuation: You've correctly identified that the loss doesn't decrease monotonically (continuously). It goes down, then pops back up. For instance:
# It drops to 1.0743 and then rises to 1.2086.
# It drops to 1.0484 and then rises to 1.2362.
# After hitting a low of 0.8783, it consistently starts to rise in the subsequent steps (0.9362, 0.9549, 1.1153).

# What Does This Fluctuation Mean?
# This "down and then up" behavior, especially within such a short run, is quite common and can be attributed to a few factors, often related to the learning_rate:
# Learning Rate Schedule: Your learning_rate is decreasing over time, from 4.62e-05 down to 2.2e-06. 
# This is a very common and effective technique called learning rate scheduling. 
# The idea is to take large steps at the beginning of training to make rapid progress and then smaller steps as you get closer to the optimal solution to fine-tune the model.
# Bouncing Around the Minimum: The fluctuations you see are characteristic of the optimization process. Imagine a U-shaped valley where the bottom represents the lowest possible loss.
# Initially, with a higher learning rate, the model takes larger steps and quickly moves down the slope of this valley.
# As it approaches the bottom, a learning rate that is still a bit too high can cause the model to "overshoot" the minimum and land on the other side, resulting in a higher loss. The next step might correct this, but it can lead to this bouncing effect.
# The Final Upward Trend: The rise in loss towards the end of this particular epoch (0.8783 -> 1.1153) is noteworthy. While some fluctuation is normal, a sustained increase can be a sign of a few things:

# Learning Rate is Too Low: Paradoxically, if the learning rate becomes too small too quickly, the model might not have enough momentum to escape a small local minimum or a "saddle point" (a flat region that isn't the true minimum) and can start to wander, leading to an increase in loss.
# Approaching Overfitting: While it's very early in the training (only one epoch), this can be an early sign that the model is starting to fit the training data too well, including its noise. This is known as overfitting. However, you'd typically confirm this by looking at a separate "validation loss" (the model's performance on data it hasn't seen during training). If the training loss goes down but the validation loss goes up, you have a clear case of overfitting.

# Other Metrics in Your Log:
# grad_norm (Gradient Norm): This represents the "steepness" of the loss landscape at the current point. A high grad_norm suggests the model is in a steep region and can learn quickly, while a lower value indicates a flatter area. The fluctuations you see here are also normal.
# epoch: This tells you how many times the model has seen the entire training dataset. You've completed one full epoch.
# train_loss (at the end): The value 1.149... is the average loss over the entire training run for this epoch.

# Prepare a user prompt for the baseline model after training
content = "how are you"
messages = [{"role": "user", "content": content}]
# Use Qwen's chat template to format the prompt (no fine-tuning applied yet)
# By setting enable_thinking equal to False, it add <think>\n</think> token to tell the model that the thinking is end.
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("prompt text is: %s" % prompt_text)

inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
# Generate a response
output_ids = model.generate(**inputs, max_new_tokens=50)
baseline_answer = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("models output: %s" % baseline_answer)
