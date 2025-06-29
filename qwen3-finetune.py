from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

print(torch.__version__)

# model_name = "Qwen/Qwen3-0.6B"
model_name = "Qwen/Qwen3-0.6B-Base" # baseline model to test and finetune

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Use automatic dtype selection
    device_map="auto"    # Automatically map to available devices
)


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
    fp16=False,                       # Disable fp16
    bf16=False,                       # Disable bf16 (not supported on your system)
    disable_tqdm=False,               # ← re-enable tqdm bars
    remove_unused_columns=False,      # <— keep extra columns like prompt_len
)

trainer = Trainer(
    model=model,
    train_dataset=tok_ds,
    data_collator=data_collator,
    args=training_args,
)

trainer.train()
