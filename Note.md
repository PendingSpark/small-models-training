# Setup notes

1. Install WSL 

2. Setup virtual env 

3. Fresh PyTorch pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
Make sure version is 2.8.0 
```
import torch
print(torch.__version__)
```

4. System restart after PyTorch install

5. NVIDIA driver version: Make sure you're on 535+ drivers.

6. Test
requirements.txt
```python
transformers>=4.52.4
datasets
accelerate>=1.8.1
```

Test files:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print(torch.__version__)

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Use automatic dtype selection
    device_map="auto"    # Automatically map to available devices
)


content = "Hello, how are you?"
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

```
