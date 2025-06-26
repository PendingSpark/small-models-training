# Setup guide for Small Models Training
## Install Windows
<TBD>

## Setup basic tools
1. VSCode https://code.visualstudio.com/docs/?dv=win64user

2. Enable WSL https://learn.microsoft.com/en-us/windows/wsl/install
A. Make sure to enable "Windows subsystem for Linux" (Go to Turn Windows Features On or Off)
B. We would just use the default one when running "wsl --install"
C. When prompt, create user: firstfloortech , password: <your password here>

3. Install LMStudio (Optional) - Or install Ollama + Openwebui
4. Install Git (Optional)
https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-git
```
sudo apt-get install git

```
5. Setup up python - version 3.12.3 (default from Ubuntu)

6. Setup virtual env
```
sudo apt update
sudo apt install python3-venv
python3 -m venv finetune-env
source finetune-env/bin/activate
```
7. Install Pytorch
```
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
Check version:
Run the following python script
```
import torch
print(torch.__version__)
```
I got 2.8.0.dev20250625+cu128 on June 25

8. Open Nvidia Control Panel and check version of the driver
(572.84)

9. Install claude code (Optional)
```
sudo apt install nodejs
sudo apt install npm
sudo npm install -g @anthropic-ai/claude-code
```

## Test the setup for finetuning
1. Create requirements.txt file
```
transformers>=4.52.4
datasets
accelerate>=1.8.1
```
2. Create simple test to make sure we would load the model and use it.
test-finetune.py
```
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
it should output something without error
