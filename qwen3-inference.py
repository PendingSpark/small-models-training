from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
print("Loading fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-qwen", device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-qwen")

# Use for inference
content = "What is the capital of France?"
messages = [{"role": "user", "content": content}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

print(f"User: {content}")
inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Extract only the assistant's response
response_start = response.find("<|im_start|>assistant\n")
if response_start != -1:
    assistant_response = response[response_start + len("<|im_start|>assistant\n"):]
    # Remove any trailing tokens
    end_marker = assistant_response.find("<|im_end|>")
    if end_marker != -1:
        assistant_response = assistant_response[:end_marker]
    print(f"Assistant: {assistant_response.strip()}")
else:
    print(f"Assistant: {response}")
