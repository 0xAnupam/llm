# pip install transformers accelerate bitsandbytes
# !huggingface-cli login

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM ,  BitsAndBytesConfig , LlamaForCausalLM
import accelerate

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

class EarlyExitModel(nn.Module):
    def __init__(self, base_model, exit_threshold=1.0):
        super(EarlyExitModel, self).__init__()
        self.base_model = base_model
        self.exit_threshold = exit_threshold

    def forward(self, input_ids):
        
        hidden_states = self.base_model.model.embed_tokens(input_ids)

        for i, layer in enumerate(self.base_model.model.layers):
            hidden_states = layer(hidden_states)[0]
            logits = self.base_model.lm_head(hidden_states)
            probs = F.softmax(logits[:, -1, :], dim=-1)

            
            max_prob, _ = torch.max(probs, dim=-1)
            if max_prob.item() > self.exit_threshold:
                print(f"Exiting early at layer {i+1}")
                break

        return logits

# Loading tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf" , torch_dtype=torch.float16 , device_map="auto")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf" , quantization_config=quant_config,torch_dtype=torch.float16 , device_map="auto")

early_exit_model = EarlyExitModel(model ,exit_threshold=0.95)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
early_exit_model.to(device)


input_text = "Tell me about Samsung Research Institute Bengalore" #prompt
input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)


def generate_text(input_ids, max_length=100):
    generated_ids = input_ids
    for _ in range(max_length):
        with torch.no_grad():
            logits = early_exit_model(generated_ids)
        
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        # Stop if the model predicts the end of sequence token
        if next_token_id.item() == tokenizer.eos_token_id:
            break
        
    return generated_ids

# Generate text
import time
t1=time.time()
generated_ids = generate_text(input_ids)
t2=time.time()
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("Generated text:", generated_text)
print("Response Time" , t2-t1)
