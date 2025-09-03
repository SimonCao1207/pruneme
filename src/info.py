import torch
from transformers import AutoModelForCausalLM, ResNetForImageClassification

model_path = "huzama/FullSkip-3.2-16L"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="auto",
    revision="pico-epoch_0",
)

resnet_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

total_params = sum(p.numel() for p in model.parameters())
resnet_total_params = sum(p.numel() for p in resnet_model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
d_model = model.config.hidden_size
ffn_inner_dimension = model.config.intermediate_size
num_attention_heads = model.config.num_attention_heads
vocab_size = model.config.vocab_size
print("Total parameters:", total_params)
print("ResNet total parameters:", resnet_total_params)
print("Trainable parameters:", trainable_params)
print("Model dimension (d_model):", d_model)
print("Feedforward inner dimension (ffn_inner_dimension):", ffn_inner_dimension)
print("Number of attention heads (num_attention_heads):", num_attention_heads)
print("Vocabulary size (vocab_size):", vocab_size)
