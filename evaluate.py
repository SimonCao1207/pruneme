from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./merged"  # or the path/model_name you have

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def generate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


input_text = "The future of AI is"
generated_text = generate_text(input_text)
print(generated_text)
