from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# 下载 tokenizer 和模型，但不做推理
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id)