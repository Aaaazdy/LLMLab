import os
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

os.environ["HF_HOME"] = "/hpcwork/yw701564/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/hpcwork/yw701564/huggingface_cache"

LLMModels = {
    "1": ("01-ai/Yi-34B-Chat", "Yi"),
    "2": ("openchat/openchat-3.6-8b-20240522", "openchat"),
    "3": ("Qwen/Qwen1.5-32B-Chat", "Qwen"),
    "4": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "DeepSeek"),
}

class LLMHandler:
    def __init__(self, model_key="1"):
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model(model_key)

    def load_model(self, key):
        # clear old cache and memory
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()

        model_id, model_name = LLMModels.get(key, LLMModels["1"])
        print(f"\n[INFO] Loading model [{model_name}] ...")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        ).to(self.device)

        self.model_name = model_name
        print(f"[INFO] Model [{self.model_name}] loaded successfully.")

    def stream_query(self, prompt, max_length=256):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = (inputs["input_ids"] != self.tokenizer.pad_token_id).long()

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        response = ""
        for token in streamer:
            print(token, end='', flush=True)
            response += token
        print()
        return response.strip()

def print_model_menu():
    print("\nAvailable models:")
    for key, (_, name) in LLMModels.items():
        print(f"  {key}. {name}")
    print("Type in the number of the model (e.g. 1)")

if __name__ == "__main__":
    llm = LLMHandler(model_key="1") # defult using model 1

    print(f"\nVirtual Assistant (Model: [{llm.model_name}]) ready. Type a question ('switch' to change model, 'exit' to quit).")

    while True:
        try:
            prompt = input("\nQuestion: ").strip()

            if prompt.lower() in ["exit", "quit"]:
                print("Bye!")
                break

            if prompt.lower() == "switch":
                print_model_menu()
                model_choice = input("Choose model: ").strip()
                if model_choice in LLMModels:
                    llm.load_model(model_choice)
                    print(f"[INFO] Switched to model: [{llm.model_name}]")
                else:
                    print("[WARN] Invalid choice. Keeping current model.")
                continue

            print(f"\n{llm.model_name}: ", end='', flush=True)
            llm.stream_query(prompt)

        except KeyboardInterrupt:
            print("\n[Interrupt]")
            break
        except Exception as e:
            print(f"[ERROR]: {e}")
