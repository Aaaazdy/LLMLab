from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from llm import LLMHandler, LLMModels

app = FastAPI()
llm = LLMHandler(model_key="2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "LLM server is running."}

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_length", 256)

    def generate_stream():
        for token in llm.stream_query(prompt, max_tokens):
            yield token

    return StreamingResponse(generate_stream(), media_type="text/plain")

@app.post("/switch_model")
async def switch_model(request: Request):
    data = await request.json()
    model_key = data.get("model_key", "1")
    if model_key in LLMModels:
        llm.load_model(model_key)
        return {"status": "success", "model": llm.model_name}
    else:
        return {"status": "error", "message": "Invalid model key"}
