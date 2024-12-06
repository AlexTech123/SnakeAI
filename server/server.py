from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from model.model import Net
import uvicorn
import torch

app = FastAPI(title="SnakeAI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    numbers: List[float]

class Response(BaseModel):
    processed_numbers: List[float]

model = Net(11, 256, 3)
model.load_state_dict(torch.load('model/model.pth', weights_only=True))

@app.post("/process", response_model=Response)
def process_numbers(request: Request):
    try:
        input_numbers = request.numbers
        final_move = [0, 0, 0]
        move = torch.argmax(model(torch.tensor(input_numbers).unsqueeze(0)).squeeze(0)).item()
        final_move[move] = 1
        return Response(processed_numbers=final_move)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


uvicorn.run(app, host="0.0.0.0", port=80)
