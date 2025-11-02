from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ── 변경: 3클래스 멀티링구얼 감정 모델
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
VERSION = "v0.2"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

app = FastAPI(title="Growlog AI Sentiment API", version=VERSION)

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": VERSION}

@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    # 트윗 모델은 전처리 특성이 있지만, 기본 토크나이저로도 충분히 동작
    enc = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)[0].cpu()  # order: [negative, neutral, positive]

    p_neg = float(probs[0])
    p_neu = float(probs[1])
    p_pos = float(probs[2])

    label = max(
        {"positive": p_pos, "neutral": p_neu, "negative": p_neg},
        key=lambda k: {"positive": p_pos, "neutral": p_neu, "negative": p_neg}[k]
    )

    return {
        "text": input.text,
        "positive": round(p_pos, 3),
        "neutral": round(p_neu, 3),
        "negative": round(p_neg, 3),
        "label": label,
        "device": str(DEVICE)
    }
