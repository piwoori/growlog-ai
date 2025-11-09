from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
import traceback

# ── 모델 설정
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
VERSION = "v0.3"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

app = FastAPI(title="Growlog AI Sentiment & Quantum API", version=VERSION)

# ── 입력/출력 모델 정의
class TextInput(BaseModel):
    text: str

class ProbInput(BaseModel):
    positive: float
    neutral: float
    negative: float

class PhaseInput(BaseModel):
    positive: float = 0.0
    neutral: float = 0.0
    negative: float = 0.0

class OmegaInput(BaseModel):
    positive: float = 1.0
    neutral: float = 1.3
    negative: float = 1.7

class QuantumSimRequest(BaseModel):
    text: Optional[str] = None
    probs: Optional[ProbInput] = None
    duration: float = 30.0
    dt: float = 0.05
    coherence: float = Field(0.75, ge=0.0, le=1.0)
    phases: PhaseInput = PhaseInput()
    omegas: OmegaInput = OmegaInput()

class ComponentOut(BaseModel):
    label: str
    probability: float
    amplitude: float
    phase: float
    omega: float

class QuantumSimResponse(BaseModel):
    time: List[float]
    I_total: List[float]
    components: List[ComponentOut]
    summary: Dict[str, float]

# ── 헬스체크
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "version": VERSION, "device": str(DEVICE)}

# ── 감정 분석 엔드포인트
@app.post("/analyze")
def analyze_sentiment(input: TextInput):
    enc = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits

    probs = F.softmax(logits, dim=-1)[0].cpu()  # [neg, neu, pos]
    p_neg, p_neu, p_pos = map(float, probs)

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
        "device": str(DEVICE),
    }

# ── Quantum 감정 파동 시뮬레이션
# NOTE: 초기 디버깅을 위해 response_model 검증을 잠시 끕니다. (에러 detail 바로 보기 위함)
#       문제 없이 동작 확인 후 아래 데코레이터를 response_model=QuantumSimResponse 로 바꿔도 됩니다.
@app.post("/quantum/simulate")  # , response_model=QuantumSimResponse
def quantum_simulate(req: QuantumSimRequest):
    try:
        # 1) 입력 소스 결정
        if req.probs is None and (req.text is None or not str(req.text).strip()):
            raise ValueError("Either text or probs must be provided.")

        if req.probs is None:
            # 텍스트 → 모델 확률
            enc = tokenizer(
                req.text, return_tensors="pt", truncation=True, padding=True, max_length=256
            ).to(DEVICE)
            with torch.no_grad():
                logits = model(**enc).logits
            probs_model = F.softmax(logits, dim=-1)[0].cpu().numpy()  # [neg, neu, pos]
            p_neg, p_neu, p_pos = float(probs_model[0]), float(probs_model[1]), float(probs_model[2])
        else:
            # 확률 직접 입력
            p_pos = float(req.probs.positive)
            p_neu = float(req.probs.neutral)
            p_neg = float(req.probs.negative)

        # 2) 정규화 및 검증
        probs = np.array([p_pos, p_neu, p_neg], dtype=float)
        s = float(probs.sum())
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"Invalid probabilities: {probs.tolist()}")
        probs = probs / s  # 합 1

        # 3) 파라미터 (전부 float 강제)
        labels = ["positive", "neutral", "negative"]
        phases = np.array(
            [float(req.phases.positive), float(req.phases.neutral), float(req.phases.negative)],
            dtype=float,
        )
        omegas = np.array(
            [float(req.omegas.positive), float(req.omegas.neutral), float(req.omegas.negative)],
            dtype=float,
        )
        amps = np.sqrt(probs.astype(float))
        c = float(req.coherence)
        if not (0.0 <= c <= 1.0):
            raise ValueError(f"coherence must be in [0,1], got {c}")

        duration = float(req.duration)
        dt = float(req.dt)
        if dt <= 0 or duration <= 0:
            raise ValueError(f"duration/dt must be > 0, got duration={duration}, dt={dt}")

        # 4) 시간축
        t = np.arange(0.0, duration + 1e-12, dt, dtype=float)
        base = float(np.sum(amps**2))
        I_total = np.full(t.shape, base, dtype=float)

        # 5) 간섭항
        for i in range(3):
            for j in range(i + 1, 3):
                d_omega = float(omegas[i] - omegas[j])
                d_phase = float(phases[i] - phases[j])
                I_total += 2.0 * c * float(amps[i] * amps[j]) * np.cos(d_omega * t + d_phase)

        # 6) 요약
        summary = {
            "mean": float(np.mean(I_total)),
            "max": float(np.max(I_total)),
            "min": float(np.min(I_total)),
            "ptp": float(np.ptp(I_total)),
        }

        components = [
            {
                "label": labels[k],
                "probability": float(probs[k]),
                "amplitude": float(amps[k]),
                "phase": float(phases[k]),
                "omega": float(omegas[k]),
            }
            for k in range(3)
        ]

        return {
            "time": t.tolist(),
            "I_total": I_total.tolist(),
            "components": components,
            "summary": summary,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))