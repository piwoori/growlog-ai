from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List
import traceback
import os
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ADVICE_MODEL = os.getenv("ADVICE_MODEL", "gpt-4.1-mini")


# ── 모델 설정
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
VERSION = "v0.3"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

app = FastAPI(title="Growlog AI Sentiment & Quantum API", version=VERSION)

# ── 입력/출력 모델 정의
class AdviceRequest(BaseModel):
    text: str           # 오늘 감정 메모 or 회고 내용
    emoji: str | None = None  # 선택: 감정 이모지

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

def build_fallback_advice(text: str, emoji: str | None = None) -> str:
    """OpenAI 호출이 실패했을 때 사용하는 간단한 규칙 기반 조언."""
    t = text.strip()

    negative_words = ["피곤", "힘들", "우울", "불안", "걱정", "짜증", "지쳤", "버겁", "무기력"]
    stress_words = ["과제", "숙제", "시험", "공부", "일이 많", "마감", "데드라인"]
    body_words = ["두통", "머리 아프", "어지럽", "몸살", "감기"]

    is_negative = any(w in t for w in negative_words)
    is_stress   = any(w in t for w in stress_words)
    is_body     = any(w in t for w in body_words)

    bad_emojis = ["😢", "😭", "😞", "😔", "😡", "😤", "😫", "😩", "😴", "🥲"]
    good_emojis = ["😄", "🙂", "🤩", "😊", "😆", "😁"]

    if emoji in bad_emojis:
        is_negative = True
    if emoji in good_emojis and not is_negative:
        is_negative = False

    # 케이스 분기
    if is_body:
        return (
            "오늘은 몸이 조금 무거운 날 같아요. 따뜻한 물 많이 마시고, "
            "무리하지 말고 일찍 쉬어 주면 좋겠어요."
        )

    if is_negative and is_stress:
        return (
            "요즘 할 일이 많아서 마음이 꽤 지친 상태인 것 같아요. "
            "오늘 해야 할 것 중에서 꼭 중요한 것 한두 개만 정리하고, "
            "잠깐 산책이나 스트레칭으로 머리를 식혀보면 어떨까요?"
        )

    if is_negative:
        return (
            "기분이 조금 아래쪽으로 내려가 있는 하루 같아요. "
            "스스로를 몰아붙이기보다는, 좋아하는 음악을 틀어놓고 "
            "짧게라도 휴식 시간을 만들어 보는 건 어떨까요?"
        )

    if is_stress:
        return (
            "해야 할 일들이 머릿속에서 빙글빙글 도는 느낌일 수 있어요. "
            "간단한 할 일 목록을 적어두고, 가장 작은 것 하나부터 "
            "차근차근 정리해보면 마음이 훨씬 가벼워질 거예요."
        )

    # 기본(무난한 날)
    return (
        "오늘 하루를 이렇게 기록한 것만으로도 이미 잘 하고 있어요. "
        "지금 느낌을 잠깐 더 돌아보고, 남은 시간에는 나를 위한 작은 보상을 준비해보면 어떨까요?"
    )


@app.post("/advice")
async def generate_advice(payload: AdviceRequest):
    """
    오늘 감정/메모를 기반으로 짧은 자기관리 피드백을 생성하는 엔드포인트.
    - input: text(필수), emoji(선택)
    - output: 한글 조언 2~3문장
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text는 필수입니다.")

    user_text = payload.text.strip()

    # OpenAI를 먼저 시도하고, 실패하면 로컬 규칙 기반 문구로 대체
    try:
        emoji_part = (
            f"사용자가 선택한 감정 이모지는 '{payload.emoji}'입니다.\n"
            if payload.emoji
            else ""
        )

        system_prompt = """
너는 사용자의 하루 감정과 메모를 바탕으로, 짧은 자기관리 피드백을 제안하는 코치야.

규칙:
- 말투는 부드럽고 편안하게, 반말/존댓말 혼용 없이 "~요"체로 통일.
- 2~3문장 정도로 짧게.
- 너무 거창한 목표 말고, 오늘 당장 할 수 있는 가벼운 행동을 제안해줘.
- 사용자를 평가하거나 비난하는 표현은 절대 사용하지 마.
- 출력 형식은 줄바꿈 포함 자유롭게, 마크다운 기호는 쓰지 말 것.
"""

        completion = openai_client.responses.create(
            model=ADVICE_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""{emoji_part}다음은 사용자의 오늘 감정 메모입니다:

\"\"\"{user_text}\"\"\"""",
                },
            ],
            max_output_tokens=120,
        )

        advice_text = completion.output[0].content[0].text.strip()

        return {
            "advice": advice_text,
            "model": ADVICE_MODEL,
            "source": "openai",
        }

    except Exception as e:
        # 크레딧/네트워크/기타 오류 → 로컬 조언으로 대체
        print("❌ /advice 생성 오류 (fallback 사용):", e)
        fallback = build_fallback_advice(user_text, payload.emoji)

        return {
            "advice": fallback,
            "model": "local-fallback",
            "source": "fallback",
            "note": "OpenAI API 쿼터/네트워크 오류로 로컬 규칙 기반 조언을 반환했어요.",
        }

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