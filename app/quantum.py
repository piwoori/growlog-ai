from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import numpy as np

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