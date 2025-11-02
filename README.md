# Growlog-AI 🤖

AI 감정 분석 서버 (FastAPI + Transformers)

### 🔧 기술 스택
- FastAPI
- HuggingFace Transformers
- PyTorch (MPS 지원, macOS M1)
- KoBERT or multilingual BERT
- Uvicorn 서버

---

### 🚀 실행 방법
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
Swagger UI: http://localhost:8000/docs