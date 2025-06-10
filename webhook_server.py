import os
import sys
import time
import asyncio
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 프로젝트 루트 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 기존 RAG 파이프라인 임포트
from llm import RAGPipeline

# 전역 RAG 인스턴스 (서버 시작시 초기화)
rag_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 생명주기 관리
    서버 시작시 RAG 시스템 초기화, 종료시 정리
    """
    global rag_pipeline
    
    print("🚀 웹훅 서버 시작 중...")
    print("=" * 50)
    
    try:
        # RAG 파이프라인 초기화 (기존 llm.py 설정 활용)
        print("🤖 RAG 파이프라인 초기화 중...")
        rag_pipeline = RAGPipeline(
            model_name="gpt-4o-mini",    # 빠른 응답을 위해 3.5 사용
            temperature=0.1-0,               # 일관성 있는 답변
            max_tokens=400,                # 카톡에 적절한 길이
            request_timeout=15,            # 타임아웃 30초
            embedding_model="korean",      # 한국어 특화 임베딩
            vector_db_type="chroma"        # Chroma DB 사용
        )
        
        # RAG 시스템 설정 (기존 벡터 저장소 로드 또는 새로 생성)
        print("🔧 RAG 시스템 설정 중...")
        if not rag_pipeline.setup_rag_system(force_rebuild=False):
            print("❌ RAG 시스템 초기화 실패!")
            raise Exception("RAG 시스템 초기화 실패")
        
        print("✅ RAG 파이프라인 초기화 완료!")
        print("🌐 웹훅 서버 준비 완료!")
        print("=" * 50)
        
        # 서버 실행
        yield
        
    except Exception as e:
        print(f"❌ 서버 초기화 중 오류: {e}")
        raise
    finally:
        # 서버 종료시 정리 작업
        print("🛑 웹훅 서버 종료 중...")
        if rag_pipeline:
            # 필요시 정리 작업 수행
            pass
        print("✅ 서버 종료 완료")

# FastAPI 앱 생성
app = FastAPI(
    title="yANUs 카카오톡 챗봇 웹훅 서버",
    description="경국대학교 공지사항 AI 챗봇 웹훅 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (개발 환경용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델 정의
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

class HealthResponse(BaseModel):
    status: str
    message: str
    rag_initialized: bool

# API 엔드포인트들

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    서버 상태 확인용 루트 엔드포인트
    """
    return {
        "message": "yANUs 카카오톡 챗봇 웹훅 서버",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    서버 헬스 체크 엔드포인트
    """
    global rag_pipeline
    
    is_rag_ready = rag_pipeline is not None and rag_pipeline.qa_chain is not None
    
    return HealthResponse(
        status="healthy" if is_rag_ready else "degraded",
        message="RAG 시스템 정상 작동" if is_rag_ready else "RAG 시스템 초기화 필요",
        rag_initialized=is_rag_ready
    )

@app.post("/webhook", response_model=ChatResponse)
async def webhook_endpoint(request: ChatRequest):
    """
    카카오톡 챗봇 메인 웹훅 엔드포인트
    
    Args:
        request (ChatRequest): { "question": "사용자질문" }
    
    Returns:
        ChatResponse: { "answer": "AI답변" }
    """
    global rag_pipeline
    
    try:
        # RAG 시스템 초기화 확인
        if not rag_pipeline or not rag_pipeline.qa_chain:
            raise HTTPException(
                status_code=503, 
                detail="RAG 시스템이 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."
            )
        
        # 입력 검증
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="질문이 비어있습니다."
            )
        
        user_question = request.question.strip()
        print(f"📝 사용자 질문: '{user_question}'")
        
        # 시작 시간 기록 (성능 모니터링)
        start_time = time.time()
        
        # RAG 파이프라인으로 질문 처리
        result = rag_pipeline.ask_question(user_question)
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        
        # 에러 처리
        if result.get("error", False):
            print(f"❌ RAG 처리 중 오류: {result.get('answer', '알 수 없는 오류')}")
            raise HTTPException(
                status_code=500,
                detail="답변 생성 중 오류가 발생했습니다."
            )
        
        # 답변 추출 및 후처리
        answer = result.get("answer", "죄송합니다. 답변을 생성할 수 없습니다.")
        
        # 카카오톡에 맞게 답변 최적화
        optimized_answer = optimize_answer_for_kakao(answer)
        
        # 성능 로깅
        print(f"✅ 답변 생성 완료 (처리시간: {processing_time:.2f}초)")
        print(f"📊 출처 문서 수: {result.get('total_sources', 0)}")
        
        return ChatResponse(answer=optimized_answer)
        
    except HTTPException:
        # 이미 정의된 HTTP 예외는 그대로 전달
        raise
    except Exception as e:
        print(f"❌ 웹훅 처리 중 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 사용자에게는 친근한 오류 메시지 반환
        raise HTTPException(
            status_code=500,
            detail="일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )

@app.post("/test", response_model=ChatResponse)
async def test_endpoint(request: ChatRequest):
    """
    테스트용 엔드포인트 (개발 중 빠른 테스트용)
    """
    return ChatResponse(
        answer=f"테스트 응답: '{request.question}' 질문을 잘 받았습니다!"
    )

# 유틸리티 함수들

def optimize_answer_for_kakao(answer: str) -> str:
    """
    카카오톡에 최적화된 답변으로 변환
    
    Args:
        answer (str): 원본 답변
    
    Returns:
        str: 카카오톡에 최적화된 답변
    """
    # 1. 길이 제한 (카카오톡 메시지 제한 고려)
    max_length = 500
    if len(answer) > max_length:
        answer = answer[:max_length-3] + "..."
    
    # 2. 줄바꿈 최적화 (카카오톡에서 읽기 좋게)
    answer = answer.replace('\n\n\n', '\n\n')  # 과도한 줄바꿈 제거
    answer = answer.replace('\n\n', '\n')      # 이중 줄바꿈을 단일로
    
    # 3. 특수 문자 정리
    answer = answer.strip()
    
    # 4. 빈 답변 처리
    if not answer:
        answer = "죄송합니다. 해당 정보를 찾을 수 없습니다. 📝\n다른 질문을 시도해보세요!"
    
    return answer

def get_server_info() -> Dict[str, Any]:
    """
    서버 정보 반환
    """
    global rag_pipeline
    
    return {
        "server": "yANUs 웹훅 서버",
        "framework": "FastAPI",
        "rag_model": rag_pipeline.model_name if rag_pipeline else "Not initialized",
        "embedding": rag_pipeline.vector_manager.embedding_model_type if rag_pipeline else "Not initialized",
        "vector_db": rag_pipeline.vector_manager.vector_db_type if rag_pipeline else "Not initialized"
    }

@app.get("/info")
async def server_info():
    """
    서버 정보 조회 엔드포인트
    """
    return get_server_info()

# 개발용 메인 함수
def main():
    """
    개발 서버 실행 함수
    """
    print("🚀 개발 모드로 웹훅 서버 시작...")
    print("📋 사용 가능한 엔드포인트:")
    print("  - GET  /          : 서버 상태")
    print("  - GET  /health    : 헬스 체크")
    print("  - GET  /info      : 서버 정보")
    print("  - POST /webhook   : 메인 웹훅 (카카오톡용)")
    print("  - POST /test      : 테스트용")
    print("  - GET  /docs      : API 문서")
    print()
    print("🌐 ngrok 사용법:")
    print("  1. 다른 터미널에서: ngrok http 8000")
    print("  2. ngrok URL + '/webhook'을 카카오 디벨로퍼스에 등록")
    print()
    
    uvicorn.run(
        "webhook_server:app",  # 모듈:앱 형식
        host="0.0.0.0",        # 모든 인터페이스에서 접근 가능
        port=8000,             # 포트 8000 사용
        reload=True,           # 코드 변경시 자동 재시작
        log_level="info"       # 로그 레벨
    )

if __name__ == "__main__":
    main()