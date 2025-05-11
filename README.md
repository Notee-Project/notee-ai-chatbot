# notee-ai-chatbot

Notee! - AI 기반 학사정보 및 공지사항 수집 및 제공 챗봇

project_root/
├── .env # API 키 등 환경 변수
├── app.py # 메인 애플리케이션
├── requirements.txt # 의존성 목록
├── data/ # 데이터 저장 폴더
│ ├── raw/ # 원본 데이터
│ └── processed/ # 처리된 데이터
├── models/ # 모델 관련 코드
│ ├── embeddings.py # 임베딩 모델 관련
│ └── llm.py # LLM 관련
├── utils/ # 유틸리티 함수
│ ├── document_processing.py # 문서 처리
│ └── vector_store.py # 벡터 저장소 관련
└── config.py # 구성 설정
