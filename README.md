# Notee! - AI 기반 학사정보 및 공지사항 수집 및 제공 챗봇
## notee-ai-chatbot
![KakaoTalk_20250421_220254600](https://github.com/user-attachments/assets/6a87db01-3952-46c9-828c-03a33800263f)
# 프로젝트 구조
```
project_root/
├── .env                  # API 키 등 환경 변수
├── app.py                # 메인 애플리케이션
├── requirements.txt      # 의존성 목록
├── data/                 # 데이터 저장 폴더
│   ├── raw/              # 원본 데이터
│   └── processed/        # 처리된 데이터
├── models/               # 모델 관련 코드
│   ├── embeddings.py     # 임베딩 모델 관련
│   └── llm.py            # LLM 관련
├── utils/                # 유틸리티 함수
│   ├── document_processing.py  # 문서 처리
│   └── vector_store.py   # 벡터 저장소 관련
└── config.py             # 구성 설정
```
# 시스템 구조도
![image](https://github.com/user-attachments/assets/90b1ab99-fc04-492e-be0a-66e00cbc56b2)
# 참조 문서
