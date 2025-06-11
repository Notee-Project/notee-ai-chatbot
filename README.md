# Notee! - AI 기반 학사정보 및 공지사항 수집 및 제공 챗봇

## notee-ai-chatbot

![KakaoTalk_20250421_220254600](https://github.com/user-attachments/assets/6a87db01-3952-46c9-828c-03a33800263f)

# 컨드리뷰트 가이드라인

-   파이썬 버전 3.9.22
-   requirements.txt 참조 패키지 설치 필수

# 프로젝트 구조

```
yANUs/
├── 📄 .env                           # 환경 변수 (API 키 등)
├── 📄 .gitignore                     # Git 무시 파일
├── 📄 app.py                         # Langchain 설정 테스트용
├── 📄 webhook_server.py              # 웹훅 서버 (메인 애플리케이션)
├── 📄 requirements.txt               # 의존성 목록
├── 📄 README.md                      # 프로젝트 설명서
├── 📄 llm.py                         # LLM 모델 관련
├── 📂 data/                          # 데이터 저장 폴더
│   ├── 📂 raw/                       # 원본 데이터
│   │   ├── 📄 document_01_학사.txt ...
│   │   └── 📄 metadata.json          # 문서 메타데이터
│   │
│   └── 📂 vector_db/                 # 벡터 데이터베이스
│       └── 📁 chroma_db/             # Chroma 벡터 컬렉션 저장소
│
├── 📂 utils/                         # 유틸리티 함수
│   ├── 📄 document_processing.py     # 문서 처리
│   └── 📄 vector_store.py            # 벡터 저장소 관련
```

# 시스템 구조도

![image](https://github.com/user-attachments/assets/90b1ab99-fc04-492e-be0a-66e00cbc56b2)

# 참조 문서
