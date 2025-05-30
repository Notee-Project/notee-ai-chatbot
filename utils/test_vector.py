from vector_store import VectorStoreManager

# 벡터 저장소 관리자 초기화
vector_manager = VectorStoreManager(
    embedding_model="korean",
    vector_db_type="chroma",
    persist_directory="../data/vector_db"  # 상위 디렉토리의 data 폴더
)

# 기존 벡터 저장소 로드
print("벡터 저장소 로딩 중...")
if vector_manager.load_existing_vector_store():
    print("✅ 벡터 저장소 로드 성공!")
    
    # 테스트 쿼리들
    test_queries = [
        "장학금 신청 방법",
        "도서관 운영시간", 
        "해외교환학생 프로그램",
        "취업박람회 일정",
        "학생식당 메뉴"
    ]
    
    # 기존 test_search 함수 사용
    vector_manager.test_search(test_queries)
    
else:
    print("❌ 벡터 저장소 로드 실패")