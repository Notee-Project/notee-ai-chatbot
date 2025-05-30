import os
import sys
sys.path.append('..')

from vector_store import VectorStoreManager

def debug_vector_store():
    """
    벡터 저장소 문제를 디버그하는 함수
    """
    print("=== 벡터 저장소 디버그 테스트 ===")
    
    # 벡터 저장소 관리자 초기화
    vector_manager = VectorStoreManager(
        embedding_model="korean",
        vector_db_type="chroma",
        persist_directory="../data/vector_db"
    )
    
    # 1. 벡터 저장소 상태 확인
    print(f"1. 초기 vector_store 상태: {vector_manager.vector_store}")
    print(f"   persist_directory: {vector_manager.persist_directory}")
    print(f"   디렉토리 존재 여부: {os.path.exists(vector_manager.persist_directory)}")
    
    # 2. 기존 벡터 저장소 로드 시도
    print("\n2. 벡터 저장소 로드 시도...")
    load_success = vector_manager.load_existing_vector_store()
    print(f"   로드 결과: {load_success}")
    print(f"   로드 후 vector_store 상태: {vector_manager.vector_store}")
    print(f"   vector_store 타입: {type(vector_manager.vector_store)}")
    
    # 3. 벡터 저장소가 None인 경우 직접 생성 시도
    if vector_manager.vector_store is None:
        print("\n3. 벡터 저장소가 None이므로 직접 생성 시도...")
        
        try:
            from langchain_community.vectorstores import Chroma
            
            vector_store = Chroma(
                persist_directory=vector_manager.persist_directory,
                embedding_function=vector_manager.embeddings,
                collection_name="school_notices"
            )
            
            # 컬렉션에 데이터가 있는지 확인
            try:
                count = vector_store._collection.count()
                print(f"   벡터 저장소 문서 개수: {count}")
                
                if count > 0:
                    vector_manager.vector_store = vector_store
                    print("   ✅ 벡터 저장소 직접 생성 성공!")
                else:
                    print("   ⚠️ 벡터 저장소가 비어있습니다.")
                    
            except Exception as e:
                print(f"   ❌ 컬렉션 확인 중 오류: {e}")
                
        except Exception as e:
            print(f"   ❌ 직접 생성 중 오류: {e}")
    
    # 4. 최종 테스트
    print(f"\n4. 최종 vector_store 상태: {vector_manager.vector_store}")
    
    if vector_manager.vector_store is not None:
        print("   ✅ 벡터 저장소 준비 완료! 검색 테스트 시작...")
        
        # 간단한 검색 테스트
        test_query = "장학금"
        try:
            results = vector_manager.similarity_search(test_query, k=2)
            print(f"\n🔍 '{test_query}' 검색 결과:")
            
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.metadata.get('title', 'N/A')}")
                print(f"   내용: {doc.page_content[:100]}...")
                
        except Exception as e:
            print(f"   ❌ 검색 중 오류: {e}")
            
    else:
        print("   ❌ 벡터 저장소를 사용할 수 없습니다.")
        
        # 디렉토리 내용 확인
        if os.path.exists(vector_manager.persist_directory):
            files = os.listdir(vector_manager.persist_directory)
            print(f"   벡터 DB 디렉토리 내용: {files}")

def simple_search_test():
    """
    간단한 검색 테스트
    """
    print("\n=== 간단한 검색 테스트 ===")
    
    try:
        # 직접 Chroma 벡터 저장소 로드
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # 임베딩 모델 초기화
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Chroma 벡터 저장소 직접 로드
        vector_store = Chroma(
            persist_directory="../data/vector_db",
            embedding_function=embeddings,
            collection_name="school_notices"
        )
        
        print("✅ 벡터 저장소 직접 로드 성공!")
        
        # 문서 개수 확인
        count = vector_store._collection.count()
        print(f"📊 저장된 문서 개수: {count}")
        
        if count > 0:
            # 테스트 검색들
            test_queries = [
                "장학금 신청",
                "도서관 시간",
                "교환학생",
                "취업박람회"
            ]
            
            for query in test_queries:
                print(f"\n🔍 '{query}' 검색:")
                results = vector_store.similarity_search(query, k=2)
                
                for i, doc in enumerate(results, 1):
                    print(f"  {i}. {doc.metadata.get('title', 'N/A')}")
        else:
            print("⚠️ 벡터 저장소가 비어있습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    debug_vector_store()
    simple_search_test()