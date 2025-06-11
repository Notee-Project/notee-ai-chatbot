"""
벡터저장소 전용 파이프라인 통합 테스트 스크립트 (테스트 전용)

📋 용도:
- 기존 벡터 저장소의 검색 기능 테스트
- 문서 처리 파이프라인 검증
- 시스템 정상 작동 여부 확인
- 버그 리포트 시 문제 지점 파악

🎯 검증 단계:
1. 데이터 상태 확인 (파일 존재, 크기, 구조)
2. 기존 벡터 저장소 로드 테스트
3. 검색 기능 테스트 (유사도 검색 → 결과 반환)

🚀 실행 방법:
- 기본 테스트: python utils/debug_test.py
- 상세 로그: python utils/debug_test.py --verbose

⚠️ 사전 준비:
벡터 저장소가 없는 경우 먼저 다음 명령을 실행하세요:
python utils/vector_store.py

📊 성공 기준:
- 기존 벡터 저장소 로드 성공
- 검색 기능 정상 작동
"""

import os
import sys
import argparse
import time
from typing import List
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 수정된 임포트
from document_processing import DocumentProcessor
from vector_store import SafeVectorStoreManager as VectorStoreManager

def check_data_directory(verbose=False):
    """데이터 디렉토리와 파일 상태 확인"""
    print("=== 📁 데이터 디렉토리 상태 확인 ===")
    
    data_dir = "../data/raw"
    print(f"데이터 디렉토리: {data_dir}")
    print(f"디렉토리 존재 여부: {os.path.exists(data_dir)}")
    
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"디렉토리 내 파일들: {files}")
        
        # .txt 파일만 필터링
        txt_files = [f for f in files if f.endswith('.txt')]
        print(f"텍스트 파일들: {txt_files}")
        
        # 각 파일 크기 확인
        for txt_file in txt_files:
            file_path = os.path.join(data_dir, txt_file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"  - {txt_file}: {size} bytes")
                
                # 상세 모드에서만 파일 내용 미리보기
                if verbose:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read(200)
                            print(f"    내용 미리보기: {content[:100]}...")
                    except Exception as e:
                        print(f"    파일 읽기 오류: {e}")
        
        # metadata.json 확인
        metadata_path = os.path.join(data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            print("✅ metadata.json 파일 존재")
            if verbose:
                try:
                    import json
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        print(f"메타데이터 문서 수: {len(metadata.get('documents', []))}")
                except Exception as e:
                    print(f"메타데이터 읽기 오류: {e}")
        else:
            print("❌ metadata.json 파일 없음")
            
        return len(txt_files) > 0
    else:
        print("❌ 데이터 디렉토리가 존재하지 않습니다!")
        return False

def test_document_processing(verbose=False):
    """문서 처리 테스트 (선택적)"""
    print("\n=== 📄 문서 처리 테스트 ===")
    
    try:
        processor = DocumentProcessor(data_directory="../data/raw")
        
        # 1. 메타데이터 로드 테스트
        print("1. 메타데이터 로드 테스트...")
        metadata = processor.load_metadata()
        print(f"메타데이터 로드 결과: {len(metadata.get('documents', []))}개 문서")
        
        # 2. 문서 로드 테스트
        print("2. 문서 로드 테스트...")
        documents = processor.load_documents()
        print(f"로드된 문서 수: {len(documents)}")
        
        if documents and verbose:
            print("첫 번째 문서 정보:")
            doc = documents[0]
            print(f"  - 소스: {doc.metadata.get('source', 'N/A')}")
            print(f"  - 내용 길이: {len(doc.page_content)} 글자")
            print(f"  - 내용 미리보기: {doc.page_content[:100]}...")
        
        # 3. 전체 처리 파이프라인 테스트 (상세 모드에서만)
        if verbose:
            print("3. 전체 처리 파이프라인 테스트...")
            chunks = processor.process_documents()
            print(f"생성된 청크 수: {len(chunks)}")
            
            if chunks:
                print("첫 번째 청크 정보:")
                chunk = chunks[0]
                print(f"  - 제목: {chunk.metadata.get('title', 'N/A')}")
                print(f"  - 카테고리: {chunk.metadata.get('category', 'N/A')}")
                print(f"  - 청크 크기: {len(chunk.page_content)} 글자")
        else:
            print("3. 전체 처리 파이프라인 테스트... (--verbose로 실행)")
        
        return True
        
    except Exception as e:
        print(f"문서 처리 중 오류 발생: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False

def check_vector_store_exists():
    """벡터 저장소 존재 여부 확인"""
    print("\n=== 🔍 벡터 저장소 확인 ===")
    
    vector_db_path = "../data/vector_db"
    
    if os.path.exists(vector_db_path):
        print(f"✅ 벡터 저장소 디렉토리 존재: {vector_db_path}")
        
        # 디렉토리 내 파일들 확인
        files = os.listdir(vector_db_path)
        if files:
            print(f"저장된 파일들: {files}")
            
            # Chroma DB 파일들 확인
            chroma_files = [f for f in files if f.startswith('chroma')]
            if chroma_files:
                print(f"Chroma DB 파일들: {chroma_files}")
                return True
            else:
                print("❌ Chroma DB 파일들이 없습니다.")
                return False
        else:
            print("❌ 벡터 저장소 디렉토리가 비어있습니다.")
            return False
    else:
        print(f"❌ 벡터 저장소 디렉토리 없음: {vector_db_path}")
        return False

def test_vector_store_load(verbose=False):
    """기존 벡터 저장소 로드 테스트 (자동으로 컬렉션 찾기)"""
    print("\n=== 🔄 벡터 저장소 로드 테스트 ===")
    
    try:
        vector_manager = VectorStoreManager(
            embedding_model="korean",
            vector_db_type="chroma",
            persist_directory="../data/vector_db"
        )
        
        vector_db_path = "../data/vector_db"
        
        # ChromaDB 클라이언트로 사용 가능한 컬렉션 찾기
        import chromadb
        client = chromadb.PersistentClient(path=vector_db_path)
        collections = client.list_collections()
        
        print(f"🔍 사용 가능한 컬렉션: {[c.name for c in collections]}")
        
        # 문서가 있는 첫 번째 컬렉션 사용
        for collection in collections:
            count = collection.count()
            if count > 0:
                print(f"✅ '{collection.name}' 컬렉션 사용 ({count}개 문서)")
                
                from langchain_chroma import Chroma
                vector_manager.vector_store = Chroma(
                    persist_directory=vector_db_path,
                    embedding_function=vector_manager.embeddings,
                    collection_name=collection.name
                )
                return vector_manager
        
        print("❌ 문서가 있는 컬렉션을 찾을 수 없습니다.")
        return None
        
    except Exception as e:
        print(f"벡터 저장소 로드 실패: {e}")
        return None

def test_search(vector_manager, verbose=False):
    """검색 기능 테스트"""
    print("\n=== 🔍 검색 기능 테스트 ===")
    
    if not vector_manager or not vector_manager.vector_store:
        print("❌ 벡터 저장소가 없어 검색을 테스트할 수 없습니다.")
        return
    
    test_queries = [
        "장학금",
        "도서관", 
        "수강신청",
        "해외교환학생",
        "학생식당"
    ]
    
    # 빠른 모드에서는 처음 3개만 테스트
    if not verbose:
        test_queries = test_queries[:3]
        print("(빠른 테스트 모드: 3개 쿼리만 실행, --verbose로 전체 테스트)")
    
    for query in test_queries:
        print(f"\n🔍 '{query}' 검색 결과:")
        try:
            results = vector_manager.similarity_search_with_score(query, k=2)
            
            if results:
                for i, (doc, score) in enumerate(results, 1):
                    print(f"  {i}. [점수: {score:.4f}]")
                    print(f"     제목: {doc.metadata.get('title', 'N/A')}")
                    print(f"     카테고리: {doc.metadata.get('category', 'N/A')}")
                    if verbose:
                        print(f"     내용: {doc.page_content[:100]}...")
            else:
                print("  검색 결과 없음")
                
        except Exception as e:
            print(f"  검색 중 오류: {e}")

def print_vector_store_creation_guide():
    """벡터 저장소 생성 가이드 출력"""
    print("\n" + "="*60)
    print("🚨 벡터 저장소가 없습니다!")
    print("="*60)
    print("\n다음 단계를 따라 벡터 저장소를 먼저 생성하세요:")
    print("\n1️⃣ 벡터 저장소 생성:")
    print("   python utils/vector_store.py")
    print("\n2️⃣ 생성 완료 후 다시 테스트:")
    print("   python utils/debug_test.py")
    print("\n📝 참고:")
    print("   - vector_store.py는 문서를 처리하고 벡터 저장소를 생성합니다")
    print("   - debug_test.py는 기존 벡터 저장소를 테스트합니다")
    print("   - 역할이 분리되어 더 효율적입니다")
    print("\n" + "="*60)

def validate_prerequisites():
    """사전 요구사항 검사"""
    # 1. 데이터 디렉토리 확인
    if not check_data_directory():
        print("\n❌ 데이터 디렉토리 문제:")
        print("   - data/raw 디렉토리에 .txt 파일들이 있는지 확인하세요")
        print("   - metadata.json 파일도 있는지 확인하세요")
        return False
    
    # 2. 벡터 저장소 확인
    if not check_vector_store_exists():
        print_vector_store_creation_guide()
        return False
    
    return True

def main():
    """메인 실행 함수 (테스트 전용)"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(
        description="벡터저장소 파이프라인 테스트 (기존 저장소 테스트 전용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python utils/debug_test.py          # 기본 테스트
  python utils/debug_test.py --verbose # 상세 로그 출력

사전 준비:
  python utils/vector_store.py        # 벡터 저장소가 없는 경우 먼저 실행
        """
    )
    parser.add_argument("--verbose", action="store_true",
                       help="상세한 로그 및 디버그 정보 출력")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("=== 🧪 벡터저장소 파이프라인 테스트 시작 ===")
    print(f"📅 실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ 모드: {'상세' if args.verbose else '빠른'}")
    
    # 1. 사전 요구사항 검사
    if not validate_prerequisites():
        print("❌ 사전 요구사항을 만족하지 않아 테스트를 중단합니다.")
        return
    
    print("✅ 사전 요구사항 만족")
    
    # 2. 문서 처리 테스트 (선택적)
    if not test_document_processing(verbose=args.verbose):
        print("⚠️ 문서 처리 테스트에서 일부 문제 발견 (계속 진행)")
    
    # 3. 벡터 저장소 로드 테스트
    vector_manager = test_vector_store_load(verbose=args.verbose)
    
    if not vector_manager:
        print("❌ 벡터 저장소 로드에 실패했습니다.")
        print("\n🔧 해결 방법:")
        print("   1. python utils/vector_store.py 를 실행하여 벡터 저장소를 다시 생성하세요")
        print("   2. 데이터 파일들이 올바른지 확인하세요")
        return
    
    # 4. 검색 기능 테스트
    test_search(vector_manager, verbose=args.verbose)
    
    # 5. 실행 시간 출력 및 성능 정보
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\n=== ✅ 벡터저장소 파이프라인 테스트 완료 ===")
    print(f"⏱️ 총 실행 시간: {execution_time:.2f}초")
    
    # 성능 정보
    if execution_time < 5:
        print("🚀 빠른 실행: 벡터 저장소가 효율적으로 로드되었습니다!")
    elif execution_time < 10:
        print("⚡ 정상 실행: 시스템이 안정적으로 동작합니다.")
    else:
        print("🐌 느린 실행: 시스템 성능을 확인해보세요.")

if __name__ == "__main__":
    main()