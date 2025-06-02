"""
벡터저장소 전용 파이프라인 통합 테스트 스크립트

📋 용도:
- 문서 처리 → 벡터 저장소 생성 → 검색 기능의 전체 흐름 검증
- 각 단계별 오류 진단 및 성능 확인
- 새로운 환경에서 시스템 정상 작동 여부 확인
- 코드 변경 후 전체 시스템 무결성 검증

🎯 검증 단계:
1. 데이터 상태 확인 (파일 존재, 크기, 구조)
2. 문서 처리 파이프라인 (로드 → 메타데이터 → 청크 분할)
3. 벡터 저장소 생성 (임베딩 → Chroma DB 저장)
4. 검색 기능 테스트 (유사도 검색 → 결과 반환)

🚀 실행 시기:
- 개발 환경 최초 설정 시
- 코드 변경 후 회귀 테스트
- 새로운 데이터 추가 후 검증
- 버그 리포트 시 문제 지점 파악

📊 성공 기준:
- 5개 문서 처리 완료
- 5개 청크 생성 성공
- 벡터 저장소 생성 및 검색 가능
"""

import os
import sys
import shutil
from typing import List

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 수정된 임포트
from document_processing import DocumentProcessor
from vector_store import SafeVectorStoreManager as VectorStoreManager

def check_data_directory():
    """데이터 디렉토리와 파일 상태 확인"""
    print("=== 데이터 디렉토리 상태 확인 ===")
    
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
                
                # 파일 내용 미리보기
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
            try:
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    print(f"메타데이터 문서 수: {len(metadata.get('documents', []))}")
            except Exception as e:
                print(f"메타데이터 읽기 오류: {e}")
        else:
            print("❌ metadata.json 파일 없음")
    else:
        print("❌ 데이터 디렉토리가 존재하지 않습니다!")

def test_document_processing():
    """문서 처리 테스트"""
    print("\n=== 문서 처리 테스트 ===")
    
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
        
        if documents:
            print("첫 번째 문서 정보:")
            doc = documents[0]
            print(f"  - 소스: {doc.metadata.get('source', 'N/A')}")
            print(f"  - 내용 길이: {len(doc.page_content)} 글자")
            print(f"  - 내용 미리보기: {doc.page_content[:100]}...")
        
        # 3. 전체 처리 파이프라인 테스트
        print("3. 전체 처리 파이프라인 테스트...")
        chunks = processor.process_documents()
        print(f"생성된 청크 수: {len(chunks)}")
        
        if chunks:
            print("첫 번째 청크 정보:")
            chunk = chunks[0]
            print(f"  - 제목: {chunk.metadata.get('title', 'N/A')}")
            print(f"  - 카테고리: {chunk.metadata.get('category', 'N/A')}")
            print(f"  - 청크 크기: {len(chunk.page_content)} 글자")
            print(f"  - 메타데이터: {chunk.metadata}")
        
        return chunks
        
    except Exception as e:
        print(f"문서 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return []

def clean_vector_store():
    """기존 벡터 저장소 완전 삭제"""
    print("\n=== 벡터 저장소 초기화 ===")
    
    vector_db_path = "../data/vector_db"
    if os.path.exists(vector_db_path):
        try:
            shutil.rmtree(vector_db_path)
            print(f"✅ 기존 벡터 저장소 삭제: {vector_db_path}")
        except Exception as e:
            print(f"❌ 벡터 저장소 삭제 실패: {e}")
    
    # 새로 생성
    os.makedirs(vector_db_path, exist_ok=True)
    print(f"✅ 새 벡터 저장소 디렉토리 생성: {vector_db_path}")

def test_vector_store_creation(chunks):
    """벡터 저장소 생성 테스트"""
    print("\n=== 벡터 저장소 생성 테스트 ===")
    
    if not chunks:
        print("❌ 처리된 청크가 없어 벡터 저장소를 생성할 수 없습니다.")
        return None
    
    try:
        # 벡터 저장소 관리자 초기화
        vector_manager = VectorStoreManager(
            embedding_model="korean",
            vector_db_type="chroma",
            persist_directory="../data/vector_db"
        )
        
        print(f"청크 수: {len(chunks)}")
        print("벡터 저장소 생성 중...")
        
        # 벡터 저장소 생성
        vector_manager.create_vector_store_safe(chunks)
        
        # 생성 확인
        if vector_manager.vector_store is not None:
            print("✅ 벡터 저장소 생성 성공!")
            
            # 문서 개수 확인
            try:
                count = vector_manager.vector_store._collection.count()
                print(f"📊 저장된 문서 개수: {count}")
            except Exception as e:
                print(f"문서 개수 확인 중 오류: {e}")
            
            return vector_manager
        else:
            print("❌ 벡터 저장소 생성 실패")
            return None
            
    except Exception as e:
        print(f"벡터 저장소 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_search(vector_manager):
    """검색 기능 테스트"""
    print("\n=== 검색 기능 테스트 ===")
    
    if not vector_manager or not vector_manager.vector_store:
        print("❌ 벡터 저장소가 없어 검색을 테스트할 수 없습니다.")
        return
    
    test_queries = [
        "장학금",
        "도서관",
        "학식",
        "수강신청",
        "기숙사"
    ]
    
    for query in test_queries:
        print(f"\n🔍 '{query}' 검색 결과:")
        try:
            results = vector_manager.similarity_search_with_score(query, k=2)
            
            if results:
                for i, (doc, score) in enumerate(results, 1):
                    print(f"  {i}. [점수: {score:.4f}]")
                    print(f"     제목: {doc.metadata.get('title', 'N/A')}")
                    print(f"     카테고리: {doc.metadata.get('category', 'N/A')}")
                    print(f"     내용: {doc.page_content[:100]}...")
            else:
                print("  검색 결과 없음")
                
        except Exception as e:
            print(f"  검색 중 오류: {e}")

def create_sample_data():
    """샘플 데이터 생성 (데이터가 없는 경우)"""
    print("\n=== 샘플 데이터 생성 ===")
    
    data_dir = "../data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    # 샘플 텍스트 파일들 생성
    sample_files = {
        "document_01_학사.txt": """
학사 관리 안내

■ 수강신청 안내
- 수강신청 기간: 2025년 2월 10일 ~ 2월 20일
- 신청 방법: 학사정보시스템 접속 후 수강신청 메뉴 이용
- 문의처: 학사팀 (031-123-4567)

■ 장학금 신청 안내
- 성적우수장학금: 직전학기 평점 3.5 이상
- 신청 기간: 매학기 개강 후 2주 이내
- 제출 서류: 장학금신청서, 성적증명서
- 문의처: 학생지원팀 (031-123-4568)

■ 학점 인정 및 편입학 안내
- 편입학 신청 자격: 전문대학 졸업자 또는 4년제 대학 2학년 수료자
- 학점 인정: 동일 계열 과목에 한해 최대 65학점까지 인정
        """,
        
        "document_02_시설.txt": """
캠퍼스 시설 이용 안내

■ 도서관 운영 안내
- 운영시간: 평일 09:00 ~ 22:00, 주말 09:00 ~ 18:00
- 대출 권수: 학부생 5권, 대학원생 10권
- 대출 기간: 학부생 14일, 대학원생 30일
- 연장: 1회 가능 (반납예정일 전일까지)

■ 학생식당 운영 안내
- 운영시간: 
  * 조식: 08:00 ~ 09:00 (평일만)
  * 중식: 11:30 ~ 14:00
  * 석식: 17:00 ~ 19:00
- 식비: 조식 3,000원, 중식 4,000원, 석식 4,500원
- 결제방법: 학생증, 현금, 카드

■ 체육시설 이용 안내
- 체육관 개방시간: 06:00 ~ 22:00
- 헬스장: 평일 06:00 ~ 22:00, 주말 09:00 ~ 18:00
- 수영장: 월/수/금 06:00 ~ 21:00
        """,
        
        "document_03_국제교류.txt": """
국제교류 프로그램 안내

■ 해외교환학생 프로그램
- 신청 자격: 2학년 이상, 평점 3.0 이상
- 파견 대학: 미국, 일본, 중국, 유럽 등 30여개 대학
- 신청 기간: 매년 3월, 9월
- 지원 혜택: 등록금 면제, 항공료 일부 지원

■ 해외인턴십 프로그램  
- 대상: 3학년 이상
- 분야: IT, 경영, 엔지니어링
- 기간: 6개월 ~ 1년
- 혜택: 현지 체재비 지원, 학점 인정

■ 어학연수 프로그램
- 기간: 하계/동계 방학 중 4주
- 국가: 미국, 영국, 호주, 필리핀
- 지원 내용: 수업료, 숙박비 일부 지원
        """
    }
    
    # 파일 생성
    for filename, content in sample_files.items():
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"✅ 생성: {filename}")
    
    # 메타데이터 파일 생성
    metadata = {
        "created_at": "2025-04-20",
        "description": "대학교 공지사항 샘플 데이터",
        "documents": [
            {
                "id": 1,
                "filename": "document_01_학사.txt",
                "title": "학사 관리 안내",
                "category": "학사",
                "date": "2025-04-01",
                "source": "학사팀"
            },
            {
                "id": 2,
                "filename": "document_02_시설.txt", 
                "title": "캠퍼스 시설 이용 안내",
                "category": "시설",
                "date": "2025-04-02",
                "source": "시설팀"
            },
            {
                "id": 3,
                "filename": "document_03_국제교류.txt",
                "title": "국제교류 프로그램 안내", 
                "category": "국제교류",
                "date": "2025-04-03",
                "source": "국제교류팀"
            }
        ]
    }
    
    import json
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("✅ metadata.json 생성 완료")

def main():
    """메인 실행 함수"""
    print("=== 전체 파이프라인 디버그 테스트 시작 ===")
    
    # 1. 데이터 디렉토리 확인
    check_data_directory()
    
    # 데이터가 없으면 샘플 데이터 생성
    if not os.path.exists("../data/raw") or len(os.listdir("../data/raw")) < 2:
        print("\n데이터가 부족하여 샘플 데이터를 생성합니다...")
        create_sample_data()
    
    # 2. 문서 처리 테스트
    chunks = test_document_processing()
    
    if not chunks:
        print("❌ 문서 처리에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    # 3. 벡터 저장소 초기화
    clean_vector_store()
    
    # 4. 벡터 저장소 생성 테스트
    vector_manager = test_vector_store_creation(chunks)
    
    # 5. 검색 기능 테스트
    test_search(vector_manager)
    
    print("\n=== 전체 파이프라인 디버그 테스트 완료 ===")

if __name__ == "__main__":
    main()