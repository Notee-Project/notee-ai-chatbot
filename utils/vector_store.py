# 안전한 벡터 저장소 관리 클래스 (Windows 파일 잠금 문제 해결)

# 점수 해석:
# 0.0 ~ 0.8: 매우 유사 (거의 정확한 매치)
# 0.8 ~ 1.2: 어느 정도 유사 (관련성 있음)
# 1.2 ~ 2.0: 약간 유사 (부분적 관련성)
# 2.0 이상: 거의 무관함

import os
import sys
import json
import time
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from langchain.schema import Document

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent  # utils 폴더의 상위 폴더
sys.path.append(str(PROJECT_ROOT))

# 최신 패키지 임포트 (Deprecation 경고 해결)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ langchain-huggingface 사용")
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("⚠️ 구버전 langchain_community.embeddings 사용 (업그레이드 권장)")

try:
    from langchain_chroma import Chroma
    print("✅ langchain-chroma 사용")
except ImportError:
    from langchain_community.vectorstores import Chroma
    print("⚠️ 구버전 langchain_community.vectorstores 사용 (업그레이드 권장)")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

try:
    from .document_processing import DocumentProcessor
except ImportError:
    from document_processing import DocumentProcessor

class SafeVectorStoreManager:
    """
    안전한 벡터 저장소 관리 클래스 (Windows 파일 잠금 문제 해결)
    """
    
    def __init__(self, 
                 embedding_model: str = "korean", 
                 vector_db_type: str = "chroma",
                 persist_directory: str = None):
        """
        SafeVectorStoreManager 초기화
        
        Args:
            embedding_model (str): 사용할 임베딩 모델 ("korean", "openai", "multilingual")
            vector_db_type (str): 벡터 DB 종류 ("chroma", "faiss")
            persist_directory (str): 벡터 DB 저장 경로 (None이면 프로젝트 루트/data/vector_db 사용)
        """
        self.embedding_model_type = embedding_model
        self.vector_db_type = vector_db_type
        
        # 경로 설정: 프로젝트 루트 기준
        if persist_directory is None:
            self.persist_directory = PROJECT_ROOT / "data" / "vector_db"
        else:
            self.persist_directory = Path(persist_directory)
        
        print(f"🗂️ 프로젝트 루트: {PROJECT_ROOT}")
        print(f"🗂️ 벡터 DB 경로: {self.persist_directory}")
        
        # 디렉토리 생성
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # 임베딩 모델 초기화
        self.embeddings = self._setup_embedding_model()
        
        # 벡터 저장소 (나중에 초기화)
        self.vector_store = None
        
    def _setup_embedding_model(self):
        """
        임베딩 모델을 설정하는 함수
        
        Returns:
            임베딩 모델 인스턴스
        """
        print(f"임베딩 모델 설정 중: {self.embedding_model_type}")
        
        if self.embedding_model_type == "korean":
            # 한국어에 최적화된 임베딩 모델
            embeddings = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",  # 한국어 특화 모델
                model_kwargs={
                    'device': 'cpu',  # GPU 사용 시 'cuda'로 변경
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True,  # 벡터 정규화
                    'batch_size': 32
                }
            )
            print("✅ 한국어 특화 임베딩 모델 로드 완료: ko-sroberta-multitask")
            
        elif self.embedding_model_type == "openai":
            # OpenAI 임베딩 모델
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            print("✅ OpenAI 임베딩 모델 로드 완료: text-embedding-ada-002")
            
        elif self.embedding_model_type == "multilingual":
            # 다국어 지원 임베딩 모델
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ 다국어 임베딩 모델 로드 완료: paraphrase-multilingual-MiniLM-L12-v2")
            
        else:
            raise ValueError(f"지원하지 않는 임베딩 모델입니다: {self.embedding_model_type}")
            
        return embeddings
    
    def _safe_cleanup_chroma_files(self):
        """
        Chroma 파일들을 안전하게 정리하는 함수
        """
        print("🧹 Chroma 파일들 안전 정리 중...")
        
        # 정리할 파일들
        files_to_clean = [
            self.persist_directory / "chroma.sqlite3",
            self.persist_directory / "chroma.sqlite3-shm",
            self.persist_directory / "chroma.sqlite3-wal"
        ]
        
        # 각 파일에 대해 안전한 삭제 시도
        for file_path in files_to_clean:
            if file_path.exists():
                print(f"🗑️ 파일 삭제 시도: {file_path}")
                
                # 방법 1: SQLite 연결 강제 종료
                if file_path.name == "chroma.sqlite3":
                    try:
                        # SQLite 파일이라면 연결을 강제로 닫기
                        conn = sqlite3.connect(str(file_path))
                        conn.close()
                        time.sleep(0.1)
                    except:
                        pass
                
                # 방법 2: 파일 속성 변경 후 삭제
                try:
                    file_path.chmod(0o777)
                    time.sleep(0.1)
                    file_path.unlink()
                    print(f"✅ 삭제 성공: {file_path.name}")
                except PermissionError:
                    print(f"⚠️ 권한 오류로 삭제 실패: {file_path.name}")
                except FileNotFoundError:
                    print(f"ℹ️ 파일이 이미 없음: {file_path.name}")
                except Exception as e:
                    print(f"⚠️ 삭제 실패: {file_path.name} - {e}")
    
    def _force_close_chroma_connections(self):
        """
        Chroma 관련 연결들을 강제로 닫는 함수
        """
        try:
            # 기존 vector_store가 있다면 정리
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                print("🔌 기존 벡터 저장소 연결 정리 중...")
                
                # Chroma 클라이언트 정리
                if hasattr(self.vector_store, '_client'):
                    try:
                        self.vector_store._client.reset()
                    except:
                        pass
                
                if hasattr(self.vector_store, '_collection'):
                    try:
                        del self.vector_store._collection
                    except:
                        pass
                
                # 객체 삭제
                del self.vector_store
                self.vector_store = None
                
                # 가비지 컬렉션 강제 실행
                import gc
                gc.collect()
                time.sleep(1)
                
                print("✅ 기존 연결 정리 완료")
                
        except Exception as e:
            print(f"⚠️ 연결 정리 중 오류: {e}")
    
    def create_vector_store_safe(self, documents: List[Document]) -> None:
        """
        안전하게 벡터 저장소를 생성하는 함수 (완전 초기화 방식)       
        Args:
            documents (List[Document]): 벡터화할 문서 리스트
        """
        if not documents:
            print("❌ 벡터화할 문서가 없습니다.")
            return
            
        print(f"🚀 안전한 벡터 저장소 생성 중... (문서 수: {len(documents)})")
        print(f"📁 저장 경로: {self.persist_directory}")
        
        # 문서 내용 확인
        print("📄 문서 샘플 확인:")
        for i, doc in enumerate(documents[:3]):
            print(f"  문서 {i+1}: {len(doc.page_content)} 글자")
            print(f"    메타데이터: {doc.metadata}")
            print(f"    내용 미리보기: {doc.page_content[:100]}...")
        
        try:
            if self.vector_db_type == "chroma":
                # 1. 기존 연결 정리
                self._force_close_chroma_connections()
                
                # 2. 전체 벡터 DB 디렉토리 완전 삭제
                import shutil
                if self.persist_directory.exists():
                    print(f"🗑️ 전체 벡터 DB 디렉토리 삭제 중: {self.persist_directory}")
                    
                    # Windows에서 권한 문제 해결
                    def remove_readonly(func, path, _):
                        import stat
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    
                    try:
                        shutil.rmtree(str(self.persist_directory), onerror=remove_readonly)
                        print("✅ 디렉토리 삭제 완료")
                    except Exception as e:
                        print(f"⚠️ 디렉토리 삭제 실패: {e}")
                        # 실패해도 계속 진행
                
                # 3. 디렉토리 다시 생성
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                print("📁 새로운 디렉토리 생성 완료")
                
                # 4. 잠시 대기 (파일 시스템 안정화)
                print("⏳ 파일 시스템 안정화 대기...")
                time.sleep(2)
                
                # 5. 완전히 새로운 벡터 저장소 생성
                collection_name = "school_notices"
                
                print(f"🔧 새 Chroma 벡터 저장소 생성 중... (컬렉션: {collection_name})")
                
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(self.persist_directory),
                    collection_name=collection_name
                )
                
                print(f"✅ Chroma 벡터 저장소 생성 완료: {self.persist_directory}")
                
                # 6. 생성 후 문서 개수 확인
                try:
                    if hasattr(self.vector_store, '_collection'):
                        count = self.vector_store._collection.count()
                        print(f"📊 저장된 문서 개수: {count}")
                    
                    # 검색 테스트
                    test_results = self.vector_store.similarity_search("테스트", k=1)
                    print(f"📊 검색 테스트 결과: {len(test_results)}개 문서 반환")
                    
                except Exception as e:
                    print(f"⚠️ 문서 개수 확인 중 오류: {e}")
                    
            elif self.vector_db_type == "faiss":
                # FAISS 벡터 저장소 생성
                print("🔧 FAISS 벡터 저장소 생성 중...")
                
                # 기존 FAISS 파일들 삭제
                faiss_path = self.persist_directory / "faiss_index"
                if faiss_path.exists():
                    import shutil
                    shutil.rmtree(str(faiss_path))
                    print(f"🗑️ 기존 FAISS 인덱스 삭제: {faiss_path}")
                
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
                # FAISS 인덱스 저장
                faiss_path.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(faiss_path))
                print(f"✅ FAISS 벡터 저장소 생성 완료: {faiss_path}")
                
            else:
                raise ValueError(f"지원하지 않는 벡터 DB입니다: {self.vector_db_type}")
                
        except Exception as e:
            print(f"❌ 벡터 저장소 생성 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 발생 시 대안 방법 시도
            print("🔄 대안 방법으로 재시도...")
            self._try_alternative_creation(documents)
    
    def _try_alternative_creation(self, documents: List[Document]):
        """
        대안 방법으로 벡터 저장소 생성
        """
        try:
            # 임시 디렉토리 사용
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="chroma_temp_"))
            print(f"🔄 임시 디렉토리 사용: {temp_dir}")
            
            # 임시 디렉토리에 벡터 저장소 생성
            temp_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(temp_dir),
                collection_name="temp_collection"
            )
            
            print("✅ 임시 위치에 벡터 저장소 생성 성공")
            
            # 성공하면 원래 위치로 이동 시도
            import shutil
            if self.persist_directory.exists():
                # 기존 디렉토리 백업
                backup_dir = self.persist_directory.parent / f"vector_db_backup_{int(time.time())}"
                shutil.move(str(self.persist_directory), str(backup_dir))
                print(f"📋 기존 디렉토리 백업: {backup_dir}")
            
            # 임시 디렉토리를 원래 위치로 이동
            shutil.move(str(temp_dir), str(self.persist_directory))
            print(f"📁 벡터 저장소를 원래 위치로 이동: {self.persist_directory}")
            
            # 새로운 벡터 저장소 로드
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name="temp_collection"
            )
            
            print("✅ 대안 방법으로 벡터 저장소 생성 완료")
            
        except Exception as e:
            print(f"❌ 대안 방법도 실패: {e}")
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 4, 
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        유사도 검색을 수행하는 함수
        
        Args:
            query (str): 검색 쿼리
            k (int): 반환할 문서 개수
            filter_dict (Optional[Dict]): 메타데이터 필터
            
        Returns:
            List[Document]: 검색 결과 문서 리스트
        """
        if not self.vector_store:
            print("❌ 벡터 저장소가 초기화되지 않았습니다.")
            return []
            
        try:
            print(f"🔍 검색 중: '{query}' (k={k})")
            
            if filter_dict and self.vector_db_type == "chroma":
                # Chroma는 메타데이터 필터링 지원
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                # 기본 유사도 검색
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )
            
            print(f"📊 검색 결과: {len(results)}개 문서 발견")
            return results
            
        except Exception as e:
            print(f"❌ 검색 중 오류 발생: {e}")
            return []
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 4) -> List[tuple]:
        """
        유사도 점수와 함께 검색하는 함수
        
        Args:
            query (str): 검색 쿼리
            k (int): 반환할 문서 개수
            
        Returns:
            List[tuple]: (Document, score) 튜플 리스트
        """
        if not self.vector_store:
            print("❌ 벡터 저장소가 초기화되지 않았습니다.")
            return []
            
        try:
            print(f"🔍 점수 포함 검색 중: '{query}' (k={k})")
            
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            print(f"📊 검색 결과: {len(results)}개 문서 발견")
            return results
            
        except Exception as e:
            print(f"❌ 점수 포함 검색 중 오류 발생: {e}")
            return []
    
    def test_search(self, test_queries: List[str]) -> None:
        """
        테스트 검색을 수행하는 함수
        
        Args:
            test_queries (List[str]): 테스트할 쿼리 리스트
        """
        if not self.vector_store:
            print("❌ 벡터 저장소가 초기화되지 않았습니다.")
            return
            
        print("\n=== 🔍 벡터 저장소 테스트 검색 ===")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 테스트 {i}: '{query}' ---")
            
            try:
                # 유사도 검색 (점수 포함)
                results = self.similarity_search_with_score(query, k=3)
                
                if results:
                    for j, (doc, score) in enumerate(results, 1):
                        print(f"  {j}. [점수: {score:.4f}] {doc.metadata.get('title', 'N/A')}")
                        print(f"     카테고리: {doc.metadata.get('category', 'N/A')}")
                        print(f"     내용: {doc.page_content[:100]}...")
                        print()
                else:
                    print("  ❌ 검색 결과가 없습니다.")
                    
            except Exception as e:
                print(f"  ❌ 검색 중 오류: {e}")
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """
        벡터 저장소 정보를 반환하는 함수
        
        Returns:
            Dict: 벡터 저장소 정보
        """
        info = {
            "embedding_model": self.embedding_model_type,
            "vector_db_type": self.vector_db_type,
            "persist_directory": str(self.persist_directory),
            "is_initialized": self.vector_store is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.vector_store and self.vector_db_type == "chroma":
            try:
                if hasattr(self.vector_store, '_collection'):
                    collection = self.vector_store._collection
                    info["document_count"] = collection.count()
                else:
                    test_results = self.vector_store.similarity_search("테스트", k=1)
                    info["document_count"] = "검색 가능" if test_results else "비어있음"
            except Exception as e:
                info["document_count"] = f"확인 불가: {e}"
        
        return info


def main():
    """
    메인 실행 함수 - 안전한 벡터 저장소 구축
    """
    print("=== 🚀 안전한 벡터 저장소 구축 시작 ===")
    print(f"📁 현재 작업 디렉토리: {Path.cwd()}")
    print(f"📁 프로젝트 루트: {PROJECT_ROOT}")
    
    # 1. 문서 처리기로 문서 로드 및 처리
    print("\n1. 📄 문서 처리 중...")
    
    processor = DocumentProcessor(data_directory=str(PROJECT_ROOT / "data" / "raw"))
    chunks = processor.process_documents()
    
    if not chunks:
        print("❌ 처리된 문서가 없습니다.")
        print("🔧 해결 방법:")
        print(f"   1. {PROJECT_ROOT / 'data' / 'raw'} 디렉토리에 .txt 파일들을 넣어주세요")
        print(f"   2. {PROJECT_ROOT / 'data' / 'raw' / 'metadata.json'} 파일을 생성해주세요")
        return
    
    print(f"✅ 처리된 청크 수: {len(chunks)}")
    
    # 2. 안전한 벡터 저장소 관리자 초기화
    print("\n2. 🔧 안전한 벡터 저장소 관리자 초기화 중...")
    vector_manager = SafeVectorStoreManager(
        embedding_model="korean",  # 한국어 특화 모델 사용
        vector_db_type="chroma",   # Chroma DB 사용
        persist_directory=None     # 기본값: PROJECT_ROOT/data/vector_db
    )
    
    # 3. 안전한 벡터 저장소 생성
    print("\n3. 🏗️ 안전한 벡터 저장소 생성 중...")
    vector_manager.create_vector_store_safe(chunks)
    
    # 4. 벡터 저장소 정보 출력
    if vector_manager.vector_store:
        print("\n4. 📊 벡터 저장소 정보:")
        info = vector_manager.get_vector_store_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # 5. 테스트 검색 수행
        test_queries = [
            "장학금 신청 방법",
            "도서관 운영시간", 
            "해외교환학생 프로그램",
            "취업박람회 일정",
            "학생식당 메뉴"
        ]
        
        vector_manager.test_search(test_queries)
        
        print("\n=== 🎉 안전한 벡터 저장소 구축 완료 ===")
        print(f"📁 벡터 DB 저장 위치: {vector_manager.persist_directory}")
    else:
        print("\n❌ 벡터 저장소 생성에 실패했습니다.")


if __name__ == "__main__":
    main()