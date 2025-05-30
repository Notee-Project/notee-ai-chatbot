# 벡터 저장소 관련
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings

# 문서 처리기 임포트 (이전에 만든 것)
from document_processing import DocumentProcessor

class VectorStoreManager:
    """
    벡터 저장소 관리 클래스
    임베딩 모델 선택, 벡터 DB 구축, 검색 기능 제공
    """
    
    def __init__(self, 
                 embedding_model: str = "korean", 
                 vector_db_type: str = "chroma",
                 persist_directory: str = "data/vector_db"):
        """
        VectorStoreManager 초기화
        
        Args:
            embedding_model (str): 사용할 임베딩 모델 ("korean", "openai", "multilingual")
            vector_db_type (str): 벡터 DB 종류 ("chroma", "faiss")
            persist_directory (str): 벡터 DB 저장 경로
        """
        self.embedding_model_type = embedding_model
        self.vector_db_type = vector_db_type
        self.persist_directory = persist_directory
        
        # 디렉토리 생성
        os.makedirs(persist_directory, exist_ok=True)
        
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
            print("한국어 특화 임베딩 모델 로드 완료: ko-sroberta-multitask")
            
        elif self.embedding_model_type == "openai":
            # OpenAI 임베딩 모델
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            print("OpenAI 임베딩 모델 로드 완료: text-embedding-ada-002")
            
        elif self.embedding_model_type == "multilingual":
            # 다국어 지원 임베딩 모델
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("다국어 임베딩 모델 로드 완료: paraphrase-multilingual-MiniLM-L12-v2")
            
        else:
            raise ValueError(f"지원하지 않는 임베딩 모델입니다: {self.embedding_model_type}")
            
        return embeddings
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        문서들로부터 벡터 저장소를 생성하는 함수
        
        Args:
            documents (List[Document]): 벡터화할 문서 리스트
        """
        if not documents:
            print("벡터화할 문서가 없습니다.")
            return
            
        print(f"벡터 저장소 생성 중... (문서 수: {len(documents)})")
        
        try:
            if self.vector_db_type == "chroma":
                # Chroma 벡터 저장소 생성
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name="school_notices"
                )
                # 데이터 영구 저장
                self.vector_store.persist()
                print(f"Chroma 벡터 저장소 생성 완료: {self.persist_directory}")
                
                # 생성 후 문서 개수 확인
                try:
                    count = self.vector_store._collection.count()
                    print(f"저장된 문서 개수: {count}")
                except Exception as e:
                    print(f"문서 개수 확인 중 오류: {e}")
                
            elif self.vector_db_type == "faiss":
                # FAISS 벡터 저장소 생성
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                # FAISS 인덱스 저장
                faiss_path = os.path.join(self.persist_directory, "faiss_index")
                self.vector_store.save_local(faiss_path)
                print(f"FAISS 벡터 저장소 생성 완료: {faiss_path}")
                
            else:
                raise ValueError(f"지원하지 않는 벡터 DB입니다: {self.vector_db_type}")
                
        except Exception as e:
            print(f"벡터 저장소 생성 중 오류 발생: {e}")
            raise
    
    def load_existing_vector_store(self) -> bool:
        """
        기존에 저장된 벡터 저장소를 로드하는 함수
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if self.vector_db_type == "chroma":
                # Chroma 벡터 저장소 로드
                if os.path.exists(self.persist_directory):
                    self.vector_store = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings,
                        collection_name="school_notices"
                    )
                    
                    # 문서 개수 확인
                    try:
                        count = self.vector_store._collection.count()
                        print(f"기존 Chroma 벡터 저장소 로드 완료 (문서 수: {count})")
                        
                        if count == 0:
                            print("⚠️ 벡터 저장소가 비어있습니다.")
                            return False  # 비어있으면 False 반환하여 새로 생성하도록 함
                        
                        return True
                    except Exception as e:
                        print(f"문서 개수 확인 중 오류: {e}")
                        return False
                    
            elif self.vector_db_type == "faiss":
                # FAISS 벡터 저장소 로드
                faiss_path = os.path.join(self.persist_directory, "faiss_index")
                if os.path.exists(faiss_path):
                    self.vector_store = FAISS.load_local(
                        faiss_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    print(f"기존 FAISS 벡터 저장소 로드 완료")
                    return True
                    
            print("기존 벡터 저장소를 찾을 수 없습니다.")
            return False
            
        except Exception as e:
            print(f"벡터 저장소 로드 중 오류 발생: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        기존 벡터 저장소에 새 문서를 추가하는 함수
        
        Args:
            documents (List[Document]): 추가할 문서 리스트
        """
        if not self.vector_store:
            print("벡터 저장소가 초기화되지 않았습니다.")
            return
            
        try:
            self.vector_store.add_documents(documents)
            
            if self.vector_db_type == "chroma":
                self.vector_store.persist()
            elif self.vector_db_type == "faiss":
                faiss_path = os.path.join(self.persist_directory, "faiss_index")
                self.vector_store.save_local(faiss_path)
                
            print(f"문서 추가 완료: {len(documents)}개")
            
        except Exception as e:
            print(f"문서 추가 중 오류 발생: {e}")
    
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
            print("벡터 저장소가 초기화되지 않았습니다.")
            return []
            
        try:
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
            
            return results
            
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
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
            print("벡터 저장소가 초기화되지 않았습니다.")
            return []
            
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            return results
            
        except Exception as e:
            print(f"점수 포함 검색 중 오류 발생: {e}")
            return []
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """
        벡터 저장소 정보를 반환하는 함수
        
        Returns:
            Dict: 벡터 저장소 정보
        """
        info = {
            "embedding_model": self.embedding_model_type,
            "vector_db_type": self.vector_db_type,
            "persist_directory": self.persist_directory,
            "is_initialized": self.vector_store is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.vector_store and self.vector_db_type == "chroma":
            try:
                # Chroma 컬렉션 정보
                collection = self.vector_store._collection
                info["document_count"] = collection.count()
            except:
                info["document_count"] = "Unknown"
        
        return info
    
    def test_search(self, test_queries: List[str]) -> None:
        """
        테스트 검색을 수행하는 함수
        
        Args:
            test_queries (List[str]): 테스트할 쿼리 리스트
        """
        if not self.vector_store:
            print("벡터 저장소가 초기화되지 않았습니다.")
            return
            
        print("\n=== 벡터 저장소 테스트 검색 ===")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 테스트 {i}: '{query}' ---")
            
            # 유사도 검색 (점수 포함)
            results = self.similarity_search_with_score(query, k=3)
            
            for j, (doc, score) in enumerate(results, 1):
                print(f"{j}. [점수: {score:.4f}] {doc.metadata.get('title', 'N/A')}")
                print(f"   카테고리: {doc.metadata.get('category', 'N/A')}")
                print(f"   내용: {doc.page_content[:100]}...")
                print()


def main():
    """
    메인 실행 함수 - 벡터 저장소 구축 데모
    """
    print("=== 벡터 저장소 구축 시작 ===")
    
    # 1. 문서 처리기로 문서 로드 및 처리
    print("1. 문서 처리 중...")
    processor = DocumentProcessor()
    chunks = processor.process_documents()
    
    if not chunks:
        print("처리된 문서가 없습니다. 먼저 테스트 데이터를 생성해주세요.")
        return
    
    print(f"처리된 청크 수: {len(chunks)}")
    
    # 2. 벡터 저장소 관리자 초기화
    print("\n2. 벡터 저장소 관리자 초기화 중...")
    vector_manager = VectorStoreManager(
        embedding_model="korean",  # 한국어 특화 모델 사용
        vector_db_type="chroma",   # Chroma DB 사용
        persist_directory="data/vector_db"
    )
    
    # 3. 기존 벡터 저장소 로드 시도
    print("\n3. 기존 벡터 저장소 확인 중...")
    if not vector_manager.load_existing_vector_store():
        # 4. 새 벡터 저장소 생성
        print("\n4. 새 벡터 저장소 생성 중...")
        
        # 기존 벡터 저장소 디렉토리 삭제 (깨끗하게 시작)
        import shutil
        if os.path.exists(vector_manager.persist_directory):
            shutil.rmtree(vector_manager.persist_directory)
            os.makedirs(vector_manager.persist_directory, exist_ok=True)
            print("기존 벡터 저장소 디렉토리 삭제 후 재생성")
        
        vector_manager.create_vector_store(chunks)
    else:
        print("기존 벡터 저장소를 성공적으로 로드했습니다.")
    
    # 5. 벡터 저장소 정보 출력
    print("\n5. 벡터 저장소 정보:")
    info = vector_manager.get_vector_store_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # 6. 테스트 검색 수행 (벡터 저장소가 제대로 초기화되었는지 확인)
    if vector_manager.vector_store is not None:
        test_queries = [
            "장학금 신청 방법",
            "도서관 운영시간",
            "해외교환학생 프로그램",
            "취업박람회 일정",
            "학생식당 메뉴"
        ]
        
        vector_manager.test_search(test_queries)
    else:
        print("⚠️ 벡터 저장소가 초기화되지 않아 테스트를 건너뜁니다.")
    
    print("\n=== 벡터 저장소 구축 완료 ===")


if __name__ == "__main__":
    main()