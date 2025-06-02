import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import time  # 이 줄 추가!

# 프로젝트 루트 경로를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
# llm.py가 프로젝트 루트에 있으므로 current_dir이 바로 프로젝트 루트
project_root = current_dir
sys.path.insert(0, project_root)

print(f"현재 디렉토리: {current_dir}")
print(f"프로젝트 루트: {project_root}")
print(f"Python 경로에 추가됨: {project_root}")

from langchain_openai import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA

# 기존에 만든 모듈들 임포트
try:
    from utils.document_processing import DocumentProcessor
    from utils.vector_store import SafeVectorStoreManager as VectorStoreManager
    print("✅ 모듈 임포트 성공")
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print(f"utils 폴더 경로 확인: {os.path.join(project_root, 'utils')}")
    print(f"utils 폴더 존재: {os.path.exists(os.path.join(project_root, 'utils'))}")
    raise

# 환경 변수 로드
load_dotenv()

class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) 파이프라인 클래스
    벡터 검색 + LLM 응답 생성을 통합 관리
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1-0,
                 max_tokens: int = 400,
                 request_timeout=15,     # 응답 대기 시간 (초)
                 embedding_model: str = "korean",
                 vector_db_type: str = "chroma"):
        """
        RAGPipeline 초기화
        
        Args:
            model_name (str): 사용할 OpenAI 모델
            temperature (float): 생성 창의성 조절 (0.0~1.0)
            max_tokens (int): 최대 응답 토큰 수
            embedding_model (str): 임베딩 모델 타입
            vector_db_type (str): 벡터 DB 타입
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # OpenAI ChatGPT 모델 초기화
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 벡터 저장소 관리자 초기화
        self.vector_manager = VectorStoreManager(
            embedding_model=embedding_model,
            vector_db_type=vector_db_type,
            persist_directory=os.path.join(os.getcwd(), "data", "vector_db")  # 절대 경로 사용
        )
        
        # 프롬프트 템플릿 설정
        self.prompt_template = self._create_prompt_template()
        
        # QA 체인 (나중에 초기화)
        self.qa_chain = None
        
        print(f"RAG 파이프라인 초기화 완료:")
        print(f"  - LLM 모델: {model_name}")
        print(f"  - Temperature: {temperature}")
        print(f"  - 임베딩: {embedding_model}")
        print(f"  - 벡터 DB: {vector_db_type}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """
        학교 공지사항 특화 프롬프트 템플릿 생성
        
        Returns:
            PromptTemplate: 설정된 프롬프트 템플릿
        """
        template = """당신은 경국대학교의 AI 공지사항 도우미입니다.
학생들의 질문에 대해 정확하고 친절하게 답변해주세요.

다음 정보를 바탕으로 질문에 답변하세요:

=== 관련 공지사항 ===
{context}

=== ⚠️ 중요한 답변 규칙 ===
1. **반드시 제공된 공지사항 정보만을 기반으로 답변하세요**
2. **제공된 문서에 해당 정보가 없으면 반드시 "해당 정보를 찾을 수 없습니다"라고 명시하세요**
3. **절대로 추측하거나 일반적인 정보로 답변하지 마세요**
4. **다른 정보와 혼동하여 답변하지 마세요**
5. 답변 끝에 관련 부서 연락처가 있으면 함께 안내하세요  📞
6. 친근하고 정중한 톤으로 답변하세요
7. 한국어로 답변하세요
8. 답변은 300자 이내로 간결하게 작성하세요
9. 이모지를 적절히 사용하여 가독성을 높이세요 📝
10. 중요한 정보는 줄바꿈으로 구분하세요

질문: {question}

답변:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def setup_rag_system(self, force_rebuild: bool = False) -> bool:
        """
        RAG 시스템을 설정하는 함수
        
        Args:
            force_rebuild (bool): 강제로 벡터 저장소 재구축 여부
            
        Returns:
            bool: 설정 성공 여부
        """
        try:
            print("=== RAG 시스템 설정 중 ===")
            
            # SafeVectorStoreManager의 메서드에 맞춰 수정
            if not force_rebuild:
                # 기존 벡터 저장소가 있는지 확인
                # 기존 벡터 저장소가 있는지 확인
                # 기존 벡터 저장소가 있는지 확인
                if os.path.exists(self.vector_manager.persist_directory):
                    print("✅ 기존 벡터 저장소 디렉토리 발견")
                    # 사용 가능한 컬렉션 찾기
                    try:
                        import chromadb
                        client = chromadb.PersistentClient(path=str(self.vector_manager.persist_directory))
                        collections = client.list_collections()
                        
                        # 문서가 있는 첫 번째 컬렉션 사용
                        for collection in collections:
                            count = collection.count()
                            if count > 0:
                                print(f"✅ '{collection.name}' 컬렉션 발견 ({count}개 문서)")
                                
                                from langchain_chroma import Chroma
                                self.vector_manager.vector_store = Chroma(
                                    persist_directory=str(self.vector_manager.persist_directory),
                                    embedding_function=self.vector_manager.embeddings,
                                    collection_name=collection.name  # 실제 컬렉션 이름 사용
                                )
                                print("✅ 기존 벡터 저장소 로드 완료")
                                break
                        else:
                            print("❌ 문서가 있는 컬렉션을 찾을 수 없습니다.")
                            force_rebuild = True
                            
                    except Exception as e:
                        print(f"기존 벡터 저장소 로드 실패: {e}")
                        force_rebuild = True
                else:
                    print("기존 벡터 저장소를 찾을 수 없습니다.")
                    force_rebuild = True
            
            if force_rebuild or not self.vector_manager.vector_store:
                # 새로운 벡터 저장소 구축
                print("🔄 새로운 벡터 저장소 구축 중...")
                
                # 문서 처리
                processor = DocumentProcessor()
                chunks = processor.process_documents()
                
                if not chunks:
                    print("❌ 처리할 문서가 없습니다.")
                    return False
                
                # 벡터 저장소 생성 (SafeVectorStoreManager의 메서드 사용)
                self.vector_manager.create_vector_store_safe(chunks)
                print("✅ 새로운 벡터 저장소 구축 완료")
            
            # 벡터 저장소가 제대로 초기화되었는지 확인
            if not self.vector_manager.vector_store:
                print("❌ 벡터 저장소 초기화 실패")
                return False
            
            # 벡터 저장소 테스트
            print("🔍 벡터 저장소 테스트 중...")
            try:
                test_results = self.vector_manager.vector_store.similarity_search("테스트", k=1)
                print(f"벡터 저장소 문서 수 확인: {len(test_results)}개 문서 발견")
                if test_results:
                    print(f"첫 번째 문서 제목: {test_results[0].metadata.get('title', 'N/A')}")
                else:
                    print("⚠️ 벡터 저장소가 비어있습니다!")
            except Exception as e:
                print(f"⚠️ 벡터 저장소 테스트 실패: {e}")
            
            # 3. QA 체인 구성
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # 모든 관련 문서를 하나의 프롬프트에 포함
                retriever=self.vector_manager.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}  # 상위 3개 유사 문서 검색
                ),
                chain_type_kwargs={
                    "prompt": self.prompt_template,
                    "verbose": True  # 디버깅을 위한 상세 출력
                },
                return_source_documents=True  # 출처 문서 반환
            )
            
            print("✅ RAG 시스템 설정 완료")
            return True
            
        except Exception as e:
            print(f"❌ RAG 시스템 설정 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변을 생성하는 함수
        
        Args:
            question (str): 사용자 질문
            
        Returns:
            Dict: 답변 결과 (답변, 출처, 메타데이터 포함)
        """
        if not self.qa_chain:
            return {
                "answer": "RAG 시스템이 초기화되지 않았습니다.",
                "source_documents": [],
                "error": True
            }
        
        try:
            print(f"\n🔍 질문 처리 중: '{question}'")
            
            # 출력 숨기기
            import contextlib
            import io
            
            with contextlib.redirect_stdout(io.StringIO()):
                result = self.qa_chain.invoke({"query": question})
            
            # 결과 정리
            answer = result.get("result", "답변을 생성할 수 없습니다.")
            source_docs = result.get("source_documents", [])
            
            # 출처 정보 정리 (간소화)
            sources = []
            for doc in source_docs:
                source_info = {
                    "title": doc.metadata.get("title", "제목 없음"),
                    "category": doc.metadata.get("category", "분류 없음"),
                    "date": doc.metadata.get("date", "날짜 없음")
                    # content_preview 제거
                }
                sources.append(source_info)
            
            response = {
                "answer": answer,
                "source_documents": sources,
                "total_sources": len(sources),
                "timestamp": datetime.now().isoformat(),
                "error": False
            }
            
            print(f"✅ 답변 생성 완료 (출처: {len(sources)}개)")
            return response
            
        except Exception as e:
            print(f"❌ 질문 처리 중 오류: {e}")
            return {
                "answer": f"질문 처리 중 오류가 발생했습니다: {str(e)}",
                "source_documents": [],
                "error": True
            }
            
    def batch_ask_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        여러 질문을 일괄 처리하는 함수
        
        Args:
            questions (List[str]): 질문 리스트
            
        Returns:
            List[Dict]: 각 질문에 대한 답변 리스트
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n=== 질문 {i}/{len(questions)} ===")
            result = self.ask_question(question)
            result["question"] = question
            results.append(result)
        
            # API 속도 제한 방지를 위한 대기
            if i < len(questions):  # 마지막 질문이 아니면
                print("⏳ API 안정화를 위해 2초 대기...")
                time.sleep(2)
        
        return results
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        질문과 유사한 문서들을 검색하는 함수 (LLM 없이 순수 검색만)
        
        Args:
            query (str): 검색 쿼리
            k (int): 반환할 문서 수
            
        Returns:
            List[Dict]: 유사 문서 정보 리스트
        """
        if not self.vector_manager.vector_store:
            return []
        
        try:
            # SafeVectorStoreManager의 메서드 사용
            results = self.vector_manager.similarity_search_with_score(query, k=k)
            
            similar_docs = []
            for doc, score in results:
                doc_info = {
                    "title": doc.metadata.get("title", "제목 없음"),
                    "category": doc.metadata.get("category", "분류 없음"),
                    "date": doc.metadata.get("date", "날짜 없음"),
                    "similarity_score": float(score),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                similar_docs.append(doc_info)
            
            return similar_docs
            
        except Exception as e:
            print(f"문서 검색 중 오류: {e}")
            return []
    
    def evaluate_system(self, test_questions: List[str]) -> Dict[str, Any]:
        """
        시스템 성능을 평가하는 함수
        
        Args:
            test_questions (List[str]): 테스트 질문들
            
        Returns:
            Dict: 평가 결과
        """
        if not self.qa_chain:
            return {"error": "RAG 시스템이 초기화되지 않았습니다."}
        
        print("=== 시스템 성능 평가 중 ===")
        
        results = self.batch_ask_questions(test_questions)
        
        # 성능 지표 계산
        total_questions = len(results)
        successful_answers = len([r for r in results if not r.get("error", False)])
        avg_sources = sum(r.get("total_sources", 0) for r in results) / total_questions if total_questions > 0 else 0
        
        evaluation = {
            "total_questions": total_questions,
            "successful_answers": successful_answers,
            "success_rate": successful_answers / total_questions if total_questions > 0 else 0,
            "average_sources_per_answer": avg_sources,
            "detailed_results": results
        }
        
        print(f"📊 평가 완료:")
        print(f"  - 총 질문 수: {total_questions}")
        print(f"  - 성공한 답변: {successful_answers}")
        print(f"  - 성공률: {evaluation['success_rate']:.1%}")
        print(f"  - 평균 출처 수: {avg_sources:.1f}")
        
        return evaluation
    
    def update_system_prompt(self, new_template: str):
        """
        시스템 프롬프트를 업데이트하는 함수
        
        Args:
            new_template (str): 새로운 프롬프트 템플릿
        """
        self.prompt_template = PromptTemplate(
            template=new_template,
            input_variables=["context", "question"]
        )
        
        # QA 체인 재구성
        if self.qa_chain:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_manager.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                    chain_type_kwargs={
                        "prompt": self.prompt_template,
                        "verbose": False  # verbose 끄기!
                    },
                return_source_documents=True
            )
        
        print("✅ 프롬프트 템플릿 업데이트 완료")


def main():
    """
    RAG 파이프라인 테스트 실행
    """
    print("=== RAG 파이프라인 테스트 시작 ===")
    
    # RAG 파이프라인 초기화
    rag = RAGPipeline(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=300,  # 더 짧은 답변으로 속도 향상
        request_timeout=30
    )
    
    # RAG 시스템 설정
    if not rag.setup_rag_system():
        print("RAG 시스템 설정에 실패했습니다.")
        return
    
    # 테스트 질문들
    test_questions = [
        "장학금 신청 방법을 알려주세요",
        "도서관 운영시간이 어떻게 되나요?",
        "해외교환학생 프로그램에 대해 설명해주세요",
        "취업박람회는 언제 열리나요?",
        "학생식당 메뉴가 변경되었나요?",
        "기숙사 신청은 어떻게 하나요?"  # 데이터에 없는 질문
    ]
    
    # 개별 질문 테스트 (전체 6개)
    print("\n=== 개별 질문 테스트 ===")
    for question in test_questions:  # [:3] 제거 → 전체 6개
        result = rag.ask_question(question)
        
        print(f"\n질문: {question}")
        print(f"답변: {result['answer']}")
        print(f"출처 수: {result['total_sources']}")
        
        if result['source_documents']:
            print("관련 문서:")
            for i, source in enumerate(result['source_documents'], 1):
                print(f"  {i}. {source['title']} ({source['category']})")
    
    # 시스템 성능 평가
    print("\n=== 시스템 성능 평가 ===")
    evaluation = rag.evaluate_system(test_questions)
    
    print(f"전체 성공률: {evaluation['success_rate']:.1%}")
    
    print("\n=== RAG 파이프라인 테스트 완료 ===")


if __name__ == "__main__":
    main()