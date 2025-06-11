import os
import json
from typing import List, Dict, Any
from datetime import datetime

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    """
    문서 로딩 및 처리를 담당하는 클래스
    """
    
    def __init__(self, data_directory: str = "../data/raw"):
        """
        DocumentProcessor 초기화
        
        Args:
            data_directory (str): 문서가 저장된 디렉토리 경로
        """
        self.data_directory = data_directory
        self.metadata_path = os.path.join(data_directory, "metadata.json")
        
        # 텍스트 분할기 설정 (한국어에 최적화)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # 청크 크기 (한국어 기준으로 적절히 설정)
            chunk_overlap=200,      # 청크 간 겹치는 부분
            length_function=len,    # 길이 계산 함수
            separators=[           # 분할 기준 (한국어에 맞게 설정)
                "\n\n",            # 문단 구분
                "\n",              # 줄바꿈
                "■",               # 한국 공지사항에서 자주 사용되는 구분자
                "。",               # 일본식 마침표 (가끔 사용됨)
                ".",               # 영어 마침표
                " ",               # 공백
                ""                 # 마지막 fallback
            ]
        )
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        메타데이터 파일을 로드하는 함수
        
        Returns:
            Dict: 메타데이터 정보
        """
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"메타데이터 로드 완료: {len(metadata['documents'])}개 문서")
            return metadata
        except FileNotFoundError:
            print(f"메타데이터 파일을 찾을 수 없습니다: {self.metadata_path}")
            return {"documents": []}
        except Exception as e:
            print(f"메타데이터 로드 중 오류 발생: {e}")
            return {"documents": []}
    
    def load_documents(self) -> List[Document]:
        """
        디렉토리에서 모든 텍스트 문서를 로드하는 함수
        
        Returns:
            List[Document]: 로드된 문서 리스트
        """
        try:
            # DirectoryLoader를 사용하여 모든 .txt 파일 로드
            loader = DirectoryLoader(
                self.data_directory,
                glob="*.txt",  # .txt 파일만 로드
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            
            documents = loader.load()
            print(f"문서 로드 완료: {len(documents)}개 파일")
            
            return documents
            
        except Exception as e:
            print(f"문서 로드 중 오류 발생: {e}")
            return []
    
    def enhance_documents_with_metadata(self, documents: List[Document]) -> List[Document]:
        """
        문서에 메타데이터를 추가하는 함수
        
        Args:
            documents (List[Document]): 원본 문서 리스트
            
        Returns:
            List[Document]: 메타데이터가 추가된 문서 리스트
        """
        metadata = self.load_metadata()
        metadata_dict = {}
        
        # 파일명을 키로 하는 메타데이터 딕셔너리 생성
        for doc_meta in metadata.get('documents', []):
            metadata_dict[doc_meta['filename']] = doc_meta
        
        enhanced_documents = []
        
        for doc in documents:
            # 파일명 추출
            filename = os.path.basename(doc.metadata.get('source', ''))
            
            # 메타데이터 추가
            if filename in metadata_dict:
                doc_meta = metadata_dict[filename]
                doc.metadata.update({
                    'title': doc_meta.get('title', ''),
                    'category': doc_meta.get('category', ''),
                    'date': doc_meta.get('date', ''),
                    'source_department': doc_meta.get('source', ''),
                    'document_id': doc_meta.get('id', 0)
                })
            
            # 문서 생성 시간 추가
            doc.metadata['processed_at'] = datetime.now().isoformat()
            
            enhanced_documents.append(doc)
        
        print(f"메타데이터 추가 완료: {len(enhanced_documents)}개 문서")
        return enhanced_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서를 적절한 크기로 분할하는 함수
        
        Args:
            documents (List[Document]): 분할할 문서 리스트
            
        Returns:
            List[Document]: 분할된 문서 청크 리스트
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # 각 청크에 청크 관련 메타데이터 추가
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_size'] = len(chunk.page_content)
            
            print(f"문서 분할 완료: {len(chunks)}개 청크 생성")
            return chunks
            
        except Exception as e:
            print(f"문서 분할 중 오류 발생: {e}")
            return []
    
    def process_documents(self) -> List[Document]:
        """
        전체 문서 처리 파이프라인을 실행하는 함수
        
        Returns:
            List[Document]: 처리된 문서 청크 리스트
        """
        print("=== 문서 처리 파이프라인 시작 ===")
        
        # 1. 문서 로드
        print("1. 문서 로딩 중...")
        documents = self.load_documents()
        
        if not documents:
            print("로드할 문서가 없습니다.")
            return []
        
        # 2. 메타데이터 추가
        print("2. 메타데이터 추가 중...")
        enhanced_documents = self.enhance_documents_with_metadata(documents)
        
        # 3. 문서 분할
        print("3. 문서 분할 중...")
        chunks = self.split_documents(enhanced_documents)
        
        print("=== 문서 처리 파이프라인 완료 ===")
        print(f"최종 결과: {len(chunks)}개 청크 생성")
        
        return chunks
    
    def preview_chunks(self, chunks: List[Document], num_preview: int = 3):
        """
        생성된 청크를 미리보기하는 함수
        
        Args:
            chunks (List[Document]): 청크 리스트
            num_preview (int): 미리볼 청크 개수
        """
        print(f"\n=== 청크 미리보기 (상위 {num_preview}개) ===")
        
        for i, chunk in enumerate(chunks[:num_preview]):
            print(f"\n--- 청크 {i+1} ---")
            print(f"제목: {chunk.metadata.get('title', 'N/A')}")
            print(f"카테고리: {chunk.metadata.get('category', 'N/A')}")
            print(f"날짜: {chunk.metadata.get('date', 'N/A')}")
            print(f"청크 크기: {chunk.metadata.get('chunk_size', 0)} 글자")
            print(f"내용 미리보기: {chunk.page_content[:200]}...")
            print("-" * 50)
    
    def get_document_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        문서 처리 통계를 생성하는 함수
        
        Args:
            chunks (List[Document]): 청크 리스트
            
        Returns:
            Dict: 통계 정보
        """
        if not chunks:
            return {}
        
        categories = {}
        total_chars = 0
        chunk_sizes = []
        
        for chunk in chunks:
            # 카테고리별 집계
            category = chunk.metadata.get('category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
            
            # 문자 수 집계
            chunk_size = len(chunk.page_content)
            total_chars += chunk_size
            chunk_sizes.append(chunk_size)
        
        stats = {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'average_chunk_size': total_chars / len(chunks),
            'categories': categories,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0
        }
        
        return stats
    
    def print_statistics(self, chunks: List[Document]):
        """
        통계 정보를 출력하는 함수
        
        Args:
            chunks (List[Document]): 청크 리스트
        """
        stats = self.get_document_statistics(chunks)
        
        print("\n=== 문서 처리 통계 ===")
        print(f"총 청크 수: {stats.get('total_chunks', 0)}")
        print(f"총 문자 수: {stats.get('total_characters', 0):,}")
        print(f"평균 청크 크기: {stats.get('average_chunk_size', 0):.1f} 글자")
        print(f"최소 청크 크기: {stats.get('min_chunk_size', 0)} 글자")
        print(f"최대 청크 크기: {stats.get('max_chunk_size', 0)} 글자")
        
        print("\n카테고리별 청크 수:")
        for category, count in stats.get('categories', {}).items():
            print(f"  - {category}: {count}개")


def main():
    """
    메인 실행 함수
    """
    # DocumentProcessor 인스턴스 생성
    processor = DocumentProcessor()
    
    # 문서 처리 실행
    chunks = processor.process_documents()
    
    if chunks:
        # 청크 미리보기
        processor.preview_chunks(chunks)
        
        # 통계 출력
        processor.print_statistics(chunks)
        
        # 처리된 청크를 파일로 저장 (선택사항)
        save_chunks_to_file(chunks)
    else:
        print("처리할 문서가 없습니다.")


def save_chunks_to_file(chunks: List[Document], output_file: str = "../data/processed/chunks.json"):
    """
    처리된 청크를 JSON 파일로 저장하는 함수
    
    Args:
        chunks (List[Document]): 저장할 청크 리스트
        output_file (str): 출력 파일 경로
    """
    try:
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 청크를 JSON 형태로 변환
        chunks_data = []
        for chunk in chunks:
            chunk_data = {
                'content': chunk.page_content,
                'metadata': chunk.metadata
            }
            chunks_data.append(chunk_data)
        
        # JSON 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n청크 데이터 저장 완료: {output_file}")
        
    except Exception as e:
        print(f"청크 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    main()