# 메인 애플리케이션
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# 환경 변수 로드
load_dotenv()

def test_langchain_setup():
    """
    LangChain 설정이 제대로 되었는지 테스트하는 함수
    """
    try:
        # ChatOpenAI 모델 초기화
        chat = ChatOpenAI(temperature=0)
        
        # 간단한 메시지 전송 및 응답 확인
        messages = [HumanMessage(content="안녕하세요! LangChain 테스트입니다.")]
        response = chat.invoke(messages)
        
        print("성공! LangChain 설정이 완료되었습니다.")
        print(f"응답: {response.content}")
        return True
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

if __name__ == "__main__":
    test_langchain_setup()