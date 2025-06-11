const express = require('express');
const axios = require('axios');
require('dotenv').config();

const app = express();
app.use(express.json());

// RAG 서버 URL
const RAG_SERVER_URL = process.env.RAG_SERVER_URL || 'https://8f11-112-214-38-159.ngrok-free.app';

// 기본 라우트
app.get('/', (req, res) => {
    res.send('Notee 카카오톡 봇 서버 실행중!');
});

// 서버 상태 확인
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// 카카오톡 메시지 처리 웹훅
app.post('/webhook', async (req, res) => {
    try {
        const userMessage = req.body.userRequest?.utterance;
        const userId = req.body.userRequest?.user?.id;
        
        console.log('=== 새 메시지 ===');
        console.log('사용자 ID:', userId);
        console.log('사용자 메시지:', userMessage);
        console.log('전체 요청:', JSON.stringify(req.body, null, 2));
        
        let responseText = '';
        
        // 빈 메시지 처리
        if (!userMessage || userMessage.trim() === '') {
            responseText = '무엇을 도와드릴까요?\n\n💡 이런 것들을 물어보세요:\n• 장학금 정보\n• 학사일정\n• 수강신청\n• 공지사항 검색';
        } else {
            try {
                // RAG 서버에 질문 보내기
                console.log('RAG 서버에 요청 중...');
                const ragResponse = await axios.post(`${RAG_SERVER_URL}/webhook`, {
                    question: userMessage,
                    user_id: userId
                });
                
                responseText = ragResponse.data.answer || '답변을 생성하지 못했습니다.';
                console.log('RAG 서버 응답:', responseText);
                
            } catch (ragError) {
                console.log('RAG 서버 연결 실패:', ragError.message);
                
                // RAG 서버 실패 시 기본 응답 로직
                responseText = getBasicResponse(userMessage);
            }
        }
        
        // 카카오톡 응답 형식
        const response = {
            version: "2.0",
            template: {
                outputs: [{
                    simpleText: {
                        text: responseText
                    }
                }]
            }
        };
        
        console.log('최종 응답:', responseText);
        console.log('================');
        
        res.json(response);
        
    } catch (error) {
        console.error('웹훅 처리 중 오류:', error);
        
        // 에러 시 기본 응답
        const errorResponse = {
            version: "2.0",
            template: {
                outputs: [{
                    simpleText: {
                        text: "죄송합니다. 일시적인 오류가 발생했습니다.\n잠시 후 다시 시도해주세요."
                    }
                }]
            }
        };
        
        res.json(errorResponse);
    }
});

function getBasicResponse(userMessage) {
    const message = userMessage.toLowerCase();
    
}

// 에러 핸들링 미들웨어
app.use((error, req, res, next) => {
    console.error('서버 에러:', error);
    res.status(500).json({ error: '서버 내부 오류' });
});

// 404 핸들링
app.use((req, res) => {
    res.status(404).json({ error: '요청한 경로를 찾을 수 없습니다.' });
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log(`=================================`);
    console.log(`🤖 Notee 카카오톡 봇 서버 시작!`);
    console.log(`📍 서버 주소: http://localhost:${PORT}`);
    console.log(`🔗 RAG 서버: ${RAG_SERVER_URL}`);
    console.log(`⏰ 시작 시간: ${new Date().toLocaleString('ko-KR')}`);
    console.log(`=================================`);
});