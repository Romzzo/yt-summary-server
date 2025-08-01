from fastapi import FastAPI, Query
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import os

app = FastAPI()

@app.get("/summarize")
def summarize(url: str = Query(...)):
    try:
        # 유튜브 영상 ID 추출
        video_id = url.split("v=")[-1].split("&")[0]

        # 자막 가져오기
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko", "en"])
        formatter = TextFormatter()
        text = formatter.format_transcript(transcript)

        # 요약
        llm = ChatOpenAI(temperature=0)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run([{"page_content": text}])

        return {"summary": summary}

    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/modules")
def check_modules():
    try:
        import youtube_transcript_api
        return {"status": "✅ youtube_transcript_api is installed"}
    except Exception as e:
        return {"status": "❌ not installed", "error": str(e)}
        
if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run("yt_summary_server:app", host="0.0.0.0", port=8000)
    except Exception as e:
        print("🔥 서버 실행 중 에러 발생:", e)
