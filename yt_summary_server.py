from fastapi import FastAPI, Query
from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import os

app = FastAPI()

@app.get("/summarize")
def summarize(url: str = Query(...)):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "API 키가 없습니다."}

    try:
        loader = YoutubeLoader.from_youtube_url(url, language="ko")
        docs = loader.load()
        llm = ChatOpenAI(temperature=0)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
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
