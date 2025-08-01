from fastapi import FastAPI, Query
from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import uvicorn
import os

app = FastAPI()

@app.get("/summarize")
def summarize(url: str = Query(...)):
    try:
        docs = YoutubeLoader.from_youtube_url(url, language="ko").load()
        llm = ChatOpenAI(temperature=0)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("yt_summary_server:app", host="0.0.0.0", port=8000)
