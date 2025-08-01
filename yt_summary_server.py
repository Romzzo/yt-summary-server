from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import re, os

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

@app.route("/summarize", methods=["GET"])
def summarize():
    url = request.args.get("url")
    vid = extract_video_id(url)
    if not vid:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=["ko", "en"])
        full_text = "\n".join([entry["text"] for entry in transcript])
        doc = Document(page_content=full_text)
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run([doc])
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
