import os
import bs4
from collections import defaultdict
import uuid
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

# -------------------------------------------------------------------
# 1) Environment & Model Setup
# -------------------------------------------------------------------
PPLX_API_KEY = ""


if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "rahul-chatbot/1.0"

if not os.getenv("PPLX_API_KEY"):
    raise ValueError(
        "Missing PPLX_API_KEY in your environment. "
        "Add it to your .env file (PPLX_API_KEY=...)."
    )

# Initialize LLM
llm = init_chat_model("sonar", model_provider="perplexity", api_key=PPLX_API_KEY)

# Initialize embeddings & vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

# -------------------------------------------------------------------
# 2) Load & Index Knowledge
# -------------------------------------------------------------------
loader = WebBaseLoader(
    web_paths=(
        "https://podstechnologysolutions.com/",
        "https://podstechnologysolutions.com/about.html",
        "https://podstechnologysolutions.com/Blog/index.html",
        "https://podstechnologysolutions.com/training.html",
        "https://podstechnologysolutions.com/cyberSservices.html",
        "https://podstechnologysolutions.com/digitalServices.html",
        "https://academy.podstechnologysolutions.com/",
        "https://academy.podstechnologysolutions.com/faq.html",
        "https://academy.podstechnologysolutions.com/contact.html",
        "https://academy.podstechnologysolutions.com/course.html",
    ),
    header_template={"User-Agent": os.environ["USER_AGENT"]},
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=(
            "col-lg-6 col-md-12",
            "testimonials__info",
            "boxed_wrapper",
            "envato_tk_templates-template envato_tk_templates-template-elementor_header_footer single single-envato_tk_templates postid-1098 elementor-default elementor-template-full-width elementor-kit-9 elementor-page elementor-page-1098 e--ua-blink e--ua-chrome e--ua-webkit"
        ))
    ),
)

docs = loader.load()
if not docs:
    print("Warning: No documents loaded from the URL. Retrieval will be empty.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
if all_splits:
    vector_store.add_documents(documents=all_splits)
    print(f"Indexed {len(all_splits)} chunks.")
else:
    print("Warning: No chunks to index.")

# -------------------------------------------------------------------
# 3) Conversation Memory (per session)
# -------------------------------------------------------------------
session_memories = defaultdict(InMemoryChatMessageHistory)

def _format_history(messages, max_chars: int = 2000) -> str:
    lines = []
    for m in messages:
        role = "User" if m.type == "human" else ("Assistant" if m.type == "ai" else m.type)
        lines.append(f"{role}: {m.content}")
    history = "\n".join(lines)
    if len(history) > max_chars:
        history = history[-max_chars:]
    return history

# -------------------------------------------------------------------
# 4) RAG + Memory Pipeline
# -------------------------------------------------------------------
def rag_pipeline_with_memory(user_query: str, session_id: str, k: int = 2) -> str:
    """Use per-session memory and RAG retrieval to get answer."""
    # Get memory for this session
    memory = session_memories[session_id]
    memory.add_user_message(user_query)

    # Retrieve relevant chunks
    docs_text = ""
    try:
        retrieved_docs = vector_store.similarity_search(user_query, k=k)
        if retrieved_docs:
            docs_text = "\n\n".join(d.page_content for d in retrieved_docs)
    except Exception as e:
        docs_text = ""
        print(f"(Retrieval warning) {e}")

    # Build system message
    history_text = _format_history(memory.messages)
    system_message_content = (
        "You are an helpful assistant of PODS Technology Solutions. "
        "Rules for replying"
        "1. Answer questions as if you are a real employee of the company. "
        "2. Don not answer anything which is **not related to *PODS Technology Solutions* and refer rule no. 8 for reply"
        "3. Do not answer questions about unrelated topics like countries, history, or general knowledge. "
        "4. Do not mention AI, your capabilities,add references, or any external references. "
        "5. Do not use markdown, asterisks, bullet points, brackets, underline or unnecessary punctuation. "
        "6. Use short, clear, natural conversational language, as if you are a real employee. "
        "7. Give short and clear answers. Include a brief conclusion if possible. "
        "8. If the question is unrelated to the website, reply: 'Sorry I can only answer questions about PODS Technology Solutions.'\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Retrieved context:\n{docs_text}"
    )

    messages = [SystemMessage(content=system_message_content),
                HumanMessage(content=user_query)]
    response = llm.invoke(messages)

    reply_text = getattr(response, "content", str(response))
    memory.add_ai_message(reply_text)
    return reply_text

# -------------------------------------------------------------------
# 5) Flask API
# -------------------------------------------------------------------
app = Flask(__name__)

CORS(app, origins="https://rcodyp.github.io") 
app.secret_key = "supersecretkey123"  # Required for session support

@app.route("/")
def home():
    return {"message": "PODS Chatbot Flask API is running"}

@app.route("/chat", methods=["POST"])
def chat():
    # Assign per-browser session_id automatically
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    session_id = session["session_id"]
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    answer = rag_pipeline_with_memory(question, session_id)
    return jsonify({"session_id": session_id, "question": question, "answer": answer})

# -------------------------------------------------------------------
# 6) Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
