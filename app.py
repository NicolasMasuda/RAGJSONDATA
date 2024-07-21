import os
import json
import tempfile
import uuid
from flask import Flask, request, jsonify, make_response
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks import get_openai_callback

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Almacén para guardar las rutas de archivos por usuario
docs_store = {}

# Almacén para guardar los historiales de chat por usuario
history_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

def initialize_vector_store(file_path: str):
    loader = JSONLoader(file_path=file_path, jq_schema='.[]', text_content=False)
    data = loader.load()
    embedding_openai = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)
    doc = FAISS.from_documents(documents=data, embedding=embedding_openai)
    return doc

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['files']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.mimetype == 'application/json':
        try:
            # Crear un archivo temporal y guardar el contenido del archivo
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file)
                temp_file_path = temp_file.name

            session_id = request.cookies.get('session_id')
            if not session_id:
                session_id = str(uuid.uuid4())
                #print(session_id)
                response = make_response(jsonify({"message": "File received and stored successfully", "session_id": session_id}), 200)
                response.set_cookie('session_id', session_id)
            else:
                response = jsonify({"message": "File received and stored successfully"}), 200

            docs_store[session_id] = temp_file_path
            #print(docs_store[session_id])
            #print(docs_store)
            return response
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON file"}), 400

@app.route('/receiveMessage', methods=['POST'])
def receive_message():
    request_data = request.get_json()
    query = request_data["query"]

    session_id = request.cookies.get('session_id')
    if not session_id or session_id not in docs_store:
        return jsonify({"error": "Vector store not initialized"}), 500

    file_path = docs_store[session_id]
    doc = initialize_vector_store(file_path)

    retriever = doc.as_retriever(search_kwargs={"k": 500})

    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0.0)

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise. Siempre responde en Español.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    with get_openai_callback() as cb:
        response = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        answer = response["answer"]

    print(history_store)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run()