from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from docx import Document

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

chat_history = []

def get_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

@app.get("/static/images/AI.png", response_class=FileResponse)
async def get_image():
    return FileResponse("path/to/AI.png", headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

@app.post('/get_response', response_class=JSONResponse)
async def get_response(user_input: str = Form(...)):
    if user_input:
        files_folder = 'data'

        text = ""
        for filename in os.listdir(files_folder):
            file_path = os.path.join(files_folder, filename)
            if filename.endswith('.pdf'):
                text += get_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                text += get_text_from_docx(file_path)

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200, 
            length_function=len 
        )
        chunks = text_splitter.split_text(text)

        # Accessing environment variable
        openai_api_key = os.environ.get("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")

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        docs = knowledge_base.similarity_search(user_input)

        llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0.4
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_input)

        chat_history.append({'user': user_input, 'assistant': response})

        return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
