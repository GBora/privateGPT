#!/usr/bin/env python3
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from web_models import questionModel
from ingest import main as ingest

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/upload")
async def upload_source(uploaded_file: UploadFile = File(...)):
    file_location = f"./source_documents/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    return {"info": f"file '{uploaded_file.filename}' saved at '{file_location}'"}


@app.post("/ingest")
async def trigger_ingest():
    res = ingest()
    return {
        "result": res
    }


@app.post("/question")
async def root(question: questionModel.Question):
    res = qa(question.text)
    return {
        "answer": res['result'],
        "sources": res['source_documents']
    }


def main():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = []
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit()
    global qa
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


main()
