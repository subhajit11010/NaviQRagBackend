# from annotated_types import doc
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo import ASCENDING
from pymongo.operations import SearchIndexModel
from fastapi.middleware.cors import CORSMiddleware
import fitz
from dotenv import load_dotenv
# import numpy as np
# import pytesseract
# from pdf2image import convert_from_bytes
import img2pdf
# from PIL import Image
from google import genai
# import time
import io
import os
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

app = Flask(__name__)
CORS(app) 
ocr_model = ocr_predictor(pretrained=True)
print("model downloaded")

client = MongoClient(os.getenv("MONGO_URI"))
db = client["NaviQ"]
collection: Collection = db["rag_db"]
collection.create_index([("organization_id", ASCENDING)])

# The embedding model
api_key = os.getenv("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key)

def get_embedding(data):
    """Generates vector embeddings for the given data."""
    result = genai_client.models.embed_content( model="gemini-embedding-001", contents=data)
    return result.embeddings[0].values

def getChunks(text, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_documents(text)

def get_query_results(org_id, query):
  """Gets results from a vector search query."""
  query_embedding = get_embedding(query)
  pipeline = [
      {
            "$vectorSearch": {
              "index": "vector_index",
              "queryVector": query_embedding,
              "path": "embedding",
              "exact": True,
              "limit": 5,
              "filter": {
                  "organization_id": org_id
              }
            }
      }, {
            "$project": {
              "_id": 0,
              "text": 1
         }
      }
  ]

  results = collection.aggregate(pipeline=pipeline)  
  array_of_results = []
  for doc in results:
      array_of_results.append(doc)
  return array_of_results

def extract_text_from_doctr(result):
    json_export = result.export()
    text = ""
    for page in json_export["pages"]:
        for block in page["blocks"]:
            for line in block["lines"]:
                text += " ".join([w["value"] for w in line["words"]]) + "\n"
    return text


@app.route("/upload", methods=["POST"])
def upload_file():
    organization_id = request.form.get("organization_id")
    file = request.files.get("file")

    if not file or not organization_id:
        return jsonify({"error": "Missing file or organization_id"}), 400
    
    contents = file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    print(doc)
    text = ""
    # Here the case 1
    if file.filename.lower().endswith(".pdf"):
        for page in doc:
            text += page.get_text()
    
        if text.strip() == "":
            # Here I will use OCR
            ocr_doc = DocumentFile.from_pdf(io.BytesIO(contents))
            result = ocr_model(ocr_doc)
            text = extract_text_from_doctr(result)

    else:
        pdf_bytes = img2pdf.convert(contents)
        ocr_doc = DocumentFile.from_pdf(io.BytesIO(pdf_bytes))
        result = ocr_model(ocr_doc)
        text = extract_text_from_doctr(result)

    print(text)
    # return text

    doc_obj = [Document(page_content=text)]
    documents = getChunks(doc_obj, 400, 20)


    # print(documents)
    docs_to_insert = [{
        "organization_id": organization_id,  # app-write id
        "text": d.page_content,
        "embedding": get_embedding(d.page_content)
    } for d in documents]

    collection.insert_many(docs_to_insert)
    index_name="vector_index"
    search_index_model = SearchIndexModel(
    definition = {
        "fields": [
            {
                "type": "vector",
                "numDimensions": 3072,
                "path": "embedding",
                "similarity": "cosine"
            },
            { 
                "type": "filter",
                "path": "organization_id"
            }
        ]
    },
    name = index_name,
    type = "vectorSearch"
    )
    # collection.create_search_index(model=search_index_model)

    try:
        collection.create_search_index(model=search_index_model)
    except Exception:
        pass
    return {"message": "File uploaded successfully"}

@app.route("/query", methods=["GET"])
def query():
    organization_id = request.args.get("organization_id")
    question = request.args.get("question")

    context_docs = get_query_results(organization_id, question)
    context_string = " ".join([doc["text"] for doc in context_docs])
    prompt = f"""Use the following pieces of context to answer the question at the end.
    {context_string}
    Question: {question}
    """
    response = genai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    return response.text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
