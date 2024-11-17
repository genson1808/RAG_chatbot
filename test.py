from llama_parse import LlamaParse
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser


from llama_index.core.schema import ImageDocument
from typing import List
from llama_index.core.node_parser import LlamaParseJsonNodeParser
from llama_index.core.schema import BaseNode, TextNode, Document
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
import glob
import nest_asyncio
import os
from dotenv import load_dotenv

import pickle
import json

nest_asyncio.apply()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import DirectoryLoader
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient

import nltk

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")


ins = """
You are a highly proficient language model designed to convert pages from PDF, PPT and other files into structured markdown text. Your goal is to accurately transcribe text, represent formulas in LaTeX MathJax notation, and identify and describe images, particularly graphs and other graphical elements.

You have been tasked with creating a markdown copy of each page from the provided PDF or PPT image. Each image description must include a full description of the content, a summary of the graphical object.

Maintain the sequence of all the elements.

For the following element, follow the requirement of extraction:
for Text:
   - Extract all readable text from the page.
   - Exclude any diagonal text, headers, and footers.

for Text which includes hyperlink:
    -Extract hyperlink and present it with the text
    
for Formulas:
   - Identify and convert all formulas into LaTeX MathJax notation.

for Image Identification and Description:
   - Identify all images, graphs, and other graphical elements on the page.
   - If image contains wording that is hard to extract , flag it with <unidentifiable section> instead of parsing.
   - For each image, include a full description of the content in the alt text, followed by a brief summary of the graphical object.
   - If the image has a subtitle or caption, include it in the description.
   - If the image has a formula convert it into LaTeX MathJax notation.
   - If the image has a organisation chart , convert it into a hierachical understandable format.
   - for graph , extract the value in table form as markdown representation

    
# OUTPUT INSTRUCTIONS

- Ensure all formulas are in LaTeX MathJax notation.
- Exclude any diagonal text, headers, and footers from the output.
- For each image and graph, provide a detailed description and summary.
"""

class llama_document_parser(object):
    def __init__(self,parsing_ins):
        self.parser = LlamaParse(
            result_tyres="json",
            api_key=llamaparse_api_key,
            verbose=True,
            ignore_errors=False,
            invalidate_cache=True,
            do_not_cache=True,
        )

    def get_image_text_nodes(self,download_path: str,json_objs: List[dict]):
        """Extract out text from images using a multimodal model."""
        image_dicts = self.parser.get_images(json_objs, download_path=download_path)
        image_documents = []
        img_text_nodes = []
        for image_dict in image_dicts:
            image_doc = ImageDocument(image_path=image_dict["path"])
            img_text_nodes.append(image_doc)
        return img_text_nodes

    def document_processing_llamaparse(self,file_name: str ,image_output_folder:str):
        """Parse document in using llamaparse and return extracted elements in json format"""
        json_objs = self.parser.get_json_result(file_name)
        json_list = json_objs[0]["pages"]
        print(json_list)
        if not os.path.exists(image_output_folder):
            os.mkdir(image_output_folder)

        image_text_nodes = self.get_image_text_nodes(image_output_folder,json_objs)

        data_file = "./store/parsed_data.pkl"
         # Save the parsed data to a file
        with open(data_file, "wb") as f:
            pickle.dump(json_list, f)
        return json_list

    # Create vector database

def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Load data from the pickle file
    with open("./store/parsed_data.pkl", "rb") as pkl_file:
        data = pickle.load(pkl_file)

    # Convert and save the data to a JSON file
    with open("./store/data.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    loader = DirectoryLoader('store/', glob="**/*.json", show_progress=True)
    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    #len(docs)
    #docs[0]

    # Initialize Embeddings
    embeddings = FastEmbedEmbeddings()

    # Create and persist a Chroma vector database from the chunked documents
    qdrant = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        url=qdrant_url,
        collection_name="rag2",
        api_key=qdrant_api_key,
        prefer_grpc=True,
    )
    return qdrant


# Đọc tệp JSON nếu đã có hoặc tạo dữ liệu từ file pickle
def load_data(pkl_file_path, json_file_path):
    """
    Đọc dữ liệu từ file JSON nếu có, nếu không có thì tạo từ file pickle và lưu dưới dạng JSON.
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            print(f"Data loaded from {json_file_path}")
    except FileNotFoundError:
        print(f"{json_file_path} not found. Creating data from pickle.")
        with open(pkl_file_path, "rb") as pkl_file:
            data = pickle.load(pkl_file)
        
        # Lưu dữ liệu dưới dạng JSON
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
            print(f"Data saved to {json_file_path}")
    
    return data

# Hàm chuẩn hóa văn bản cho embedding
def format_for_embedding(data):
    """
    Chuẩn hóa dữ liệu từ đối tượng `data` thành chuỗi văn bản để tạo embedding.
    """
    text_content = f"Page: {data['page']}\n"
    text_content += f"Text: {data['text']}\n"
    text_content += f"Markdown: {data['md']}\n"

    # Thêm thông tin từ 'items' nếu có
    for item in data.get("items", []):
        # Lấy 'value' với giá trị mặc định là 'N/A' nếu không có
        value = item.get('value', 'N/A')
        # Kiểm tra 'lvl' trước khi sử dụng
        level = item.get('lvl', '')
        # Tạo chuỗi thông tin từ 'type', 'level', và 'value'
        text_content += f"{item['type'].capitalize()} (Level {level}): {value}\n"

    return text_content

# Khởi tạo embeddings
def generate_embeddings(formatted_texts):
    """
    Tạo embeddings cho các văn bản đã chuẩn hóa.
    """
    embeddings = FastEmbedEmbeddings()

    # Sử dụng phương thức embed_documents thay vì embed
    return embeddings.embed_documents(formatted_texts)

# Lưu vào Qdrant
def save_to_qdrant(embeddings_list, data_objects, qdrant_url, qdrant_api_key):
    """
    Lưu embeddings và dữ liệu vào Qdrant.
    """
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=True)

    # Tạo points từ embeddings và metadata
    points = [
        PointStruct(
            id=index,  # unique ID cho từng document
            vector=embedding,
            payload={
                "page": data_obj["page"],
                "text": data_obj["text"],
                "markdown": data_obj["md"],
                "items": data_obj.get("items", []),
                "images": data_obj.get("images", [])
            }
        )
        for index, (embedding, data_obj) in enumerate(zip(embeddings_list, data_objects))
    ]
    # Lưu vào Qdrant (upsert)
    qdrant.upsert(collection_name="rag2", points=points)
    print("Data successfully upserted to Qdrant.")

# Hàm chính để xử lý và lưu trữ
def main():
    # Bước 1: Load dữ liệu
    data_objects = load_data("./store/parsed_data.pkl", "./store/data.json")
    # Chuẩn hóa các đối tượng thành chuỗi văn bản cho embedding
    formatted_texts = [format_for_embedding(obj) for obj in data_objects]

    # Tạo embeddings cho các văn bản đã chuẩn hóa
    embeddings_list = generate_embeddings(formatted_texts)

    # Lưu embeddings vào Qdrant
    save_to_qdrant(embeddings_list, data_objects, qdrant_url, qdrant_api_key)
    print("Dữ liệu đã được lưu vào Qdrant thành công.")

if __name__ == "__main__":
    # llama_parser = llama_document_parser(parsing_ins=ins)
    # # llamaparse to extract documents
    # json_list = llama_parser.document_processing_llamaparse(file_name="./store/wp2_hdr_j1_181031.pdf",
    #                             image_output_folder="images_output")
    # qdrant = create_vector_database()
    main()
    # print(json_list)