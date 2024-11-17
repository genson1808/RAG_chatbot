from rag.load_documents import PdfLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
import os
from dotenv import load_dotenv
load_dotenv()

class VectorDB:
    def __init__(self, qdrant_url = None, qdrant_api_key = None, loader: PdfLoader = None, dir_store = "../../store", embeddings=None)-> None:
        if qdrant_api_key is not None:
            self.qdrant_api_key = qdrant_api_key
        else:
            self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url is not None:
            self.qdrant_url = qdrant_url
        else:
            self.qdrant_url = os.getenv("QDRANT_URL")

        self.loader = loader
        self.dir_store = dir_store
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = FastEmbedEmbeddings()

# Create vector database
    def load_data_into_vector_db(self, chunk_size = 2000, chunk_overlap = 100, collection = "RAG"):
        """
        Creates a vector database using document loaders and embeddings.

        This function loads urls,
        splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
        and finally persists the embeddings into a Chroma vector database.

        """
        # Call the function to either load or parse the data
        llama_parse_documents = self.loader.load()
        print(llama_parse_documents[1].text[:100])

        output_file = self.dir_store + '/output.md'

        with open(output_file, 'a') as f:  # Open the file in append mode ('a')
            for doc in llama_parse_documents:
                f.write(doc.text + '\n')

        loader = DirectoryLoader(self.dir_store, glob="**/*.md", show_progress=True)
        documents = loader.load()
        # Split loaded documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)

        #len(docs)
        #docs[0]

        # Create and persist a Chroma vector database from the chunked documents
        qdrant = Qdrant.from_documents(
            documents=docs,
            embedding=self.embeddings,
            url=self.qdrant_url,
            collection_name=collection,
            api_key=self.qdrant_api_key,
            prefer_grpc=True,
        )

        #query it
        #query = "what is the agend of Financial Statements for 2022 ?"
        #found_doc = qdrant.similarity_search(query, k=3)
        #print(found_doc[0][:100])

        print('Vector DB created successfully !')