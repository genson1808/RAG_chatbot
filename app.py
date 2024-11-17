# import required dependencies
# https://docs.chainlit.io/integrations/langchain
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from langchain_community.chat_models import ChatOllama


import chainlit as cl
from langchain.chains import RetrievalQA

from src.base.llm import SingletonChatLLM

# bring in our GROQ_API_KEY
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# custom_prompt_template = """あなたは有用なアシスタントです。以下はシステムに関する文書のコンテキストです。
#     コンテキスト:
#     {context}

#     上記の情報に基づいて、以下の質問に答えてください：
#     質問: {question}

#     答え:
#     もし情報が不足している場合は、知らないと答えてください。無理に答えを作らないでください。
#     """

# custom_prompt_template = """あなたは有用なアシスタントです。以下はシステムに関する文書のコンテキストです。
#     コンテキスト:
#     {context}

#     上記の情報に基づいて、以下の質問に答えてください：
#     質問: {question}

#     以下のルールを守って回答してください：
#     1. 質問が日本語の場合は必ず日本語で回答してください。
#     2. 質問が他の言語でされた場合は、その言語で回答してください。
#     3. 十分な情報がない場合は、「申し訳ありませんが、分かりません」と回答してください。無理に答えを作らないでください。

#     答え:
#     """


custom_prompt_template = """You are a helpful assistant, You must use japanese to answer the question, conversing with a user about the subjects contained in a set of documents.
Use the information from the DOCUMENTS section to provide accurate answers. If unsure or if the answer
isn't found in the DOCUMENTS section, simply state that you don't know the answer.

Documents:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


# chat_model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
# chat_model = ChatGroq(temperature=0, model_name="llama3-groq-70b-8192-tool-use-preview")
#chat_model = ChatGroq(temperature=0, model_name="Llama2-70b-4096")
#chat_model = ChatOllama(model="llama2", request_timeout=30.0)

chat_model = SingletonChatLLM(llm_name='CHAT_GROQ').get_llm()

client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url,)

from langchain_community.document_compressors import JinaRerank
from langchain.retrievers import ContextualCompressionRetriever

os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")
def retrieval_qa_chain(llm, prompt, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
    compressor = JinaRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain


def qa_bot():
    embeddings = FastEmbedEmbeddings()
    vectorstore = Qdrant(client=client, embeddings=embeddings, collection_name="rag2")
    llm = chat_model
    qa_prompt=set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)
    return qa


@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "こんにちは！ご利用いただきありがとうございます。質問があれば、何でもお気軽にお尋ねください。"
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the user's session,
    sets up a callback handler for the bot's response, and executes the bot's
    call method with the given message and callback. The bot's answer and source
    documents are then extracted from the response.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    # res=await chain.acall(message, callbacks=[cb])
    res = await chain.acall(message.content, callbacks=[cb])
    #print(f"response: {res}")
    answer = res["result"]
    #answer = answer.replace(".", ".\n")
    source_documents = res["source_documents"]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()