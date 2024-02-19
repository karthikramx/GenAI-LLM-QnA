# document loading into langchain
# all formats - langchain can download documenet from various resources
# you can also load documents from public services such as wikipedia

import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
import warnings
warnings.filterwarnings('ignore')
chunks = None

# from langchain.document_loaders import 

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == 'pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    else:
        print("Document format is not supported")
        return None
    data = loader.load()
    return data

def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain_community.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

def chunk_data(data):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum(len(enc.encode(page.page_content)) for page in texts)
    print(f'Total Tokens:{total_tokens}')
    print(f'Embedding cost in USD:{total_tokens/1000*0.0004:.6f}')


def delete_pinecone_index(index_name='all'):
    from pinecone import Pinecone
    pinecone = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print("Deleting all indexes")
        for index in indexes:
            pinecone.delete_index(index['name'])
        print("Done deleting all indexes")
    else:
        print(f"Deleting index: {index_name} ...", end='')
        pinecone.delete_index(index_name)
        print(f"Done deleting {index_name}")


def insert_or_fetch_embeddings(index_name):
    # from langchain.vectorstores import pinecone as Pinecone
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import Pinecone as langPineCone
    from pinecone import Pinecone
    pinecone = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

    embeddings = OpenAIEmbeddings()
    pinecone = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'),)
    list_of_indexes = [index['name'] for index in pinecone.list_indexes()]

    if index_name in list_of_indexes:
        print(f'Index {index_name} already exists. Loading embeddings ...',end='')
        vector_store = langPineCone.from_existing_index(index_name=index_name, embedding=embeddings)
    else:
        print_embedding_cost(chunks)
        print(f'Creating index {index_name} and embedddings ...', end='')
        pinecone.create_index(index_name, 
                                dimension=1536, 
                                metric='cosine',
                                spec= {'serverless': {'cloud': 'aws', 'region': 'us-west-2'}})
        vector_store = langPineCone.from_documents(chunks, 
                                                    embeddings, 
                                                    index_name=index_name,
                                                    )
    return vector_store

def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity',
                                         search_kwargs={'k':3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.invoke(q)
    return answer


topic = input("Topic you would like to research about:").lower()

data = load_from_wikipedia(topic)
chunks = chunk_data(data)
index_name = topic
# delete_pinecone_index(index_name=index_name)
vector_store = insert_or_fetch_embeddings(index_name=index_name)


while True:
    print("\n"+"-"*50)
    question = input(f"\nEnter your question about {topic}, enter q to exit:")
    if len(question) == 0:
        break
    answer = ask_and_get_answer(vector_store=vector_store, q=question)
    print("\nAnswer:",answer["result"])
    print("\n"+"-"*50)

delete_pinecone_index()



