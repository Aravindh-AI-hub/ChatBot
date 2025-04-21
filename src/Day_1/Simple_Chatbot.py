import os 
import re
from groq import Groq 
from PyPDF2 import PdfReader #Pdf extraction
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma #VECTOR DB
from langchain.text_splitter import RecursiveCharacterTextSplitter #MAKEING CHUNKING 
from langchain_huggingface import HuggingFaceEmbeddings # TEXT EMBEDDING (NUMERICAL REPRESENTATION)

load_dotenv()

#Load Environment Variables
grop_api_key = os.environ.get("GROQ_API_KEY","")
llm_model = "llama-3.3-70b-versatile"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

def LLM(UserInput, RagContent):
    Client = Groq(api_key = grop_api_key)
    llm = Client.chat.completions.create(
        messages = [
            {
                "role":"user",
                "content":f"{UserInput}"
            },
            {
                "role":"system",
                "content":f"you are the helpfull assistant. you should extract content from the given context: {RagContent} should response user queries only from the given context. if the answer is not in the context, say i'i dont know' .avoide using any other information."
            }
        ],
        model=llm_model
    )
    return llm.choices[0].message.content



def Pdf_Reader_(pdf_file):
    Extracted_text = PdfReader(pdf_file)
    raw_text = ""
    for each_page in Extracted_text.pages:
        Text = each_page.extract_text() #Extract overll content from teh pdf file
        raw_text += Text
    return raw_text


def preprocessing(raw_text):
    cleaned_text = re.sub(r'\s+', ' ', raw_text) #Remove extra spaces
    return cleaned_text.strip()

# print(preprocessing(Pdf_Reader("QA.pdf"))) #Preprocessing the text) => 
def CHunk_splitter(Extractedtext):
    text_spliterr = RecursiveCharacterTextSplitter(chunk_size=5,chunk_overlap=2,)
    chuks = text_spliterr.split_text(Extractedtext)
    print("no of chunk :",len(chuks))
    return chuks

def VectorDatabase():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model) #Embedding model
    VectorDatabse_ = Chroma(embedding_function=embeddings, persist_directory="chroma_db") #Chroma vector database
    return VectorDatabse_
    VectorDatabase_ = VectorDatabase() #Creating the vector database
VectorDatabase_ = VectorDatabase()


def Vecotor_Store(chunk_data):
    for chunk in chunk_data:
        doc = Document(page_content=chunk, metadata={})
        VectorDatabase_.add_documents([doc])
    VectorDatabase_.persist() #Persist the data in the vector database

def Reteriver(UserInput):
    reteriver = VectorDatabase_.as_retriever(search_kwargs={'k':3})
    revertiver_doc = reteriver.invoke(UserInput)
    similiarcontent = list(set([doc.page_content for doc in revertiver_doc]))
    return similiarcontent


def main():
    while True:
        raw_text = Pdf_Reader_('QA.pdf')
        Cleaned_content = preprocessing(raw_text)
        chunk_data = CHunk_splitter(Cleaned_content)


        #Store vectore store 
        Vecotor_Store(chunk_data)

        UserQuery = input("You: ")
        Similiar_content = Reteriver(UserQuery)
        # print(Similiar_content)
        ModelResponse =  LLM(UserQuery, Similiar_content)
        print(ModelResponse)


main()
