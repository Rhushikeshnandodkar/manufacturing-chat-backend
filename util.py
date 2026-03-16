import os
import time # Needed for waiting during Pinecone index operations
from typing import List, Dict
from fastapi import FastAPI, HTTPException, Request, status # Removed Depends
from pydantic import BaseModel
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone # Corrected import for clarity
from pinecone import Pinecone, ServerlessSpec # Import ServerlessSpec for index creation

# Load environment variables from a .env file
load_dotenv()

# --- Configuration and Initialization ---
# Retrieve API keys and other configurations from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION")
INDEX_NAME = os.getenv("INDEX_NAME")

# --- Initialize Pinecone Client ---
# The Pinecone object is the main entry point for interacting with your indexes.
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define embedding dimension for 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384 # This MUST match the output dimension of "all-MiniLM-L6-v2"

# Print the index name for debugging/confirmation
print(f"Configured Pinecone Index Name: {INDEX_NAME}")

# --- Pinecone Index Management (Robust Creation/Recreation) ---
# Check if the index exists. If it does, delete it and wait for deletion to complete.
# This ensures that if the index was previously created with a different dimension,
# or is in a "terminating" state, we can create it cleanly with the correct settings.
if INDEX_NAME in pc.list_indexes():
    print(f"Existing Pinecone index '{INDEX_NAME}' found. Deleting it...")
    pc.delete_index(INDEX_NAME)
    # Wait for the index to be fully deleted before attempting to recreate it
    while INDEX_NAME in pc.list_indexes():
        print(f"Waiting for index '{INDEX_NAME}' to finish deleting (check again in 5s)...")
        time.sleep(5) # Wait for 5 seconds before checking again
    print(f"Index '{INDEX_NAME}' successfully deleted.")
else:
    print(f"Pinecone index '{INDEX_NAME}' does not exist. Proceeding with creation.")

# Create the Pinecone index with the correct dimension matching all-MiniLM-L6-v2 (384)
print(f"Creating Pinecone index: '{INDEX_NAME}' with dimension {EMBEDDING_DIM}...")
try:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM, # Explicitly set dimension to 384 to match your embedding model
        metric="cosine", # Use cosine similarity, consistent with sentence-transformers
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION) # Use ServerlessSpec
    )
    print(f"Pinecone index '{INDEX_NAME}' created successfully with dimension {EMBEDDING_DIM}.")
except Exception as e:
    # This catch block is for unexpected errors during creation AFTER deletion/wait.
    # The primary "ALREADY_EXISTS" should be handled by the waiting loop.
    print(f"Error creating Pinecone index '{INDEX_NAME}': {e}")
    raise # Re-raise the exception if index creation truly failed

# --- Langchain and Model Settings ---
# Initialize HuggingFace embeddings for text processing (outputs 384-dim vectors)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Groq LLM
llm = ChatGroq(model_name="llama3-70b-8192", api_key=GROQ_API_KEY)

# Initialize Sentence Transformer model for video similarity (outputs 384-dim vectors)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Dictionary to store conversation memory for different chat sessions
chat_memories: Dict[str, ConversationBufferMemory] = {}

# Initialize FastAPI application
app = FastAPI()

# --- Prompt Template for RAG ---
instruction_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are a helpful assistant trained on product manuals.
Based on the manual, answer the user's question in step-by-step format.

User's Question: {question}

{context}

Provide a step-by-step guide in numbered format.
"""
)

# --- Pydantic Models for Request Body Validation ---
class Video(BaseModel):
    description: str
    video_link: str

class SuggestRequest(BaseModel):
    query: str
    videos: List[Video]

class StoreRequest(BaseModel):
    name: str
    unique_id: str
    text: str

class AskRequest(BaseModel):
    name: str
    unique_id: str
    question: str

# --- API Endpoints (No API Key Authentication) ---

@app.post("/suggest-videos")
async def suggest_videos(request: SuggestRequest):
    """
    Suggests relevant videos based on a query and a list of video descriptions.
    Calculates cosine similarity between query embedding and video description embeddings.
    """
    if not request.videos:
        return {"matched_videos": [], "message": "No videos provided."}

    query_embedding = model.encode(request.query, convert_to_tensor=True)
    video_descriptions = [v.description for v in request.videos] # Corrected list comprehension
    video_embeddings = model.encode(video_descriptions, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, video_embeddings)[0]

    matched_videos = [
        {
            "description": request.videos[i].description,
            "video_link": request.videos[i].video_link,
            "similarity": round(score.item(), 2)
        }
        for i, score in enumerate(similarities) if score.item() > 0.5
    ]

    return {"matched_videos": matched_videos} if matched_videos else {"matched_videos": [], "message": "No similar videos found."}


@app.post("/store/")
async def store_text(request_data: StoreRequest):
    """
    Stores text chunks into Pinecone after splitting them into smaller documents.
    Uses a namespace for organization based on name and unique_id.
    """
    name, unique_id, text = request_data.name, request_data.unique_id, request_data.text

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([text])

    # Store documents in Pinecone using Langchain's Pinecone integration
    LangchainPinecone.from_documents(
        documents=docs,
        embedding=embedding_model, # This model generates 384-dim embeddings
        index_name=INDEX_NAME,
        namespace=f"{name}_{unique_id}"
    )

    return {"message": "Text stored successfully in Pinecone!"}


@app.post("/ask/")
async def ask_question(request_data: AskRequest):
    """
    Retrieves relevant information from Pinecone and generates an answer using Groq LLM.
    Maintains chat history for context.
    """
    name, unique_id, question = request_data.name, request_data.unique_id, request_data.question
    
    # Initialize LangchainPinecone from an existing index for retrieval
    try:
        vector_store = LangchainPinecone.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embedding_model, # Ensure this matches the model used for storing
            namespace=f"{name}_{unique_id}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector store for '{name}_{unique_id}' not found or inaccessible: {e}"
        )

    retriever = vector_store.as_retriever()
    memory_key = f"{name}_{unique_id}"

    if memory_key not in chat_memories:
        chat_memories[memory_key] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = instruction_prompt.format(question=question, context=context)
    response = llm.invoke(prompt)

    return {"answer": response.content}