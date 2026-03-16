from rest_framework import status
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response

from .models import PDFDocument, Chat, Message, LinkDocument
from .serializers import (
    PDFDocumentSerializer,
    LinkDocumentSerializer,
    ChatSerializer,
    ChatListSerializer,
    MessageSerializer,
    SendMessageSerializer,
)
from .utils import extract_text_from_pdf

import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from rest_framework.permissions import IsAuthenticated
import requests
from bs4 import BeautifulSoup

load_dotenv()

EMBEDDING_DIM = 384


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Make sure it exists in your environment or .env file."
        )
    return value


def get_pinecone_client() -> Pinecone:
    api_key = _get_env("PINECONE_API_KEY")
    return Pinecone(api_key=api_key)


def ensure_pinecone_index():
    index_name = _get_env("INDEX_NAME")
    region = _get_env("PINECONE_REGION")

    pc = get_pinecone_client()
    print(f"Configured Pinecone Index Name: {index_name}")
    if not pc.has_index(index_name):
        print(f"Creating Pinecone index: '{index_name}' with dimension {EMBEDDING_DIM}...")
        try:
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=region),
            )
            print(f"Pinecone index '{index_name}' created successfully with dimension {EMBEDDING_DIM}.")
        except Exception as e:
            message = str(e)
            if "ALREADY_EXISTS" in message or "409" in message:
                print(f"Pinecone index '{index_name}' already exists; continuing.")
            else:
                raise

    return index_name


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm() -> ChatGroq:
    """Lazily create Groq LLM client so imports don't fail."""
    api_key = _get_env("GROQ_API_KEY")
    return ChatGroq(model_name="llama-3.3-70b-versatile", api_key=api_key)

instruction_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are a helpful assistant. Answer the user's question using ONLY the context below, which may contain excerpts from multiple uploaded PDFs. Use all relevant parts of the context from any document.

If the answer is not found in the context at all, say clearly: "This information is not available in the uploaded documents."
Do not invent information. Do not answer from only one document if the question relates to another—check all provided context.

User's question: {question}

Context from uploaded PDFs:
{context}

Answer (concise; use numbered steps if appropriate):""",
)


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([IsAuthenticated])
def upload_pdf(request):
    return _upload_pdf_impl(request)


def _upload_pdf_impl(request):
    files = request.FILES.getlist("files")
    if not files:
        single = request.FILES.get("file")
        if single:
            files = [single]

    if not files:
        return Response(
            {"error": "No file(s) provided. Use form field 'file' or 'files'."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    index_name = ensure_pinecone_index()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    created_docs = []

    for f in files:
        if not f.name.lower().endswith(".pdf"):
            return Response(
                {"error": f"Only PDF files are allowed. Invalid: {f.name}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        title = request.data.get("title", "") or f.name
        doc = PDFDocument(title=title, file=f)
        doc.extracted_text = extract_text_from_pdf(f)
        doc.save()

        docs = text_splitter.create_documents([doc.extracted_text])
        for d in docs:
            d.metadata["document_id"] = str(doc.id)
        try:
            LangchainPinecone.from_documents(
                documents=docs,
                embedding=embedding_model,
                index_name=index_name,
                namespace="default",
            )
        except Exception as e:
            doc.delete()
            return Response(
                {
                    "error": "Pinecone upload failed. Check that index exists and dimension is 384.",
                    "detail": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        created_docs.append(doc.id)

    return Response(
        {
            "message": "PDF(s) uploaded and text chunks stored in Pinecone successfully! In the Pinecone dashboard, open your index and select namespace 'default' to see the vectors.",
            "document_ids": created_docs,
        },
        status=status.HTTP_201_CREATED,
    )


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_pdf_public(request):
    return _upload_pdf_impl(request)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def list_pdfs(request):
    docs = PDFDocument.objects.all()
    serializer = PDFDocumentSerializer(docs, many=True)
    return Response(serializer.data)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def delete_pdf(request, pk):
    try:
        pdf_doc = PDFDocument.objects.get(pk=pk)
    except PDFDocument.DoesNotExist:
        return Response(
            {"error": "PDF document not found."},
            status=status.HTTP_404_NOT_FOUND,
        )

    try:
        index_name = ensure_pinecone_index()
        pc = get_pinecone_client()
        index = pc.Index(index_name)
        index.delete(
            filter={"document_id": {"$eq": str(pdf_doc.id)}},
            namespace="default",
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to delete from Pinecone: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    try:
        if pdf_doc.file:
            pdf_doc.file.delete(save=False)
        pdf_doc.delete()
    except Exception as e:
        return Response(
            {"error": f"Failed to delete file or record: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def scrape_link(request):
    """
    Fetches a URL, extracts readable text, trims it to ~5000 words and returns it
    so the admin can review/edit before saving.
    """
    url = request.data.get("url")
    if not url:
        return Response(
            {"error": "URL is required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return Response(
            {"error": f"Failed to fetch URL: {e}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Normalize whitespace and limit to ~5000 words
    words = " ".join(text.split()).split()
    max_words = 5000
    trimmed = " ".join(words[:max_words])

    title = url
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    return Response({"title": title, "text": trimmed})


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated])
def links(request):
    """
    GET: list all scraped links that are indexed in Pinecone.
    POST: save (or update) a link + text into DB and Pinecone (namespace 'default').
    """
    if request.method == "GET":
        items = LinkDocument.objects.all()
        serializer = LinkDocumentSerializer(items, many=True)
        return Response(serializer.data)

    url = request.data.get("url")
    text = request.data.get("text")
    title = request.data.get("title") or url

    if not url or not text:
        return Response(
            {"error": "Both 'url' and 'text' are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    index_name = ensure_pinecone_index()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([text])

    # Either update existing record for URL or create new one
    link_doc, _created = LinkDocument.objects.update_or_create(
        url=url,
        defaults={"title": title, "extracted_text": text},
    )

    for d in docs:
        d.metadata["link_id"] = str(link_doc.id)
        d.metadata["source"] = "link"
        d.metadata["url"] = url

    try:
        LangchainPinecone.from_documents(
            documents=docs,
            embedding=embedding_model,
            index_name=index_name,
            namespace="default",
        )
    except Exception as e:
        # If Pinecone write fails, roll back DB changes so list stays consistent
        if link_doc.pk:
            link_doc.delete()
        return Response(
            {
                "error": "Failed to store link text in Pinecone.",
                "detail": str(e),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    serializer = LinkDocumentSerializer(link_doc)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def delete_link(request, pk):
    try:
        link_doc = LinkDocument.objects.get(pk=pk)
    except LinkDocument.DoesNotExist:
        return Response(
            {"error": "Link not found."},
            status=status.HTTP_404_NOT_FOUND,
        )

    try:
        index_name = ensure_pinecone_index()
        pc = get_pinecone_client()
        index = pc.Index(index_name)
        index.delete(
            filter={"link_id": {"$eq": str(link_doc.id)}},
            namespace="default",
        )
    except Exception as e:
        return Response(
            {"error": f"Failed to delete link vectors from Pinecone: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    link_doc.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


# ---------- Chat ----------

@api_view(["GET", "POST"])
def chat_list(request):
    if request.method == "GET":
        chats = Chat.objects.all()
        serializer = ChatListSerializer(chats, many=True)
        return Response(serializer.data)

    document_id = request.data.get("document_id")
    title = request.data.get("title", "New Chat")
    document = None
    if document_id:
        try:
            document = PDFDocument.objects.get(pk=document_id)
        except PDFDocument.DoesNotExist:
            return Response(
                {"error": "PDF document not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
    chat = Chat.objects.create(title=title, document=document)
    serializer = ChatSerializer(chat)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(["GET"])
def chat_detail(request, pk):
    try:
        chat = Chat.objects.get(pk=pk)
    except Chat.DoesNotExist:
        return Response(
            {"error": "Chat not found."},
            status=status.HTTP_404_NOT_FOUND,
        )
    serializer = ChatSerializer(chat)
    return Response(serializer.data)

@api_view(["POST"])
def ask_question(request):
    serializer = SendMessageSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    question = serializer.validated_data["content"]

    try:
        index_name = ensure_pinecone_index()
        vector_store = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=embedding_model,
            namespace="default",
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 12})
        relevant_docs = retriever.invoke(question)
        context = "\n".join(doc.page_content for doc in relevant_docs)

        prompt = instruction_prompt.format(question=question, context=context)
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"Sorry, I could not generate an answer: {e}"

    return Response(
        {
            "answer": answer
        },
        status=status.HTTP_200_OK,
    )
