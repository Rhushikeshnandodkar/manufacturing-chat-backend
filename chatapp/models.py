from django.db import models


class PDFDocument(models.Model):
    """Stores uploaded PDF files and extracted text for use as chat context."""
    title = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to="pdfs/%Y/%m/%d/")
    extracted_text = models.TextField(blank=True, help_text="Text extracted from PDF for AI context")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.title or self.file.name


class LinkDocument(models.Model):
    """
    Stores scraped web pages (URL + extracted text) that are indexed in Pinecone
    alongside PDF content so the assistant can search across both.
    """

    url = models.URLField(max_length=500, unique=True)
    title = models.CharField(max_length=255, blank=True)
    extracted_text = models.TextField(
        blank=True,
        help_text="Up to ~500 words of cleaned text extracted from the web page",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title or self.url


class Chat(models.Model):
    """A chat session; can be linked to a PDF document for context."""
    title = models.CharField(max_length=255, default="New Chat")
    document = models.ForeignKey(
        PDFDocument,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chats",
        help_text="Optional PDF used as data/context for this chat",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title


class Message(models.Model):
    """A single message in a chat (user question or AI reply)."""
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=20, choices=Role.choices)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
