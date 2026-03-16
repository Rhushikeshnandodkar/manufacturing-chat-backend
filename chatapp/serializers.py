from rest_framework import serializers
from .models import PDFDocument, Chat, Message, LinkDocument


class PDFDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = PDFDocument
        fields = ["id", "title", "file", "extracted_text", "uploaded_at"]
        read_only_fields = ["extracted_text", "uploaded_at"]


class LinkDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = LinkDocument
        fields = ["id", "url", "title", "extracted_text", "created_at"]
        read_only_fields = ["created_at"]


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "role", "content", "created_at"]
        read_only_fields = ["role", "created_at"]


class ChatSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Chat
        fields = ["id", "title", "document", "created_at", "updated_at", "messages"]
        read_only_fields = ["created_at", "updated_at"]


class ChatListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing chats (no messages)."""
    class Meta:
        model = Chat
        fields = ["id", "title", "document", "created_at", "updated_at"]


class SendMessageSerializer(serializers.Serializer):
    """Input for sending a user message."""
    content = serializers.CharField(allow_blank=False)
