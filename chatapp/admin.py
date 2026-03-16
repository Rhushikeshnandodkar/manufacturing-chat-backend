from django.contrib import admin
from .models import PDFDocument, Chat, Message


@admin.register(PDFDocument)
class PDFDocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "file", "uploaded_at")
    list_filter = ("uploaded_at",)
    search_fields = ("title",)


@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ("title", "document", "created_at", "updated_at")
    list_filter = ("created_at",)
    search_fields = ("title",)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("chat", "role", "content_preview", "created_at")
    list_filter = ("role", "created_at")

    def content_preview(self, obj):
        return (obj.content or "")[:60] + "..." if len(obj.content or "") > 60 else (obj.content or "")

    content_preview.short_description = "Content"
