from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenObtainPairView

urlpatterns = [
    # Auth: login to get JWT
    path("login/", TokenObtainPairView.as_view(), name="token_obtain_pair"),

    # PDF management (JWT protected upload; public upload for end-users)
    path("pdf/upload/", views.upload_pdf),
    path("pdf/upload/public/", views.upload_pdf_public),
    path("pdfs/", views.list_pdfs),
    path("pdfs/<int:pk>/", views.delete_pdf),

    # Link scraping + management (JWT protected)
    path("links/scrape/", views.scrape_link),
    path("links/", views.links),
    path("links/<int:pk>/", views.delete_link),

    # (Optional) chat listing/detail still available
    path("chats/", views.chat_list),
    path("chats/<int:pk>/", views.chat_detail),

    # One-shot QA endpoint (no auth required)
    path("ask", views.ask_question),
]
