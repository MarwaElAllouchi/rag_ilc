from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_chat import router as chat_router
from app.core.config import get_settings
from app.core.logging_config import setup_logging


# ================================
# Configuration
# ================================
settings = get_settings()
setup_logging(settings.log_level)


# ================================
# App initialization
# ================================
app = FastAPI(
    title="RAG Institut Langues et Cultures",
    description="API de chatbot basée sur un système RAG",
    version="1.0.0",
)


# ================================
# CORS (⚠️ à adapter en prod)
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👉 remplacer en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# Routers
# ================================
app.include_router(chat_router, prefix="/api/v1")


# ================================
# Health check (très important)
# ================================
@app.get("/health", tags=["health"])
def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
    }


# ================================
# Root endpoint
# ================================
@app.get("/", tags=["root"])
def root():
    return {
        "message": "API RAG ILC en cours d'exécution",
        "version": "1.0.0",
    }