from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Requête envoyée à l'endpoint /chat.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question de l'utilisateur",
        example="Quels sont les tarifs ?",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre de documents à récupérer",
        example=5,
    )


class DocumentMetadata(BaseModel):
    """
    Métadonnées d'un document retourné (optionnel).
    """

    source_file: Optional[str] = None
    category: Optional[str] = None
    source_type: Optional[str] = None


class DocumentResult(BaseModel):
    """
    Document retourné par le retriever (optionnel).
    """

    content: str
    metadata: DocumentMetadata
    distance: Optional[float] = None


class ChatResponse(BaseModel):
    """
    Réponse renvoyée par l'endpoint /chat.
    """

    answer: str = Field(
        ...,
        description="Réponse générée par le système RAG",
    )

    intent: Optional[str] = Field(
        default=None,
        description="Intention détectée (tarif, niveau, inscription, general)",
    )

    route: Optional[str] = Field(
        default=None,
        description="Route utilisée (business, rag, small_talk...)",
    )

    documents: Optional[List[DocumentResult]] = Field(
        default=None,
        description="Documents utilisés pour générer la réponse (debug / transparence)",
    )

    class Config:
        extra = "ignore"