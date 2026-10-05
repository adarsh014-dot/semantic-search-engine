"""Application errors. Each carries an HTTP status, a machine-readable code and a
user-safe message. Raw exceptions are logged server-side, never shown to the user."""


class AppError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class DocumentProcessingError(Exception):
    """Raised when a file cannot be read/extracted. Message is safe to show to users."""


class EmbeddingError(Exception):
    """Raised when the embedding model cannot be loaded or fails."""


class VectorStoreError(Exception):
    """Raised for FAISS / metadata problems."""
