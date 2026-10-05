from fastapi import APIRouter, Depends

from backend.api.deps import get_services
from backend.container import Services
from backend.models.schemas import SearchRequest, SearchResponse

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, svc: Services = Depends(get_services)):
    """Semantic search. Declared as a sync function so FastAPI runs the (CPU-bound)
    embedding in a worker thread instead of blocking the event loop."""
    top_k = min(body.top_k, svc.settings.max_top_k)
    return svc.search.search(body.query, top_k)
