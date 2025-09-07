from .predictions import router as predictions_router
from .health import router as health_router  
from .admin import router as admin_router
from .webhooks import router as webhooks_router

__all__ = ["predictions_router", "health_router", "admin_router", "webhooks_router"]