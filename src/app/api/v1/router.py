from fastapi import APIRouter

from src.app.api.v1.routes.auth import router as auth_router
from src.app.api.v1.routes.alerts import router as alerts_router
from src.app.api.v1.routes.comments import router as comments_router
from src.app.api.v1.routes.feature_config import router as feature_config_router
from src.app.api.v1.routes.models import router as models_router
from src.app.api.v1.routes.posts import router as posts_router
from src.app.api.v1.routes.reports import router as moderation_router
from src.app.api.v1.routes.users import router as users_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(posts_router)
api_router.include_router(comments_router)
api_router.include_router(moderation_router)
api_router.include_router(alerts_router)
api_router.include_router(feature_config_router)
api_router.include_router(models_router)
api_router.include_router(users_router)
