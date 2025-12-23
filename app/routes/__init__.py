"""
Routes Package
"""
from app.routes.main import main_bp
from app.routes.upload import upload_bp
from app.routes.search import search_bp
from app.routes.media import media_bp
from app.routes.feature_routes import feature_bp

__all__ = ['main_bp', 'upload_bp', 'search_bp', 'media_bp', 'feature_bp']
