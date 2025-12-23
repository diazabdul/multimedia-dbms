"""
Main Routes - Serves the frontend
"""
from flask import Blueprint, render_template, send_from_directory, current_app
import os

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Serve the main application page"""
    return send_from_directory(current_app.static_folder, 'index.html')


@main_bp.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'message': 'Multimedia DBMS is running'}
