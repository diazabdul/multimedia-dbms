"""
SQLAlchemy Models for Multimedia DBMS
"""
from datetime import datetime
from pgvector.sqlalchemy import Vector
from app import db


class Media(db.Model):
    """Main media table storing file metadata"""
    __tablename__ = 'media'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    media_type = db.Column(db.String(10), nullable=False)  # 'image', 'audio', 'video'
    mime_type = db.Column(db.String(100))
    file_size = db.Column(db.BigInteger)
    
    # Metadata
    title = db.Column(db.String(255))
    description = db.Column(db.Text)
    tags = db.Column(db.ARRAY(db.String))
    
    # Media-specific metadata
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    duration = db.Column(db.Float)  # seconds for audio/video
    
    # Timestamps
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Processing status
    is_processed = db.Column(db.Boolean, default=False)
    processing_error = db.Column(db.Text)
    
    # Relationships
    image_features = db.relationship('ImageFeatures', backref='media', uselist=False, cascade='all, delete-orphan')
    audio_features = db.relationship('AudioFeatures', backref='media', uselist=False, cascade='all, delete-orphan')
    video_features = db.relationship('VideoFeatures', backref='media', uselist=False, cascade='all, delete-orphan')
    thumbnails = db.relationship('Thumbnail', backref='media', cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary for JSON response"""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'media_type': self.media_type,
            'mime_type': self.mime_type,
            'file_size': self.file_size,
            'title': self.title,
            'description': self.description,
            'tags': self.tags or [],
            'width': self.width,
            'height': self.height,
            'duration': self.duration,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_processed': self.is_processed,
            'thumbnail_url': f'/api/media/{self.id}/thumbnail' if self.thumbnails else None
        }


class ImageFeatures(db.Model):
    """Image feature vectors"""
    __tablename__ = 'image_features'
    
    id = db.Column(db.Integer, primary_key=True)
    media_id = db.Column(db.Integer, db.ForeignKey('media.id', ondelete='CASCADE'), unique=True)
    
    # Feature vectors
    color_histogram = db.Column(Vector(192))
    texture_lbp = db.Column(Vector(256))
    deep_features = db.Column(Vector(1280))
    combined_features = db.Column(Vector(1728))
    
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)


class AudioFeatures(db.Model):
    """Audio feature vectors"""
    __tablename__ = 'audio_features'
    
    id = db.Column(db.Integer, primary_key=True)
    media_id = db.Column(db.Integer, db.ForeignKey('media.id', ondelete='CASCADE'), unique=True)
    
    # Feature vectors
    mfcc_features = db.Column(Vector(39))
    spectral_features = db.Column(Vector(6))
    waveform_stats = db.Column(Vector(5))
    combined_features = db.Column(Vector(50))
    
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)


class VideoFeatures(db.Model):
    """Video feature vectors"""
    __tablename__ = 'video_features'
    
    id = db.Column(db.Integer, primary_key=True)
    media_id = db.Column(db.Integer, db.ForeignKey('media.id', ondelete='CASCADE'), unique=True)
    
    # Feature vectors
    keyframe_features = db.Column(Vector(1280))
    motion_features = db.Column(Vector(64))
    scene_stats = db.Column(Vector(10))
    combined_features = db.Column(Vector(1354))
    
    # Keyframe timestamps
    keyframe_timestamps = db.Column(db.ARRAY(db.Float))
    
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)


class Thumbnail(db.Model):
    """Media thumbnails"""
    __tablename__ = 'thumbnails'
    
    id = db.Column(db.Integer, primary_key=True)
    media_id = db.Column(db.Integer, db.ForeignKey('media.id', ondelete='CASCADE'))
    thumbnail_path = db.Column(db.String(512), nullable=False)
    thumbnail_type = db.Column(db.String(50), default='default')
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
