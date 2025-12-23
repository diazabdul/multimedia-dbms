-- Multimedia DBMS Schema
-- PostgreSQL 15+ with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Media types enum
CREATE TYPE media_type AS ENUM ('image', 'audio', 'video');

-- Main media table
CREATE TABLE IF NOT EXISTS media (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    media_type media_type NOT NULL,
    mime_type VARCHAR(100),
    file_size BIGINT,
    
    -- Metadata
    title VARCHAR(255),
    description TEXT,
    tags TEXT[],
    
    -- Media-specific metadata
    width INTEGER,
    height INTEGER,
    duration FLOAT,  -- seconds for audio/video
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Processing status
    is_processed BOOLEAN DEFAULT FALSE,
    processing_error TEXT
);

-- Image features table
CREATE TABLE IF NOT EXISTS image_features (
    id SERIAL PRIMARY KEY,
    media_id INTEGER REFERENCES media(id) ON DELETE CASCADE,
    
    -- Color histogram (HSV, 192 dimensions)
    color_histogram vector(192),
    
    -- Texture features (LBP, 256 dimensions)
    texture_lbp vector(256),
    
    -- Deep features (MobileNetV2, 1280 dimensions)
    deep_features vector(1280),
    
    -- Combined normalized vector for fast search
    combined_features vector(1728),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(media_id)
);

-- Audio features table
CREATE TABLE IF NOT EXISTS audio_features (
    id SERIAL PRIMARY KEY,
    media_id INTEGER REFERENCES media(id) ON DELETE CASCADE,
    
    -- MFCC features (39 dimensions: 13 coefficients + delta + delta-delta)
    mfcc_features vector(39),
    
    -- Spectral features (6 dimensions)
    spectral_features vector(6),
    
    -- Waveform statistics (5 dimensions)
    waveform_stats vector(5),
    
    -- Combined normalized vector
    combined_features vector(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(media_id)
);

-- Video features table
CREATE TABLE IF NOT EXISTS video_features (
    id SERIAL PRIMARY KEY,
    media_id INTEGER REFERENCES media(id) ON DELETE CASCADE,
    
    -- Aggregated keyframe features (average of MobileNetV2 features)
    keyframe_features vector(1280),
    
    -- Motion features (optical flow histogram, 64 dimensions)
    motion_features vector(64),
    
    -- Scene statistics (10 dimensions)
    scene_stats vector(10),
    
    -- Combined normalized vector
    combined_features vector(1354),
    
    -- Keyframe data stored as JSON
    keyframe_timestamps FLOAT[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(media_id)
);

-- Thumbnails table
CREATE TABLE IF NOT EXISTS thumbnails (
    id SERIAL PRIMARY KEY,
    media_id INTEGER REFERENCES media(id) ON DELETE CASCADE,
    thumbnail_path VARCHAR(512) NOT NULL,
    thumbnail_type VARCHAR(50) DEFAULT 'default',  -- 'default', 'frame_1', 'frame_2', etc.
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type);
CREATE INDEX IF NOT EXISTS idx_media_tags ON media USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_media_created ON media(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_media_title ON media(title);

-- Vector indexes using IVFFlat for approximate nearest neighbor search
-- These should be created after populating data for best results
-- CREATE INDEX ON image_features USING ivfflat (combined_features vector_l2_ops) WITH (lists = 100);
-- CREATE INDEX ON audio_features USING ivfflat (combined_features vector_l2_ops) WITH (lists = 100);
-- CREATE INDEX ON video_features USING ivfflat (combined_features vector_l2_ops) WITH (lists = 100);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
CREATE TRIGGER update_media_updated_at
    BEFORE UPDATE ON media
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
