# Multimedia Database Management System (DBMS)

A Flask-based multimedia database management system that supports image, audio, and video files with advanced feature extraction and similarity search capabilities using PostgreSQL and pgvector.

## Features

- 📸 **Image Processing**: Color histogram, texture (LBP), and deep features (MobileNetV2)
- 🎵 **Audio Processing**: MFCC, spectral features, and waveform statistics
- 🎬 **Video Processing**: Keyframe analysis, motion features, and scene statistics
- 🔍 **Similarity Search**: Query by example (QBE), metadata search, and hybrid search
- 📊 **Distance Metrics**: Euclidean, Manhattan, and K-Nearest Neighbors (KNN)
- 💾 **Vector Database**: PostgreSQL with pgvector extension for efficient similarity search

## Prerequisites

Before installing, ensure you have the following:

- **Python**: 3.9 or higher
- **PostgreSQL**: 15.x or higher
- **pgvector Extension**: Version 0.5.1 or higher
- **pip**: Python package installer
- **Git**: (Optional) For cloning the repository

### System Dependencies

#### Windows
- Install [PostgreSQL](https://www.postgresql.org/download/windows/)
- Install [Python](https://www.python.org/downloads/)

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib python3 python3-pip python3-venv
```

#### macOS
```bash
brew install postgresql python3
```

## Installation

### 1. Clone or Download the Project

```bash
git clone https://github.com/diazabdul/multimedia-dbms.git
```

Or download and extract the ZIP file to your desired location.

### 2. Create Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with system Python packages.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Installation may take several minutes as it downloads PyTorch (CPU version), librosa, OpenCV, and other dependencies.

### 4. Install PostgreSQL pgvector Extension

#### Option A: Using SQL (Recommended)
Connect to your PostgreSQL server and run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Option B: Manual Installation
Follow the [pgvector installation guide](https://github.com/pgvector/pgvector#installation) for your operating system.

### 5. Configure Environment Variables

Copy the example environment file and configure it:

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

Edit the `.env` file and update the following values:

```ini
# Database Configuration
DATABASE_URL=postgresql://your_username:your_password@localhost:5432/multimedia_db

# Flask Configuration
SECRET_KEY=your-secret-key-change-in-production
FLASK_ENV=development

# Redis (Optional - for caching)
REDIS_URL=redis://localhost:6379/0

# Upload Settings
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=104857600  # 100MB max file size
```

> **Important**: Replace `your_username`, `your_password`, and `your-secret-key-change-in-production` with your actual values.

### 6. Database Setup

You have **two options** for setting up the database:

#### **Option A: Empty Database (Schema Only)**

Use this option if you want to start with a fresh, empty database.

1. Create the database:
```bash
# Windows (PowerShell)
psql -U postgres -c "CREATE DATABASE multimedia_db;"

# Linux/macOS
sudo -u postgres psql -c "CREATE DATABASE multimedia_db;"
```

2. Import the schema:
```bash
# Windows (PowerShell)
psql -U postgres -d multimedia_db -f database\schema.sql

# Linux/macOS
sudo -u postgres psql -d multimedia_db -f database/schema.sql
```

#### **Option B: Production Database with Sample Data**

Use this option if you want to start with pre-populated sample data (images, audio, video) and extracted features.

> ⚠️ **Warning**: This file is approximately **36 MB** and contains sample media records with pre-extracted features.

1. Create the database:
```bash
# Windows (PowerShell)
psql -U postgres -c "CREATE DATABASE multimedia_db;"

# Linux/macOS
sudo -u postgres psql -c "CREATE DATABASE multimedia_db;"
```

2. Import the production database:
```bash
# Windows (PowerShell)
psql -U postgres -d multimedia_db -f mmdb_production.sql

# Linux/macOS
sudo -u postgres psql -d multimedia_db -f mmdb_production.sql
```

> **Note**: The production database includes the pgvector extension, schema, and sample data. This process may take a few minutes.

## Running the Application

### 1. Start the Development Server

```bash
python run.py
```

The application will start on `http://localhost:5000`

### 2. Access the Application

Open your web browser and navigate to:

```
http://localhost:5000
```

## Usage

### Uploading Media

1. Click the **"Upload"** tab
2. Select your media file (image, audio, or video)
3. Fill in optional metadata (title, description, tags)
4. Click **"Upload"** button
5. The system will automatically extract features and store them in the database

### Searching for Similar Media

#### Query by Example (QBE)
1. Click the **"Search"** tab
2. Upload a media file to use as a query
3. Select the distance metric (Euclidean or Manhattan)
4. Set the number of results (K)
5. Click **"Search"** to find similar media

#### Query by Metadata
1. Click the **"Search"** tab
2. Select **"Metadata Search"** mode
3. Enter search terms (title, description, or tags)
4. View matching results

#### Hybrid Search
1. Combines both similarity search and metadata filtering
2. Upload a query media file
3. Add metadata filters (optional)
4. Get results that match both visual/audio similarity and metadata criteria

### Browse Media Gallery

1. Click the **"Browse"** tab
2. Filter by media type (All, Images, Audio, Video)
3. Click on any media to view details, play audio/video, or find similar items

## Project Structure

```
Tubes/
├── app/                    # Flask application package
│   ├── models/            # Database models
│   ├── routes/            # API routes
│   ├── services/          # Business logic (feature extraction, search)
│   └── utils/             # Utility functions
├── database/              # Database schema
│   └── schema.sql         # Database schema (empty)
├── static/                # Frontend static files
│   ├── css/              # Stylesheets
│   ├── js/               # JavaScript files
│   └── index.html        # Main HTML page
├── uploads/               # Uploaded media files
├── .env.example           # Environment variables template
├── requirements.txt       # Python dependencies
├── run.py                # Application entry point
└── mmdb_production.sql   # Production database with sample data
```

## Troubleshooting

### Common Issues

#### 1. `psycopg2` Installation Error
**Error**: `Error: pg_config executable not found`

**Solution**:
- Windows: Install PostgreSQL development files
- Linux: `sudo apt install libpq-dev python3-dev`
- macOS: `brew install postgresql`

#### 2. pgvector Extension Not Found
**Error**: `extension "vector" does not exist`

**Solution**: Install pgvector extension following the [installation guide](https://github.com/pgvector/pgvector#installation)

#### 3. Database Connection Error
**Error**: `could not connect to server`

**Solution**:
- Ensure PostgreSQL is running: `sudo systemctl status postgresql` (Linux)
- Check DATABASE_URL in `.env` file
- Verify username and password

#### 4. Port Already in Use
**Error**: `Address already in use`

**Solution**: Change the port in `run.py` or stop the application using port 5000

#### 5. Out of Memory During Feature Extraction
**Error**: Memory error during video processing

**Solution**: 
- Reduce `MOBILENET_BATCH_SIZE` in `.env`
- Reduce `VIDEO_THUMBNAIL_COUNT` in `.env`
- Use smaller video files during testing

## System Requirements

### Minimum
- **CPU**: Dual-core processor
- **RAM**: 4 GB
- **Storage**: 10 GB free space
- **OS**: Windows 10, Ubuntu 20.04, or macOS 11+

### Recommended
- **CPU**: Quad-core processor or higher
- **RAM**: 8 GB or more
- **Storage**: 50 GB free space (for media files)
- **OS**: Windows 11, Ubuntu 22.04, or macOS 12+

## Tips for Best Performance

1. **Use CPU-optimized PyTorch**: The requirements.txt already uses CPU-only PyTorch for faster installation
2. **Enable Vector Indexes**: After populating data, create IVFFlat indexes for faster similarity search (see `database/schema.sql` comments)
3. **Batch Processing**: When uploading multiple files, consider using the batch upload feature
4. **File Formats**: Use optimized formats (JPEG for images, MP3 for audio, MP4 with H.264 for video)

## Development

### Running Tests
```bash
pytest tests/
```

### Production Deployment
For production deployment (e.g., with aaPanel):

1. Set `FLASK_ENV=production` in `.env`
2. Use a production WSGI server (gunicorn is already in requirements.txt)
3. Set a strong `SECRET_KEY`
4. Configure proper database backups
5. Use HTTPS with SSL certificates

## License

This project is developed for educational purposes as part of the Multimedia Data Processing & Management course.

## Support

For issues, questions, or contributions, please contact the development team or refer to the project documentation.

---

**Happy Multimedia Database Management! 🎨🎵🎬**
