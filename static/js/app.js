/**
 * Multimedia DBMS - Main Application JavaScript
 */
onload = () => {
    switchView('home');
}
const state = {
    currentView: 'dashboard',
    currentPage: 1,
    perPage: 24,
    totalPages: 1,
    selectedMedia: null,
    qbeFile: null,
    hybridFile: null,
    currentFilter: 'all',
    // Infinite scroll state
    galleryItems: [],
    galleryDisplayed: 0,
    galleryTotal: 0,
    galleryLoading: false,
    galleryInitialLoad: 26,
    galleryLoadMore: 12
};
const API_BASE = '/api';

document.addEventListener('DOMContentLoaded', () => {
    console.log('[MMDB] Initializing...');
    initNavigation();
    initUpload();
    initSearch();
    initBrowse();
    initModal();
    loadStats();
    loadRecentMedia();
});

// Global filter function - called from onclick in HTML
function filterMedia(type, btn) {
    console.log('[MMDB] Filter clicked:', type);

    // Update state
    state.currentFilter = type;
    state.currentPage = 1;

    // Update button active states
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');

    // Reload media with new filter
    loadRecentMedia();
}


function initNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            // Handle clicks on spans inside the link
            const target = e.target.closest('.nav-link');
            const page = target.dataset.page;
            if (page) switchView(page);
        });
    });
}

function switchView(pageName) {
    state.currentView = pageName;

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(item => {
        item.classList.toggle('active', item.dataset.page === pageName);
    });

    // Update pages/sections
    document.querySelectorAll('.page').forEach(page => {
        page.classList.toggle('active', page.id === `${pageName}-page`);
    });

    if (pageName === 'home') {
        // Home is effectively the gallery/browse view + stats
        loadStats();
        loadRecentMedia();
    }
}


async function loadStats() {
    console.log('[MMDB] Loading stats...');
    try {
        const res = await fetch(`${API_BASE}/search/stats`);
        const data = await res.json();
        console.log('[MMDB] Stats response:', data);

        // Backend returns: { total_media, by_type: { image, audio, video }, available_tags }
        const imageEl = document.getElementById('image-count');
        const audioEl = document.getElementById('audio-count');
        const videoEl = document.getElementById('video-count');
        const totalEl = document.getElementById('total-count');

        const imageCount = data.by_type?.image || 0;
        const audioCount = data.by_type?.audio || 0;
        const videoCount = data.by_type?.video || 0;
        const totalCount = data.total_media || 0;

        console.log('[MMDB] Setting counts:', { imageCount, audioCount, videoCount, totalCount });

        if (imageEl) imageEl.textContent = imageCount;
        if (audioEl) audioEl.textContent = audioCount;
        if (videoEl) videoEl.textContent = videoCount;
        if (totalEl) totalEl.textContent = totalCount;
    } catch (e) {
        console.error('[MMDB] Failed to load stats:', e);
    }
}


// Filter buttons functionality
function initFilter() {
    const filterBtns = document.querySelectorAll('.filter-btn');
    console.log('[MMDB] initFilter - Found', filterBtns.length, 'filter buttons');

    filterBtns.forEach(btn => {
        console.log('[MMDB] Adding listener to button:', btn.dataset.type);
        btn.addEventListener('click', () => {
            const type = btn.dataset.type;
            console.log('[MMDB] Filter button clicked:', type);
            state.currentFilter = type;
            state.currentPage = 1;

            // Update active state
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Reload media with filter
            loadRecentMedia();
        });
    });
}

// Initialize infinite scroll
function initInfiniteScroll() {
    const mediaGrid = document.getElementById('media-grid');
    if (!mediaGrid) {
        console.log('[MMDB] media-grid not found - skipping infinite scroll (not on gallery page)');
        return;
    }

    // Remove existing sentinel if any
    const existingSentinel = document.getElementById('scroll-sentinel');
    if (existingSentinel) {
        console.log('[MMDB] Removing existing sentinel');
        existingSentinel.remove();
    }

    if (!state.galleryItems || state.galleryItems.length === 0) {
        console.log('[MMDB] No gallery items loaded yet, skipping sentinel creation');
        return;
    }

    // Create sentinel element that will trigger loading more items
    const sentinel = document.createElement('div');
    sentinel.id = 'scroll-sentinel';
    sentinel.style.cssText = 'height: 1px; width: 100%; margin-top: 20px; background: transparent;';

    // Insert sentinel as the next sibling of media-grid
    if (mediaGrid.nextSibling) {
        mediaGrid.parentNode.insertBefore(sentinel, mediaGrid.nextSibling);
    } else {
        mediaGrid.parentNode.appendChild(sentinel);
    }

    console.log('[MMDB] Sentinel element created and placed');
    console.log('[MMDB] Sentinel position:', sentinel.getBoundingClientRect());

    // Create intersection observer with generous margins
    const observer = new IntersectionObserver((entries) => {
        const entry = entries[0];
        console.log('[MMDB] Sentinel intersection check:', {
            isIntersecting: entry.isIntersecting,
            intersectionRatio: entry.intersectionRatio,
            galleryLoading: state.galleryLoading,
            displayed: state.galleryDisplayed,
            total: state.galleryItems.length
        });

        if (entry.isIntersecting && !state.galleryLoading) {
            console.log('[MMDB] 🔄 Triggering loadMoreGalleryItems...');
            loadMoreGalleryItems();
        }
    }, {
        root: null,  // Use viewport as root
        rootMargin: '400px',  // Start loading 400px before reaching sentinel
        threshold: [0, 0.1, 0.5, 1]  // Multiple thresholds for better detection
    });

    observer.observe(sentinel);
    console.log('[MMDB] ✅ Infinite scroll observer attached and active');
}

// Load more items when scrolling
function loadMoreGalleryItems() {
    if (state.galleryLoading) return;
    if (state.galleryDisplayed >= state.galleryItems.length) return;

    state.galleryLoading = true;

    const container = document.getElementById('media-grid');
    if (!container) {
        state.galleryLoading = false;
        return;
    }

    // Calculate how many to display
    const startIndex = state.galleryDisplayed;
    const endIndex = Math.min(startIndex + state.galleryLoadMore, state.galleryItems.length);

    console.log(`[MMDB] Loading more: ${startIndex} to ${endIndex}`);

    // Add new items with animation
    for (let i = startIndex; i < endIndex; i++) {
        const card = createMediaCardAnimated(state.galleryItems[i], i - startIndex);
        container.insertAdjacentHTML('beforeend', card);
    }

    // Attach events to new cards only
    const newCards = container.querySelectorAll('.media-card.animate-in:not(.event-attached)');
    newCards.forEach(card => {
        card.classList.add('event-attached');
        card.addEventListener('click', () => openMediaModal(card.dataset.id));
    });

    state.galleryDisplayed = endIndex;
    state.galleryLoading = false;

    // Update scroll indicator
    updateScrollIndicator();
}

// Create animated media card
function createMediaCardAnimated(media, delay = 0) {
    const icons = {
        image: 'M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z',
        audio: 'M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3',
        video: 'M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z'
    };

    const hasThumbnail = media.thumbnail_url;
    let thumbnailHtml;

    if (hasThumbnail) {
        thumbnailHtml = `<img class="thumbnail" src="${media.thumbnail_url}" alt="${media.title || media.original_filename}" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
        <div class="thumbnail-placeholder" style="display:none;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="${icons[media.media_type] || icons.image}"/>
            </svg>
        </div>`;
    } else {
        thumbnailHtml = `<div class="thumbnail-placeholder">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="${icons[media.media_type] || icons.image}"/>
            </svg>
        </div>`;
    }

    // Add animation delay for staggered effect
    const animDelay = delay * 50; // 50ms between each card
    return `<div class="media-card animate-in" data-id="${media.id}" style="animation-delay: ${animDelay}ms;">
        ${thumbnailHtml}
        <div class="card-info">
            <div class="card-title">${media.title || media.original_filename}</div>
            <div class="card-meta">
                <span class="media-type-badge ${media.media_type}">${media.media_type}</span>
                <span>${formatFileSize(media.file_size)}</span>
            </div>
        </div>
    </div>`;
}

// Update scroll indicator
function updateScrollIndicator() {
    let indicator = document.getElementById('scroll-indicator');

    if (state.galleryDisplayed < state.galleryItems.length) {
        if (!indicator) {
            const container = document.getElementById('media-grid');
            if (container) {
                container.insertAdjacentHTML('afterend',
                    `<div id="scroll-indicator" class="scroll-indicator">
                        <span>Scroll for more</span>
                        <span class="scroll-count">${state.galleryDisplayed} / ${state.galleryItems.length}</span>
                    </div>`);
            }
        } else {
            indicator.querySelector('.scroll-count').textContent =
                `${state.galleryDisplayed} / ${state.galleryItems.length}`;
        }
    } else if (indicator) {
        indicator.remove();
    }
}

async function loadRecentMedia() {
    const container = document.getElementById('media-grid');
    if (!container) {
        console.error('[MMDB] media-grid container not found!');
        return;
    }

    container.innerHTML = '<div class="loading-spinner"></div>';

    // Reset gallery state
    state.galleryItems = [];
    state.galleryDisplayed = 0;
    state.galleryLoading = true;

    try {
        // Fetch more items for infinite scroll (up to 500)
        let url = `${API_BASE}/media?per_page=500&sort=created_at&order=desc`;
        if (state.currentFilter && state.currentFilter !== 'all') {
            url += `&type=${state.currentFilter}`;
        }

        console.log('[MMDB] loadRecentMedia - Fetching all items for infinite scroll');

        const res = await fetch(url);
        const data = await res.json();

        console.log('[MMDB] loadRecentMedia - Got', data.items?.length || 0, 'total items');

        if (data.items && data.items.length > 0) {
            state.galleryItems = data.items;
            state.galleryTotal = data.items.length;

            // Display initial batch with animation
            container.innerHTML = '';
            const initialCount = Math.min(state.galleryInitialLoad, data.items.length);

            for (let i = 0; i < initialCount; i++) {
                const card = createMediaCardAnimated(data.items[i], i);
                container.insertAdjacentHTML('beforeend', card);
            }

            state.galleryDisplayed = initialCount;

            // Attach click events
            container.querySelectorAll('.media-card').forEach(card => {
                card.classList.add('event-attached');
                card.addEventListener('click', () => openMediaModal(card.dataset.id));
            });

            const emptyState = document.getElementById('empty-state');
            if (emptyState) emptyState.style.display = 'none';

            // Update scroll indicator
            updateScrollIndicator();

            // Initialize infinite scroll AFTER items are displayed
            console.log('[MMDB] Initializing infinite scroll with', state.galleryItems.length, 'items');
            initInfiniteScroll();
        } else {
            container.innerHTML = '';
            const emptyState = document.getElementById('empty-state');
            if (emptyState) emptyState.style.display = 'block';
        }
    } catch (e) {
        console.error('Failed to load media:', e);
        container.innerHTML = '<div class="empty-state"><h3>Failed to load media</h3><p>Please check your connection</p></div>';
    }

    state.galleryLoading = false;
}


function createMediaCard(media) {
    const icons = {
        image: 'M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z',
        audio: 'M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3',
        video: 'M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z'
    };

    // Use thumbnail_url from backend (only set if thumbnails exist in DB)
    const hasThumbnail = media.thumbnail_url;

    let thumbnailHtml;
    if (hasThumbnail) {
        thumbnailHtml = `<img class="thumbnail" src="${media.thumbnail_url}" alt="${media.title || media.original_filename}" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
        <div class="thumbnail-placeholder" style="display:none;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="${icons[media.media_type] || icons.image}"/>
            </svg>
        </div>`;
    } else {
        thumbnailHtml = `<div class="thumbnail-placeholder">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="${icons[media.media_type] || icons.image}"/>
            </svg>
        </div>`;
    }

    return `<div class="media-card" data-id="${media.id}">
        ${thumbnailHtml}
        <div class="card-info">
            <div class="card-title">${media.title || media.original_filename}</div>
            <div class="card-meta">
                <span class="media-type-badge ${media.media_type}">${media.media_type}</span>
                <span>${formatFileSize(media.file_size)}</span>
            </div>
        </div>
    </div>`;
}

function attachMediaCardEvents(container) { container.querySelectorAll('.media-card').forEach(card => { card.addEventListener('click', () => openMediaModal(card.dataset.id)); }); }

function initUpload() { const zone = document.getElementById('upload-zone'); const input = document.getElementById('file-input'); const form = document.getElementById('metadata-form'); let files = []; zone.addEventListener('click', () => input.click()); zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); }); zone.addEventListener('dragleave', () => zone.classList.remove('dragover')); zone.addEventListener('drop', (e) => { e.preventDefault(); zone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); }); input.addEventListener('change', () => handleFiles(input.files)); function handleFiles(fileList) { files = Array.from(fileList); if (files.length === 0) return; zone.style.display = 'none'; form.style.display = 'block'; document.getElementById('upload-queue').innerHTML = files.map((f, i) => `<div class="selected-file"><span>${f.name}</span><button class="remove-file" data-index="${i}">×</button></div>`).join(''); document.getElementById('upload-btn').disabled = false; } window.resetUpload = function () { files = []; document.getElementById('upload-zone').style.display = 'block'; form.style.display = 'none'; document.getElementById('upload-progress').style.display = 'none'; document.getElementById('progress-fill').style.width = '0%'; document.getElementById('media-title').value = ''; document.getElementById('media-description').value = ''; document.getElementById('media-tags').value = ''; document.getElementById('upload-btn').disabled = true; }; document.getElementById('upload-btn').addEventListener('click', async () => { const btn = document.getElementById('upload-btn'); const progress = document.getElementById('upload-progress'); btn.disabled = true; form.style.display = 'none'; progress.style.display = 'block'; let uploaded = 0; for (const file of files) { const formData = new FormData(); formData.append('file', file); formData.append('title', document.getElementById('media-title').value || file.name); formData.append('description', document.getElementById('media-description').value); formData.append('tags', document.getElementById('media-tags').value); try { await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData }); uploaded++; document.getElementById('progress-fill').style.width = `${(uploaded / files.length) * 100}%`; document.getElementById('progress-status').textContent = `Uploaded ${uploaded} of ${files.length}`; } catch (e) { showToast(`Failed to upload ${file.name}`, 'error'); } } showToast(`Successfully uploaded ${uploaded} files`, 'success'); loadStats(); setTimeout(window.resetUpload, 1500); }); }


function initSearch() {
    document.querySelectorAll('.search-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.search-tab').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.search-panel').forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const searchType = btn.dataset.search;
            const panel = document.getElementById(`${searchType}-panel`);
            if (panel) panel.classList.add('active');
        });
    });

    initQBE();
    const metaBtn = document.getElementById('meta-search-btn');
    if (metaBtn) metaBtn.addEventListener('click', searchMetadata);

    initHybrid();
}


function initQBE() {
    const zone = document.getElementById('qbe-upload-zone');
    const input = document.getElementById('qbe-file-input');
    const preview = document.getElementById('qbe-preview');
    const searchBtn = document.getElementById('qbe-search-btn');

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        handleQBEFile(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', () => handleQBEFile(input.files[0]));

    function handleQBEFile(file) {
        if (!file) return;
        state.qbeFile = file;
        if (searchBtn) searchBtn.disabled = false;

        // Update preview content
        let content = `<div class="upload-icon">🔎</div><p>${file.name}</p>`;
        if (file.type.startsWith('image/')) {
            content = `<img src="${URL.createObjectURL(file)}" alt="Query" style="max-height: 200px; margin-bottom: 1rem;">`;
        } else if (file.type.startsWith('video/')) {
            content = `<video src="${URL.createObjectURL(file)}" controls style="max-height: 200px; margin-bottom: 1rem;"></video>`;
        }
        preview.innerHTML = content + `<br><button class="btn btn-secondary btn-sm" id="qbe-clear" style="margin-top:1rem">Change File</button>`;

        document.getElementById('qbe-clear').addEventListener('click', (e) => {
            e.stopPropagation();
            state.qbeFile = null;
            preview.innerHTML = `
                <div class="upload-icon">🔎</div>
                <p>Drop a file to find similar media</p>`;
            if (searchBtn) searchBtn.disabled = true;
            input.value = '';
        });
    }

    if (searchBtn) searchBtn.addEventListener('click', searchQBE);
}

async function searchQBE() {
    const startTime = performance.now();

    // Check if this is a "Find Similar" search (using existing media ID)
    if (state.qbeMediaId) {
        const k = document.getElementById('qbe-k').value || 10;
        const metric = document.getElementById('qbe-metric').value || 'euclidean';

        showSearchLoading();
        try {
            const res = await fetch(`${API_BASE}/search/qbe/${state.qbeMediaId}?k=${k}&metric=${metric}`);
            const data = await res.json();
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
            displayResults(data, elapsed);
            showToast(`Found ${data.results?.length || 0} similar items`, 'success');
        } catch (e) {
            showToast('Search failed', 'error');
            hideSearchResults();
        }
        return;
    }

    // Regular file upload QBE search
    if (!state.qbeFile) return;
    const formData = new FormData();
    formData.append('file', state.qbeFile);
    formData.append('k', document.getElementById('qbe-k').value);
    formData.append('distance_metric', document.getElementById('qbe-metric').value);

    showSearchLoading();
    try {
        const res = await fetch(`${API_BASE}/search/qbe`, { method: 'POST', body: formData });
        const data = await res.json();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        displayResults(data, elapsed);
    } catch (e) {
        showToast('Search failed', 'error');
        hideSearchResults();
    }
}

async function searchMetadata() {
    const startTime = performance.now();
    const params = {
        title: document.getElementById('meta-title').value,
        media_type: document.getElementById('meta-type').value,
        tags: document.getElementById('meta-tags').value ? document.getElementById('meta-tags').value.split(',').map(t => t.trim()) : null,
        limit: parseInt(document.getElementById('meta-limit').value)
    };

    // Remove empty/null parameters
    Object.keys(params).forEach(k => {
        if (!params[k] || (Array.isArray(params[k]) && params[k].length === 0)) {
            delete params[k];
        }
    });

    console.log('[MMDB] Metadata search params:', params);

    showSearchLoading();
    try {
        const res = await fetch(`${API_BASE}/search/metadata`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        const data = await res.json();
        console.log('[MMDB] Metadata search response:', data);
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

        if (!res.ok) {
            showToast(data.error || 'Search failed', 'error');
            hideSearchResults();
            return;
        }

        displayResults(data, elapsed);
        showToast(`Found ${data.results_count || data.results?.length || 0} results`, 'success');
    } catch (e) {
        console.error('[MMDB] Metadata search error:', e);
        showToast('Search failed', 'error');
        hideSearchResults();
    }
}


function initHybrid() {
    const zone = document.getElementById('hybrid-upload-zone');
    const input = document.getElementById('hybrid-file-input');
    const preview = document.getElementById('hybrid-preview');
    const searchBtn = document.getElementById('hybrid-search-btn');

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        handleHybridFile(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', () => handleHybridFile(input.files[0]));

    function handleHybridFile(file) {
        if (!file) return;
        state.hybridFile = file;
        if (searchBtn) searchBtn.disabled = false;

        // Update preview content based on file type
        let content = `<div class="upload-icon small">📎</div><p>${file.name}</p>`;
        if (file.type.startsWith('image/')) {
            content = `<img src="${URL.createObjectURL(file)}" alt="Query" style="max-height: 150px; margin-bottom: 1rem;">`;
        } else if (file.type.startsWith('video/')) {
            content = `<video src="${URL.createObjectURL(file)}" controls style="max-height: 150px; margin-bottom: 1rem;"></video>`;
        }
        preview.innerHTML = content + `<br><button class="btn btn-secondary btn-sm" id="hybrid-clear" style="margin-top:1rem">Change File</button>`;

        document.getElementById('hybrid-clear').addEventListener('click', (e) => {
            e.stopPropagation();
            state.hybridFile = null;
            preview.innerHTML = `
                <div class="upload-icon small">📎</div>
                <p>Add query file</p>`;
            if (searchBtn) searchBtn.disabled = true;
            input.value = '';
        });
    }

    const featureSlider = document.getElementById('hybrid-feature-weight');
    const metaSlider = document.getElementById('hybrid-metadata-weight');

    if (featureSlider && metaSlider) {
        featureSlider.addEventListener('input', () => {
            document.getElementById('feature-weight-val').textContent = featureSlider.value;
            metaSlider.value = (1 - parseFloat(featureSlider.value)).toFixed(1);
            document.getElementById('metadata-weight-val').textContent = metaSlider.value;
        });

        metaSlider.addEventListener('input', () => {
            document.getElementById('metadata-weight-val').textContent = metaSlider.value;
            featureSlider.value = (1 - parseFloat(metaSlider.value)).toFixed(1);
            document.getElementById('feature-weight-val').textContent = featureSlider.value;
        });
    }

    if (searchBtn) {
        searchBtn.addEventListener('click', async () => {
            if (!state.hybridFile) {
                showToast('Please upload a file first', 'error');
                return;
            }

            const startTime = performance.now();
            const formData = new FormData();
            formData.append('file', state.hybridFile);
            formData.append('k', document.getElementById('hybrid-k').value);
            formData.append('metric', document.getElementById('hybrid-metric').value);
            formData.append('weight_feature', parseFloat(featureSlider.value));
            formData.append('weight_metadata', parseFloat(metaSlider.value));
            formData.append('title', document.getElementById('hybrid-title').value);
            formData.append('tags', document.getElementById('hybrid-tags').value);

            console.log('[MMDB] Hybrid search params:', {
                k: formData.get('k'),
                metric: formData.get('metric'),
                weight_feature: formData.get('weight_feature'),
                weight_metadata: formData.get('weight_metadata'),
                title: formData.get('title'),
                tags: formData.get('tags')
            });

            showSearchLoading();
            try {
                const res = await fetch(`${API_BASE}/search/hybrid`, { method: 'POST', body: formData });
                const data = await res.json();
                const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

                console.log('[MMDB] Hybrid search response:', data);

                if (!res.ok) {
                    showToast(data.error || 'Search failed', 'error');
                    hideSearchResults();
                    return;
                }

                displayResults(data, elapsed);
                showToast(`Found ${data.results_count || data.results?.length || 0} results`, 'success');
            } catch (e) {
                console.error('[MMDB] Hybrid search error:', e);
                showToast('Search failed', 'error');
                hideSearchResults();
            }
        });
    }
}

function showSearchLoading() { const results = document.getElementById('search-results'); results.style.display = 'block'; document.getElementById('results-grid').innerHTML = '<div class="loading-spinner"></div>'; document.getElementById('results-count').textContent = 'Searching...'; document.getElementById('query-time').textContent = ''; }
function hideSearchResults() { document.getElementById('search-results').style.display = 'none'; }
function displayResults(data, elapsedTime = null) {
    const results = document.getElementById('search-results');
    const grid = document.getElementById('results-grid');
    const count = document.getElementById('results-count');
    const queryTimeEl = document.getElementById('query-time');
    results.style.display = 'block';

    // Store query media ID and metric for comparison (when using existing media)
    state.lastSearchQueryMediaId = data.query_media_id || state.qbeMediaId || null;
    state.lastSearchMediaType = data.media_type || null;
    state.lastSearchMetric = data.metric || 'cosine'; // Store the metric used

    // Handle both results_count (metadata) and total_results (QBE) formats
    const totalResults = data.total_results || data.results_count || data.results?.length || 0;
    count.textContent = `${totalResults} results`;

    // Display query time
    if (elapsedTime) {
        queryTimeEl.textContent = `⏱️ ${elapsedTime}s`;
    } else {
        queryTimeEl.textContent = '';
    }

    if (!data.results || data.results.length === 0) {
        grid.innerHTML = '<div class="empty-state"><h3>No results found</h3></div>';
        return;
    }

    // Check if this is a QBE search (has similarity scores) and can show compare buttons
    const isQBESearch = data.query_type === 'qbe' || data.results[0]?.similarity !== undefined;
    const canCompare = isQBESearch && state.lastSearchQueryMediaId;

    const resultsHtml = data.results.map(r => {
        // Don't show compare button for the query itself
        const showCompareBtn = canCompare && r.id !== state.lastSearchQueryMediaId;

        return `
        <div class="result-card" data-id="${r.id}" data-media-type="${r.media_type || state.lastSearchMediaType}">
            ${r.thumbnail_url ? `<img class="thumbnail" src="${r.thumbnail_url}" loading="lazy">` : `<div class="thumbnail-placeholder"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg></div>`}
            ${r.similarity !== undefined ? `<span class="similarity-badge">${(r.similarity * 100).toFixed(1)}%</span>` : (r.distance !== undefined ? `<span class="similarity-badge" style="background:#ff9800;">D: ${r.distance.toFixed(2)}</span>` : '')}
            ${showCompareBtn ? `<button class="compare-btn" onclick="event.stopPropagation(); openComparisonModal(${state.lastSearchQueryMediaId}, ${r.id}, '${state.lastSearchMetric || 'cosine'}')" title="Compare Features">🔬</button>` : ''}
            <div class="card-info">
                <div class="card-title">${r.title || r.original_filename}</div>
            </div>
        </div>`;
    }).join('');

    grid.innerHTML = resultsHtml;

    grid.querySelectorAll('.result-card').forEach(card => {
        card.addEventListener('click', () => openMediaModal(card.dataset.id));
    });
}

function initBrowse() { document.getElementById('browse-type').addEventListener('change', () => { state.currentPage = 1; loadBrowseMedia(); }); document.getElementById('browse-sort').addEventListener('change', loadBrowseMedia); document.getElementById('browse-order').addEventListener('change', loadBrowseMedia); document.getElementById('prev-page').addEventListener('click', () => { if (state.currentPage > 1) { state.currentPage--; loadBrowseMedia(); } }); document.getElementById('next-page').addEventListener('click', () => { if (state.currentPage < state.totalPages) { state.currentPage++; loadBrowseMedia(); } }); }

async function loadBrowseMedia() {
    const grid = document.getElementById('browse-grid');
    if (!grid) return;

    grid.innerHTML = '<div class="loading-spinner"></div>';
    const type = document.getElementById('browse-type')?.value || '';
    const sort = document.getElementById('browse-sort')?.value || 'created_at';
    const order = document.getElementById('browse-order')?.value || 'desc';

    try {
        let url = `${API_BASE}/media?page=${state.currentPage}&per_page=${state.perPage}&sort=${sort}&order=${order}`;
        if (type) url += `&type=${type}`;

        const res = await fetch(url);
        const data = await res.json();

        // Backend returns: { items, total, page, per_page, total_pages, has_next, has_prev }
        state.totalPages = data.total_pages || 1;

        const pageInfo = document.getElementById('page-info');
        if (pageInfo) pageInfo.textContent = `Page ${state.currentPage} of ${state.totalPages}`;

        const prevBtn = document.getElementById('prev-page');
        const nextBtn = document.getElementById('next-page');
        if (prevBtn) prevBtn.disabled = state.currentPage <= 1;
        if (nextBtn) nextBtn.disabled = state.currentPage >= state.totalPages;

        if (data.items && data.items.length > 0) {
            grid.innerHTML = data.items.map(m => createMediaCard(m)).join('');
            attachMediaCardEvents(grid);
        } else {
            grid.innerHTML = '<div class="empty-state"><h3>No media found</h3></div>';
        }
    } catch (e) {
        grid.innerHTML = '<div class="empty-state"><h3>Failed to load media</h3></div>';
    }
}

function initModal() {
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.querySelector('.modal-overlay').addEventListener('click', closeModal);
    document.getElementById('modal-find-similar').addEventListener('click', findSimilar);
    document.getElementById('modal-download').addEventListener('click', downloadMedia);
    document.getElementById('modal-delete').addEventListener('click', deleteMedia);

    // Edit mode handlers
    document.getElementById('modal-edit').addEventListener('click', enableEditMode);
    document.getElementById('modal-cancel-edit').addEventListener('click', disableEditMode);
    document.getElementById('modal-save').addEventListener('click', saveMediaChanges);
}

function enableEditMode() {
    if (!state.selectedMedia) return;
    const m = state.selectedMedia;

    // Populate edit fields
    document.getElementById('edit-title').value = m.title || '';
    document.getElementById('edit-description').value = m.description || '';
    document.getElementById('edit-tags').value = (m.tags || []).join(', ');

    // Toggle modes
    document.getElementById('modal-view-mode').style.display = 'none';
    document.getElementById('modal-edit-mode').style.display = 'block';
}

function disableEditMode() {
    document.getElementById('modal-view-mode').style.display = 'block';
    document.getElementById('modal-edit-mode').style.display = 'none';
}

async function saveMediaChanges() {
    if (!state.selectedMedia) return;

    const title = document.getElementById('edit-title').value;
    const description = document.getElementById('edit-description').value;
    const tagsStr = document.getElementById('edit-tags').value;
    const tags = tagsStr.split(',').map(t => t.trim()).filter(t => t);

    try {
        const res = await fetch(`${API_BASE}/media/${state.selectedMedia.id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, description, tags })
        });

        const data = await res.json();

        if (res.ok) {
            showToast('Media updated successfully', 'success');
            // Update the modal with new data
            state.selectedMedia = data.media;
            document.getElementById('modal-title').textContent = title || state.selectedMedia.original_filename;
            if (tags.length > 0) {
                document.getElementById('modal-tags').innerHTML = tags.map(t => `<span class="tag">${t}</span>`).join('');
                document.getElementById('modal-tags-row').style.display = 'flex';
            } else {
                document.getElementById('modal-tags-row').style.display = 'none';
            }
            disableEditMode();
            loadRecentMedia(); // Refresh gallery if title changed
        } else {
            showToast(data.error || 'Update failed', 'error');
        }
    } catch (e) {
        showToast('Failed to save changes', 'error');
    }
}


async function openMediaModal(mediaId) {
    const modal = document.getElementById('media-modal');
    const preview = document.getElementById('modal-preview');

    // Reset to view mode
    disableEditMode();

    try {
        const res = await fetch(`${API_BASE}/media/${mediaId}`);
        const m = await res.json(); // Backend returns the media object directly

        if (m.error) {
            showToast(m.error, 'error');
            return;
        }

        state.selectedMedia = m;

        // Check if file exists on server for dynamic download button
        try {
            const fileCheck = await fetch(`${API_BASE}/media/${mediaId}/file_available`);
            const fileAvailable = await fileCheck.json();

            const downloadBtn = document.getElementById('modal-download');
            if (downloadBtn) {
                if (fileAvailable.file_exists) {
                    downloadBtn.style.display = '';  // Show button
                    console.log('[MMDB] Download available for:', m.title);
                } else {
                    downloadBtn.style.display = 'none';  // Hide button
                    console.log('[MMDB] File not available for download:', m.title);
                }
            }
        } catch (e) {
            console.error('[MMDB] Error checking file availability:', e);
            // On error, show download button (default behavior)
        }

        // Set preview based on media type
        console.log('[MMDB] Setting preview for media type:', m.media_type, 'id:', m.id);
        if (m.media_type === 'image') {
            preview.innerHTML = `<img src="${API_BASE}/media/${m.id}/file">`;
        } else if (m.media_type === 'video') {
            // Use /file instead of /stream - browser-native handling
            preview.innerHTML = `<video src="${API_BASE}/media/${m.id}/file" controls style="width:100%;max-height:400px;"></video>`;
            console.log('[MMDB] Video URL:', `${API_BASE}/media/${m.id}/file`);
        } else if (m.media_type === 'audio') {
            preview.innerHTML = `<audio src="${API_BASE}/media/${m.id}/file" controls style="width:100%;"></audio>`;
            console.log('[MMDB] Audio URL:', `${API_BASE}/media/${m.id}/file`);
        }

        // Populate details
        document.getElementById('modal-title').textContent = m.title || m.original_filename;
        document.getElementById('modal-type').textContent = m.media_type;
        document.getElementById('modal-size').textContent = formatFileSize(m.file_size);

        const dimRow = document.getElementById('modal-dimensions-row');
        if (m.width && m.height) {
            dimRow.style.display = 'flex';
            document.getElementById('modal-dimensions').textContent = `${m.width} × ${m.height}`;
        } else {
            dimRow.style.display = 'none';
        }

        const durRow = document.getElementById('modal-duration-row');
        if (m.duration) {
            durRow.style.display = 'flex';
            document.getElementById('modal-duration').textContent = formatDuration(m.duration);
        } else {
            durRow.style.display = 'none';
        }

        document.getElementById('modal-date').textContent = new Date(m.created_at).toLocaleDateString();

        const tagsRow = document.getElementById('modal-tags-row');
        if (m.tags && m.tags.length) {
            tagsRow.style.display = 'flex';
            document.getElementById('modal-tags').innerHTML = m.tags.map(t => `<span class="tag">${t}</span>`).join('');
        } else {
            tagsRow.style.display = 'none';
        }

        modal.classList.add('active');
    } catch (e) {
        showToast('Failed to load media', 'error');
    }
}

function closeModal() {
    document.getElementById('media-modal').classList.remove('active');
    document.getElementById('modal-preview').innerHTML = '';
}

// Find Similar - navigate to search page and use existing media
function findSimilar() {
    if (!state.selectedMedia) return;

    const mediaId = state.selectedMedia.id;
    const mediaTitle = state.selectedMedia.title || state.selectedMedia.original_filename;
    const mediaType = state.selectedMedia.media_type;

    // Close modal
    closeModal();

    // Switch to search page
    switchView('search');

    // Activate QBE tab
    document.querySelectorAll('.search-tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.search-panel').forEach(panel => panel.classList.remove('active'));
    const qbeTab = document.querySelector('[data-search="qbe"]');
    if (qbeTab) qbeTab.classList.add('active');
    const qbePanel = document.getElementById('qbe-panel');
    if (qbePanel) qbePanel.classList.add('active');

    // Show benchmark image in the upload zone
    const uploadZone = document.getElementById('qbe-upload-zone');
    if (uploadZone) {
        uploadZone.innerHTML = `
            <div class="qbe-preview" style="display: block;">
                <img src="/api/media/${mediaId}/thumbnail" 
                     style="width: 100%; max-width: 280px; height: auto; aspect-ratio: 1; object-fit: cover; border-radius: 12px; border: 4px solid #4caf50; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 0.75rem;"
                     onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22%3E%3Crect fill=%22%23e0e0e0%22 width=%22100%22 height=%22100%22/%3E%3Ctext x=%2250%22 y=%2255%22 text-anchor=%22middle%22 fill=%22%23666%22 font-size=%2212%22%3ENo Preview%3C/text%3E%3C/svg%3E'">
                <p style="margin: 0.5rem 0 0.25rem 0; font-weight: bold; color: #2e7d32; font-size: 1rem;">${mediaTitle}</p>
                <p style="margin: 0 0 0.75rem 0; color: #666; font-size: 0.875rem;">Type: ${mediaType.toUpperCase()}</p>
                <button class="btn btn-secondary btn-sm" onclick="resetQBEPanel()" style="width: 100%; max-width: 200px;">🔄 Change File</button>
            </div>`;
    }

    // Store media ID for QBE search
    state.qbeMediaId = mediaId;

    // Enable search button
    const searchBtn = document.getElementById('qbe-search-btn');
    if (searchBtn) {
        searchBtn.disabled = false;
        searchBtn.textContent = '🔍 Find Similar';
    }

    showToast('Adjust options and click "Find Similar"', 'success');
}

// Reset QBE panel to upload mode
function resetQBEPanel() {
    state.qbeMediaId = null;
    state.qbeFile = null;

    const uploadZone = document.getElementById('qbe-upload-zone');
    if (uploadZone) {
        uploadZone.innerHTML = `
            <input type="file" id="qbe-file-input" accept="image/*,audio/*,video/*" hidden>
            <div class="qbe-preview" id="qbe-preview">
                <div class="upload-icon">🔎</div>
                <p>Drop a file to find similar media</p>
            </div>`;

        // Re-attach event listeners for file upload
        const input = document.getElementById('qbe-file-input');
        const zone = uploadZone;
        zone.addEventListener('click', () => input.click());
        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            if (e.dataTransfer.files[0]) {
                // Handle file - this would need the handleQBEFile function from initQBE
                state.qbeFile = e.dataTransfer.files[0];
            }
        });
        input.addEventListener('change', () => {
            if (input.files[0]) {
                state.qbeFile = input.files[0];
            }
        });
    }

    const searchBtn = document.getElementById('qbe-search-btn');
    if (searchBtn) {
        searchBtn.disabled = true;
        searchBtn.textContent = 'Search Similar';
    }
}

function closeSimilarModal() {
    document.getElementById('similar-modal').classList.remove('active');
}

async function executeFindSimilar() {
    closeSimilarModal();
}
function downloadMedia() { if (state.selectedMedia) window.open(`${API_BASE}/media/${state.selectedMedia.id}/download`, '_blank'); }
async function deleteMedia() { if (!state.selectedMedia) return; if (!confirm('Are you sure you want to delete this media?')) return; try { await fetch(`${API_BASE}/media/${state.selectedMedia.id}`, { method: 'DELETE' }); showToast('Media deleted', 'success'); closeModal(); loadStats(); loadBrowseMedia(); } catch (e) { showToast('Failed to delete', 'error'); } }

function formatFileSize(bytes) { if (!bytes) return '0 B'; const k = 1024; const sizes = ['B', 'KB', 'MB', 'GB']; const i = Math.floor(Math.log(bytes) / Math.log(k)); return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]; }
function formatDuration(seconds) { const m = Math.floor(seconds / 60); const s = Math.floor(seconds % 60); return `${m}:${s.toString().padStart(2, '0')}`; }
function showToast(message, type = 'info') { const container = document.getElementById('toast-container'); const toast = document.createElement('div'); toast.className = `toast ${type}`; toast.innerHTML = `<span class="toast-message">${message}</span><button class="toast-close">×</button>`; container.appendChild(toast); toast.querySelector('.toast-close').addEventListener('click', () => toast.remove()); setTimeout(() => toast.remove(), 4000); }

// ==========================================
// Feature Comparison Visualization
// ==========================================

// Store chart instances for cleanup
let comparisonCharts = {};

function destroyAllCharts() {
    Object.keys(comparisonCharts).forEach(key => {
        if (comparisonCharts[key]) {
            comparisonCharts[key].destroy();
            comparisonCharts[key] = null;
        }
    });
    comparisonCharts = {};
}

function closeComparisonModal() {
    const modal = document.getElementById('comparison-modal');
    if (modal) {
        modal.classList.remove('active');
    }
    destroyAllCharts();
    // Hide all chart containers
    document.querySelectorAll('.media-charts').forEach(el => el.style.display = 'none');
}

async function openComparisonModal(queryId, resultId, metric = 'cosine') {
    console.log('[MMDB] Opening comparison modal:', queryId, 'vs', resultId, 'metric:', metric);

    const modal = document.getElementById('comparison-modal');
    const loading = document.getElementById('comparison-loading');
    const breakdown = document.getElementById('similarity-breakdown');
    const thumbnails = document.querySelector('.comparison-thumbnails');

    // Hide all chart containers initially
    document.querySelectorAll('.media-charts').forEach(el => el.style.display = 'none');

    // Show modal with loading state
    modal.classList.add('active');
    loading.style.display = 'flex';
    breakdown.style.display = 'none';
    thumbnails.style.display = 'none';

    try {
        // Fetch comparison data from API with metric parameter
        const response = await fetch(`${API_BASE}/compare/${queryId}/${resultId}?metric=${metric}`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to load comparison');
        }

        console.log('[MMDB] Comparison data:', data);

        // Hide loading, show content
        loading.style.display = 'none';
        thumbnails.style.display = 'flex';
        breakdown.style.display = 'block';

        // Update thumbnails
        document.getElementById('comparison-query-thumb').src = data.query.thumbnail_url;
        document.getElementById('comparison-result-thumb').src = data.result.thumbnail_url;
        document.getElementById('comparison-query-title').textContent = data.query.title;
        document.getElementById('comparison-result-title').textContent = data.result.title;

        // Render similarity breakdown with media type
        renderSimilarityBreakdown(data.similarities, data.media_type);

        // Show and render the appropriate chart container based on media type
        if (data.media_type === 'image') {
            document.getElementById('image-charts').style.display = 'block';
            renderImageCharts(data.features);
        } else if (data.media_type === 'audio') {
            document.getElementById('audio-charts').style.display = 'block';
            renderAudioCharts(data.features);
        } else if (data.media_type === 'video') {
            document.getElementById('video-charts').style.display = 'block';
            renderVideoCharts(data.features);
        }

    } catch (error) {
        console.error('[MMDB] Comparison error:', error);
        loading.innerHTML = `<p style="color: #ff7675;">❌ ${error.message}</p>`;
        showToast('Failed to load feature comparison', 'error');
    }
}

function renderSimilarityBreakdown(similarities, mediaType) {
    const container = document.querySelector('.breakdown-bars');

    // Define items based on media type
    let items;
    if (mediaType === 'audio') {
        items = [
            { key: 'mfcc', label: '🎵 MFCC Match', class: 'color' },
            { key: 'spectral', label: '🌊 Spectral Match', class: 'texture' },
            { key: 'waveform', label: '📊 Waveform Match', class: 'deep' },
            { key: 'overall', label: '⭐ Overall', class: 'overall' }
        ];
    } else if (mediaType === 'video') {
        items = [
            { key: 'keyframe', label: '🎬 Keyframe Match', class: 'color' },
            { key: 'motion', label: '🏃 Motion Match', class: 'texture' },
            { key: 'scene_stats', label: '🎞️ Scene Match', class: 'deep' },
            { key: 'overall', label: '⭐ Overall', class: 'overall' }
        ];
    } else {
        // Default: image
        items = [
            { key: 'color_histogram', label: '🎨 Color Match', class: 'color' },
            { key: 'texture_lbp', label: '🔲 Texture Match', class: 'texture' },
            { key: 'deep_features', label: '🧠 Deep Features', class: 'deep' },
            { key: 'overall', label: '⭐ Overall', class: 'overall' }
        ];
    }

    container.innerHTML = items.map(item => {
        const value = similarities[item.key] || 0;
        return `
            <div class="breakdown-item">
                <span class="breakdown-label">${item.label}</span>
                <div class="breakdown-bar-container">
                    <div class="breakdown-bar ${item.class}" style="width: 0%">
                        <span class="breakdown-value">${value.toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Animate bars after render
    setTimeout(() => {
        container.querySelectorAll('.breakdown-bar').forEach((bar, index) => {
            const value = similarities[items[index].key] || 0;
            bar.style.width = `${Math.max(value, 8)}%`; // Minimum width to show value
        });
    }, 100);
}

// Chart.js default options (shared across all chart types)
const chartCommonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
            labels: {
                usePointStyle: true,
                padding: 15,
                font: { family: 'Fredoka', size: 11 }
            }
        }
    },
    scales: {
        y: {
            beginAtZero: true,
            grid: { color: 'rgba(0,0,0,0.05)' }
        },
        x: {
            grid: { display: false }
        }
    }
};

function renderImageCharts(features) {
    destroyAllCharts();

    // 1. Color Histogram Chart
    const colorCtx = document.getElementById('color-chart').getContext('2d');
    const colorData = features.color_histogram;

    if (colorData && colorData.query && colorData.result) {
        const allLabels = [
            ...colorData.query.hue.map((_, i) => `H${i + 1}`),
            ...colorData.query.saturation.map((_, i) => `S${i + 1}`),
            ...colorData.query.value.map((_, i) => `V${i + 1}`)
        ];
        const queryValues = [...colorData.query.hue, ...colorData.query.saturation, ...colorData.query.value];
        const resultValues = [...colorData.result.hue, ...colorData.result.saturation, ...colorData.result.value];

        comparisonCharts.color = new Chart(colorCtx, {
            type: 'bar',
            data: {
                labels: allLabels,
                datasets: [
                    { label: 'Query', data: queryValues, backgroundColor: 'rgba(116, 185, 255, 0.7)', borderColor: '#0984e3', borderWidth: 1 },
                    { label: 'Result', data: resultValues, backgroundColor: 'rgba(162, 155, 254, 0.7)', borderColor: '#6c5ce7', borderWidth: 1 }
                ]
            },
            options: chartCommonOptions
        });
    }

    // 2. Texture LBP Chart
    const textureCtx = document.getElementById('texture-chart').getContext('2d');
    const textureData = features.texture_lbp;

    if (textureData && textureData.query && textureData.result) {
        comparisonCharts.texture = new Chart(textureCtx, {
            type: 'line',
            data: {
                labels: textureData.labels || textureData.query.map((_, i) => i + 1),
                datasets: [
                    { label: 'Query', data: textureData.query, borderColor: '#00b894', backgroundColor: 'rgba(85, 239, 196, 0.2)', fill: true, tension: 0.3, pointRadius: 2 },
                    { label: 'Result', data: textureData.result, borderColor: '#e17055', backgroundColor: 'rgba(250, 177, 160, 0.2)', fill: true, tension: 0.3, pointRadius: 2 }
                ]
            },
            options: chartCommonOptions
        });
    }

    // 3. Deep Features Chart
    const deepCtx = document.getElementById('deep-chart').getContext('2d');
    const deepData = features.deep_features;

    if (deepData && deepData.query && deepData.result) {
        comparisonCharts.deep = new Chart(deepCtx, {
            type: 'bar',
            data: {
                labels: deepData.labels || deepData.query.map((_, i) => `G${i + 1}`),
                datasets: [
                    { label: 'Query', data: deepData.query, backgroundColor: 'rgba(253, 203, 110, 0.7)', borderColor: '#f39c12', borderWidth: 1 },
                    { label: 'Result', data: deepData.result, backgroundColor: 'rgba(108, 92, 231, 0.7)', borderColor: '#6c5ce7', borderWidth: 1 }
                ]
            },
            options: chartCommonOptions
        });
    }
}

function renderAudioCharts(features) {
    destroyAllCharts();

    // 1. MFCC Features Chart
    const mfccCtx = document.getElementById('mfcc-chart').getContext('2d');
    const mfccData = features.mfcc;

    if (mfccData && mfccData.query && mfccData.result) {
        comparisonCharts.mfcc = new Chart(mfccCtx, {
            type: 'line',
            data: {
                labels: mfccData.labels || mfccData.query.map((_, i) => i + 1),
                datasets: [
                    { label: 'Query', data: mfccData.query, borderColor: '#0984e3', backgroundColor: 'rgba(9, 132, 227, 0.1)', fill: true, tension: 0.3, pointRadius: 2 },
                    { label: 'Result', data: mfccData.result, borderColor: '#6c5ce7', backgroundColor: 'rgba(108, 92, 231, 0.1)', fill: true, tension: 0.3, pointRadius: 2 }
                ]
            },
            options: chartCommonOptions
        });
    }

    // 2. Spectral Features Chart
    const spectralCtx = document.getElementById('spectral-chart').getContext('2d');
    const spectralData = features.spectral;

    if (spectralData && spectralData.query && spectralData.result) {
        comparisonCharts.spectral = new Chart(spectralCtx, {
            type: 'bar',
            data: {
                labels: spectralData.labels || ['Centroid', 'Rolloff', 'Bandwidth', 'Contrast', 'Flatness', 'ZCR'],
                datasets: [
                    { label: 'Query', data: spectralData.query, backgroundColor: 'rgba(0, 184, 148, 0.7)', borderColor: '#00b894', borderWidth: 1 },
                    { label: 'Result', data: spectralData.result, backgroundColor: 'rgba(225, 112, 85, 0.7)', borderColor: '#e17055', borderWidth: 1 }
                ]
            },
            options: chartCommonOptions
        });
    }

    // 3. Waveform Stats Chart
    const waveformCtx = document.getElementById('waveform-chart').getContext('2d');
    const waveformData = features.waveform;

    if (waveformData && waveformData.query && waveformData.result) {
        comparisonCharts.waveform = new Chart(waveformCtx, {
            type: 'bar',
            data: {
                labels: waveformData.labels || ['RMS', 'Peak', 'Crest', 'DynRange', 'Silence'],
                datasets: [
                    { label: 'Query', data: waveformData.query, backgroundColor: 'rgba(253, 203, 110, 0.7)', borderColor: '#fdcb6e', borderWidth: 1 },
                    { label: 'Result', data: waveformData.result, backgroundColor: 'rgba(225, 112, 85, 0.7)', borderColor: '#e17055', borderWidth: 1 }
                ]
            },
            options: chartCommonOptions
        });
    }
}

function renderVideoCharts(features) {
    destroyAllCharts();

    // 1. Keyframe Features Chart
    const keyframeCtx = document.getElementById('keyframe-chart').getContext('2d');
    const keyframeData = features.keyframe;

    if (keyframeData && keyframeData.query && keyframeData.result) {
        comparisonCharts.keyframe = new Chart(keyframeCtx, {
            type: 'bar',
            data: {
                labels: keyframeData.labels || keyframeData.query.map((_, i) => `K${i + 1}`),
                datasets: [
                    { label: 'Query', data: keyframeData.query, backgroundColor: 'rgba(116, 185, 255, 0.7)', borderColor: '#0984e3', borderWidth: 1 },
                    { label: 'Result', data: keyframeData.result, backgroundColor: 'rgba(162, 155, 254, 0.7)', borderColor: '#6c5ce7', borderWidth: 1 }
                ]
            },
            options: chartCommonOptions
        });
    }

    // 2. Motion Features Chart
    const motionCtx = document.getElementById('motion-chart').getContext('2d');
    const motionData = features.motion;

    if (motionData && motionData.query && motionData.result) {
        // Downsample motion data if too large (64 -> 16)
        let queryMotion = motionData.query;
        let resultMotion = motionData.result;
        let labels = motionData.labels;

        if (queryMotion.length > 16) {
            const step = Math.floor(queryMotion.length / 16);
            queryMotion = queryMotion.filter((_, i) => i % step === 0).slice(0, 16);
            resultMotion = resultMotion.filter((_, i) => i % step === 0).slice(0, 16);
            labels = queryMotion.map((_, i) => `M${i + 1}`);
        }

        comparisonCharts.motion = new Chart(motionCtx, {
            type: 'line',
            data: {
                labels: labels || queryMotion.map((_, i) => `M${i + 1}`),
                datasets: [
                    { label: 'Query', data: queryMotion, borderColor: '#00b894', backgroundColor: 'rgba(85, 239, 196, 0.2)', fill: true, tension: 0.3, pointRadius: 2 },
                    { label: 'Result', data: resultMotion, borderColor: '#e17055', backgroundColor: 'rgba(250, 177, 160, 0.2)', fill: true, tension: 0.3, pointRadius: 2 }
                ]
            },
            options: chartCommonOptions
        });
    }

    // 3. Scene Stats Chart
    const sceneCtx = document.getElementById('scene-chart').getContext('2d');
    const sceneData = features.scene_stats;

    if (sceneData && sceneData.query && sceneData.result) {
        comparisonCharts.scene = new Chart(sceneCtx, {
            type: 'bar',
            data: {
                labels: sceneData.labels || ['Dur', 'FPS', 'Scenes', 'Bright', 'Std', 'Contrast', 'Sat', 'DomCol', 'Motion', 'Cmplx'],
                datasets: [
                    { label: 'Query', data: sceneData.query, backgroundColor: 'rgba(253, 203, 110, 0.7)', borderColor: '#f39c12', borderWidth: 1 },
                    { label: 'Result', data: sceneData.result, backgroundColor: 'rgba(108, 92, 231, 0.7)', borderColor: '#6c5ce7', borderWidth: 1 }
                ]
            },
            options: chartCommonOptions
        });
    }
}

