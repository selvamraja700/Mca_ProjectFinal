document.addEventListener('DOMContentLoaded', () => {
    const API_BASE = 'http://127.0.0.1:5000/api';

    let currentImageFile = null;
    let currentImageBase64 = null;
    let activeVisualizations = {
        original: null,
        ela: null,
        heatmap: null
    };
    let currentViewMode = 'original';

    const dropzone = document.getElementById('image-dropzone');
    const fileInput = document.getElementById('file-input');
    const emptyState = document.getElementById('dropzone-empty-state');
    const previewState = document.getElementById('dropzone-preview-state');
    const previewImage = document.getElementById('preview-image');
    const analyzeBtn = document.getElementById('analyze-btn');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            processFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            processFile(e.target.files[0]);
        }
    }

    function processFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPEG, PNG, BMP, etc.)');
            return;
        }

        currentImageFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImageBase64 = e.target.result;
            displayImagePreview(currentImageBase64);
            analyzeBtn.disabled = false;
            resetResults();
        };
        reader.readAsDataURL(file);
    }

    function displayImagePreview(src) {
        previewImage.src = src;
        emptyState.classList.add('hidden');
        previewState.classList.remove('hidden');
        activeVisualizations.original = src;
        setViewMode('original');
    }

    const loadAuthenticBtn = document.getElementById('load-authentic-btn');
    const loadForgedBtn = document.getElementById('load-forged-btn');

    if (loadAuthenticBtn) loadAuthenticBtn.addEventListener('click', () => loadSample('authentic'));
    if (loadForgedBtn) loadForgedBtn.addEventListener('click', () => loadSample('forged'));

    async function loadSample(type) {
        try {
            const res = await fetch(`${API_BASE}/samples/${type}`);
            const data = await res.json();
            if (data.image_base64) {
                currentImageFile = null;
                currentImageBase64 = data.image_base64;
                displayImagePreview(currentImageBase64);
                analyzeBtn.disabled = false;
                resetResults();
            }
        } catch (err) {
            console.error('Failed to load sample image:', err);
            alert('Failed to load sample. Ensure backend is running.');
        }
    }

    const viewModeBtns = document.querySelectorAll('.view-mode-btn');
    viewModeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.getAttribute('data-mode');
            setViewMode(mode);
        });
    });

    function setViewMode(mode) {
        currentViewMode = mode;
        viewModeBtns.forEach(b => b.classList.remove('active'));
        document.querySelector(`.view-mode-btn[data-mode="${mode}"]`)?.classList.add('active');

        if (activeVisualizations[mode]) {
            previewImage.src = activeVisualizations[mode];
        }
    }

    analyzeBtn.addEventListener('click', runForgeryAnalysis);

    async function runForgeryAnalysis() {
        if (!currentImageBase64) return;

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = `ANALYZING IMAGE...`;
        analyzeBtn.classList.add('loading');

        try {
            let bodyData;
            let headers = {};

            if (currentImageFile) {
                const formData = new FormData();
                formData.append('image', currentImageFile);
                bodyData = formData;
            } else {
                bodyData = JSON.stringify({ image_base64: currentImageBase64 });
                headers['Content-Type'] = 'application/json';
            }

            const res = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: headers,
                body: bodyData
            });

            const data = await res.json();

            if (data.success) {
                renderResults(data);
            } else {
                alert(`Analysis Failed: ${data.error || 'Unable to analyze this image. Please try again.'}`);
            }
        } catch (err) {
            console.error('Analysis API call failed:', err);
            alert('ANALYSIS FAILED: Backend API connection failed. Ensure backend Flask server is running.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `ANALYZE IMAGE AUTHENTICITY`;
            analyzeBtn.classList.remove('loading');
        }
    }

    function renderResults(data) {
        const placeholder = document.getElementById('results-placeholder');
        const activeResults = document.getElementById('results-active-state');

        placeholder.classList.add('hidden');
        activeResults.classList.remove('hidden');

        activeVisualizations.original = data.visualizations.original || currentImageBase64;
        activeVisualizations.ela = data.visualizations.ela;
        activeVisualizations.heatmap = data.visualizations.heatmap;

        const banner = document.getElementById('verdict-banner');
        const badge = document.getElementById('verdict-badge');

        if (data.prediction === 'Forged') {
            banner.className = 'verdict-banner forged';
            badge.innerText = 'FORGED / ALTERED';
        } else {
            banner.className = 'verdict-banner genuine';
            badge.innerText = 'AUTHENTIC';
        }

        const confidenceVal = document.getElementById('confidence-value');
        if (confidenceVal) {
            const genuineProb = parseFloat(data.probabilities.genuine) || 0;
            const forgedProb = parseFloat(data.probabilities.forged) || 0;
            const maxProb = Math.max(genuineProb, forgedProb);
            confidenceVal.innerText = `${maxProb.toFixed(1)}%`;
        }

        document.getElementById('prob-genuine-text').innerText = `${data.probabilities.genuine}%`;
        document.getElementById('prob-genuine-fill').style.width = `${data.probabilities.genuine}%`;

        document.getElementById('prob-forged-text').innerText = `${data.probabilities.forged}%`;
        document.getElementById('prob-forged-fill').style.width = `${data.probabilities.forged}%`;

        const elaNoiseElement = document.getElementById('meta-ela-noise');
        if (elaNoiseElement) {
            elaNoiseElement.innerText = data.ela_noise_level;
        }

        const resElement = document.getElementById('meta-resolution');
        if (resElement) {
            resElement.innerText = data.dimensions;
        }
    }

    function resetResults() {
        document.getElementById('results-placeholder').classList.remove('hidden');
        document.getElementById('results-active-state').classList.add('hidden');
        activeVisualizations = { original: currentImageBase64, ela: null, heatmap: null };
    }
});
