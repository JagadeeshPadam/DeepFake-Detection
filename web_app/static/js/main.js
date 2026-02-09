document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('result-container');
    const predictionBadge = document.getElementById('prediction-badge');
    const predictionText = document.getElementById('prediction-text');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    const loader = document.getElementById('loader');
    const btnText = document.querySelector('.btn-text');

    let selectedFile = null;

    // Handle drag and drop
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('video/')) {
            alert('Please select a video file.');
            return;
        }
        selectedFile = file;
        fileInfo.textContent = `Selected: ${file.name}`;
        predictBtn.disabled = false;
        resultContainer.classList.add('hidden');
    }

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI State: Loading
        predictBtn.disabled = true;
        loader.style.display = 'block';
        btnText.textContent = 'Analyzing...';
        resultContainer.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Internal Server Error');
            }

            const data = await response.json();
            displayResult(data);
        } catch (error) {
            console.error('Error:', error);
            alert(`Analysis failed: ${error.message}`);
        } finally {
            predictBtn.disabled = false;
            loader.style.display = 'none';
            btnText.textContent = 'Detect Authenticity';
        }
    });

    function displayResult(data) {
        resultContainer.classList.remove('hidden');
        predictionText.textContent = data.prediction;

        // Update badge style
        predictionBadge.className = 'prediction-badge';
        predictionBadge.classList.add(data.prediction.toLowerCase());

        // Update confidence
        const score = parseFloat(data.confidence);
        confidenceText.textContent = data.confidence;

        // Trigger progress bar animation
        setTimeout(() => {
            confidenceFill.style.width = data.confidence;
            // Apply color to progress bar based on result
            if (data.prediction.toLowerCase() === 'real') {
                confidenceFill.style.background = '#22c55e';
            } else {
                confidenceFill.style.background = '#ef4444';
            }
        }, 100);

        // Smooth scroll to result
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
});
