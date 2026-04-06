document.addEventListener("DOMContentLoaded", () => {
    // Check if CONFIG was loaded successfully from config.js
    if (typeof CONFIG === 'undefined') {
        alert("Configuration data missing! Please run 'extract_config.py' to generate 'config.js'.");
        return;
    }

    const stateSelect = document.getElementById('state');
    const districtSelect = document.getElementById('district');
    const cropSelect = document.getElementById('crop');
    const seasonSelect = document.getElementById('season');
    
    const rainInput = document.getElementById('rainfall');
    const tempInput = document.getElementById('temperature');
    const rainLabel = document.getElementById('rain-bounds');
    const tempLabel = document.getElementById('temp-bounds');

    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');
    const resultContainer = document.getElementById('result-container');

    // --- 1. SET BOUNDS FROM DATASET ---
    const bounds = CONFIG.bounds;
    
    rainInput.min = bounds.rainfall.min;
    rainInput.max = bounds.rainfall.max;
    rainInput.placeholder = `Min: ${bounds.rainfall.min}, Max: ${bounds.rainfall.max}`;
    rainLabel.textContent = `[${bounds.rainfall.min} - ${bounds.rainfall.max}]`;

    tempInput.min = bounds.temperature.min;
    tempInput.max = bounds.temperature.max;
    tempInput.placeholder = `Min: ${bounds.temperature.min}, Max: ${bounds.temperature.max}`;
    tempLabel.textContent = `[${bounds.temperature.min} - ${bounds.temperature.max}]`;

    // --- 2. POPULATE DROPDOWNS ---
    function populateDropdown(element, dataList) {
        dataList.forEach(item => {
            let option = document.createElement('option');
            option.value = item;
            option.textContent = item;
            element.appendChild(option);
        });
    }

    populateDropdown(stateSelect, Object.keys(CONFIG.locationData));
    populateDropdown(cropSelect, CONFIG.crops);
    populateDropdown(seasonSelect, CONFIG.seasons);

    // Cascading District Selection
    stateSelect.addEventListener('change', function() {
        const selectedState = this.value;
        districtSelect.innerHTML = '<option value="" disabled selected>Select District</option>'; 
        
        if (selectedState) {
            populateDropdown(districtSelect, CONFIG.locationData[selectedState]);
            districtSelect.disabled = false;
        } else {
            districtSelect.disabled = true;
        }
    });

    // --- 3. API SUBMISSION ---
    // ⚠️ UPDATE THIS WHEN YOU DEPLOY YOUR BACKEND (e.g., https://your-render-url.com/predict)
    const API_URL = "http://127.0.0.1:8000/predict"; 

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        resultContainer.style.display = 'none';
        resultContainer.className = 'result-box';
        
        const payload = {
            State: stateSelect.value,
            District: districtSelect.value,
            Crop: cropSelect.value,
            Season: seasonSelect.value,
            Area: parseFloat(document.getElementById('area').value),
            Rainfall: parseFloat(rainInput.value),
            Temperature: parseFloat(tempInput.value),
            Crop_Year: parseInt(document.getElementById('year').value)
        };

        submitBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        spinner.style.display = "block";

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Server failed to process request.");
            }

            resultContainer.classList.add('result-success');
            resultContainer.innerHTML = `
                <h3>🌾 Prediction Successful</h3>
                <p style="font-size: 2rem; font-weight: 700; color: var(--primary-color); margin: 1rem 0;">
                    ${data.predicted_yield_t_ha} <span style="font-size: 1.1rem; color: var(--text-muted)">tonnes / hectare</span>
                </p>
                <p>Estimated Total Production: <b style="color: var(--accent-color)">${data.estimated_production_tonnes} Tonnes</b></p>
            `;

        } catch (error) {
            console.error("API Error:", error);
            resultContainer.classList.add('result-error');
            
            let errorMsg = error.message;
            if(errorMsg.includes('Failed to fetch')){
                errorMsg = "Cannot connect to server. Ensure FastAPI is running on port 8000 and CORS is enabled.";
            }

            resultContainer.innerHTML = `
                <h3>⚠️ Prediction Failed</h3>
                <p>${errorMsg}</p>
            `;
        } finally {
            submitBtn.disabled = false;
            btnText.textContent = "Predict Yield";
            spinner.style.display = "none";
            resultContainer.style.display = 'block';
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });
});