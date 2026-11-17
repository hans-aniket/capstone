const API_URL = '/api/v1';

export async function checkHealth() {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) throw new Error('API unavailable');
    return await response.json();
  } catch (error) {
    return { status: 'offline', loaded_models: [] };
  }
}

export async function analyzeSentiment(text, model = null) {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, model }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Inference pipeline failed');
    }
    
    return await response.json();
  } catch (error) {
    throw new Error(error.message || 'Failed to connect to inference engine');
  }
}

export async function getExplanation(text, model = 'DistilBERT') {
  try {
    const response = await fetch(`${API_URL}/explain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, model }),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Explainability failed');
    }
    return await response.json();
  } catch (error) {
    throw new Error(error.message || 'Explainability service error');
  }
}

export async function getRobustnessBenchmark(refresh = false) {
  try {
    const response = await fetch(`${API_URL}/robustness/benchmark?refresh=${refresh}`);
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Benchmark failed');
    }
    return await response.json();
  } catch (error) {
    throw new Error(error.message || 'Robustness service error');
  }
}

export async function runStressTest(text) {
  try {
    const response = await fetch(`${API_URL}/robustness/stress-test`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Stress-test failed');
    }
    return await response.json();
  } catch (error) {
    throw new Error(error.message || 'Stress-test service error');
  }
}

export async function uploadBatchCSV(file, primaryModel = 'DistilBERT') {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('primary_model', primaryModel);

    const response = await fetch(`${API_URL}/batch/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Batch upload failed');
    }
    return await response.json();
  } catch (error) {
    throw new Error(error.message || 'Batch upload error');
  }
}

export async function getSampleBatch() {
  try {
    const response = await fetch(`${API_URL}/batch/sample`);
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Sample batch failed');
    }
    return await response.json();
  } catch (error) {
    throw new Error(error.message || 'Sample batch service error');
  }
}
