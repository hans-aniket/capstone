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

export async function analyzeSentiment(text) {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
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
