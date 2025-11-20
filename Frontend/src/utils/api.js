import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Predict label and probability for input text.
 * @param {string} text - Input text to analyze
 * @returns {Promise<Object>} Prediction result with label, prob, tokens, token_importances
 */
export const predictText = async (text) => {
  try {
    const response = await api.post('/predict', { text });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Prediction failed');
    } else if (error.request) {
      throw new Error('Could not connect to backend. Make sure the server is running on port 8000.');
    } else {
      throw new Error(error.message || 'Prediction failed');
    }
  }
};

/**
 * Generate detailed explanation for input text.
 * @param {string} text - Input text to explain
 * @returns {Promise<Object>} Explanation result with tokens, importances, summary
 */
export const explainText = async (text) => {
  try {
    const response = await api.post('/explain', { text });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Explanation failed');
    } else if (error.request) {
      throw new Error('Could not connect to backend. Make sure the server is running on port 8000.');
    } else {
      throw new Error(error.message || 'Explanation failed');
    }
  }
};

/**
 * Handle speech input (with transcript).
 * @param {string} transcript - Speech transcript
 * @returns {Promise<Object>} Prediction result
 */
export const speechInput = async (transcript) => {
  try {
    const response = await api.post('/speech', { transcript });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Speech processing failed');
    } else if (error.request) {
      throw new Error('Could not connect to backend. Make sure the server is running on port 8000.');
    } else {
      throw new Error(error.message || 'Speech processing failed');
    }
  }
};

export default api;

