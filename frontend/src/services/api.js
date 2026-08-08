import axios from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '"https://plant-disease-project-0xbv.onrender.com"';
const api = axios.create({ baseURL: API_BASE_URL, timeout: 25000 });

function requestMessage(error) {
  console.log('API ERROR', error);
  console.log('API RESPONSE', error.response);

  if (error.code === 'ECONNABORTED') {
    return 'The prediction timed out. Please try again.';
  }

  if (!error.response) {
    return 'The backend is unavailable. Start the API and try again.';
  }

  return (
    error.response.data?.detail ||
    JSON.stringify(error.response.data) ||
    'Unable to analyze this image. Please try another one.'
  );
}

export async function predictImage(file) {
  const body = new FormData();
  body.append('image', file);
  try {
    const { data } = await api.post('/predict', body);
    return data;
  } catch (error) {
    throw new Error(requestMessage(error));
  }
}

export async function getResearchAssets() {
  try {
    const { data } = await api.get('/research-assets');
    return data.assets;
  } catch (error) {
    throw new Error(requestMessage(error));
  }
}

export function researchAssetUrl(assetPath) {
  return `${API_BASE_URL}/research-assets/${assetPath.split('/').map(encodeURIComponent).join('/')}`;
}
