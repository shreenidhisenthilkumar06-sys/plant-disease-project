import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ImageUploader from '../components/ImageUploader';
import LoadingSpinner from '../components/LoadingSpinner';
import { API_BASE_URL, predictImage } from '../services/api';


export default function PredictPage() {
  const [file, setFile] = useState(null); const [loading, setLoading] = useState(false); const [error, setError] = useState(''); const navigate = useNavigate();
  useEffect(() => {
  // Wake up the Render backend as soon as the Diagnose page opens
  fetch(`${API_BASE_URL}/health`).catch(() => {});
  }, []);
  const submit = async () => { if (!file) return setError('Select an image before running the analysis.'); setLoading(true); setError(''); try { const result = await predictImage(file); const imageUrl = URL.createObjectURL(file); sessionStorage.setItem('latestPrediction', JSON.stringify({ result, imageUrl })); navigate('/result', { state: { result, imageUrl } }); } catch (requestError) { setError(requestError.message); } finally { setLoading(false); } };
  return <section className="centered-page"><p className="eyebrow">Leaf diagnosis</p><h1>Upload a leaf image</h1><p className="lead">For the best result, use one in-focus leaf against a simple background.</p><div className="upload-card card"><ImageUploader file={file} onFileChange={setFile} disabled={loading} /><button className="button button-primary predict-button" disabled={loading || !file} onClick={submit}>{loading ? <LoadingSpinner /> : 'Predict disease'}</button>{error && <p className="form-error" role="alert">{error}</p>}</div></section>;
}
