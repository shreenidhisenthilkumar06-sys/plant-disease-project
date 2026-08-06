import { Link, useLocation } from 'react-router-dom';
import ResultCard from '../components/ResultCard';

export default function ResultPage() {
  const location = useLocation(); const stored = sessionStorage.getItem('latestPrediction'); const saved = stored ? JSON.parse(stored) : null; const data = location.state || saved;
  if (!data?.result || !data?.imageUrl) return <section className="centered-page empty-state"><h1>No prediction to display</h1><p>Upload an image to start a new analysis.</p><Link className="button button-primary" to="/predict">Analyze a leaf</Link></section>;
  return <section className="result-page"><Link className="back-link" to="/predict">← Analyze another leaf</Link><ResultCard result={data.result} imageUrl={data.imageUrl} /></section>;
}
