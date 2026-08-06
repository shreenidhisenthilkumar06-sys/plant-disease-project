import { useEffect, useState } from 'react';
import { getResearchAssets, researchAssetUrl } from '../services/api';
import LoadingSpinner from '../components/LoadingSpinner';

function explanation(name) { const lower = name.toLowerCase(); if (lower.includes('history')) return 'Training and validation trends show how performance changed across epochs.'; if (lower.includes('evaluation') || lower.includes('confusion')) return 'Evaluation output summarizes how consistently the model separated classes across test data.'; if (lower.includes('novelty')) return 'Novelty experiment output documents image-selection and optimization comparisons.'; if (lower.includes('feature') || lower.includes('map')) return 'Feature maps reveal the visual patterns emphasized by intermediate model layers.'; return 'Research artifact generated during model development and evaluation.'; }

export default function ResearchPage() {
  const [assets, setAssets] = useState([]); const [status, setStatus] = useState('loading');
  useEffect(() => { getResearchAssets().then((items) => { setAssets(items); setStatus('ready'); }).catch(() => setStatus('error')); }, []);
  return <section className="research-page"><p className="eyebrow">Model development</p><h1>Research</h1><p className="lead">Training history, evaluation figures, novelty experiments, and feature visualizations are served directly from the untouched research output.</p>{status === 'loading' && <LoadingSpinner />}{status === 'error' && <p className="form-error">Research images are unavailable because the backend could not be reached.</p>}{status === 'ready' && (assets.length ? <div className="research-grid">{assets.map((asset) => <figure className="card research-item" key={asset}><img src={researchAssetUrl(asset)} alt={asset.replaceAll(/[_-]/g, ' ')} /><figcaption><strong>{asset.split('/').at(-1).replaceAll(/[_-]/g, ' ')}</strong><span>{explanation(asset)}</span></figcaption></figure>)}</div> : <div className="card empty-state"><h2>No research images found</h2><p>Add your existing graphs and experiment images to the research folder; this gallery discovers them automatically.</p></div>)}</section>;
}
