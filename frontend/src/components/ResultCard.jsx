const labels = [['Description', 'description'], ['Symptoms', 'symptoms'], ['Likely cause', 'causes'], ['Prevention', 'prevention'], ['Treatment', 'treatment']];

export default function ResultCard({ result, imageUrl }) {
  const healthy = result.prediction.toLowerCase().startsWith('healthy');
  return <section className="result-layout"><div className="image-card"><img src={imageUrl} alt="Analyzed leaf" /></div><div className="result-summary card"><span className={`badge ${healthy ? 'healthy' : 'diseased'}`}>{healthy ? 'Healthy leaf' : 'Disease detected'}</span><h1>{result.prediction}</h1><div className="confidence"><div><span>Confidence</span><strong>{result.confidence.toFixed(2)}%</strong></div><div className="confidence-track"><i style={{ width: `${result.confidence}%` }} /></div></div></div><div className="detail-grid">{labels.map(([label, key]) => <article className="card detail-card" key={key}><h2>{label}</h2><p>{result[key]}</p></article>)}</div></section>;
}
