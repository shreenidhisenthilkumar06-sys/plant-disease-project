import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <>
      <section className="hero">
        <p className="eyebrow">Plant health, made clearer</p>

        <h1>Understand what your grape leaf is telling you.</h1>

        <p>
          Upload a clear grape leaf photo for a fast, TensorFlow-powered disease
          assessment and practical care guidance.
        </p>

        <div className="card" style={{ marginTop: '1.5rem', textAlign: 'left' }}>
          <p style={{ margin: 0, fontWeight: 700 }}>🍇 Supported input</p>
          <p style={{ marginTop: '0.5rem' }}>
            This deployed version currently supports <strong>grape leaf disease detection only</strong>.
            Uploading leaves from apple, peach, corn, other plants, fruits, objects, or unrelated photos may
            produce incorrect predictions.
          </p>

          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginTop: '0.75rem' }}>
            <span className="tag">Healthy</span>
            <span className="tag">Black rot</span>
            <span className="tag">Esca</span>
            <span className="tag">Leaf blight</span>
          </div>
        </div>

        <Link className="button button-primary" to="/predict">
          Analyze a grape leaf <span>→</span>
        </Link>
      </section>

      <section className="feature-grid">
        {[
          ['Fast analysis', 'Get a model prediction in seconds.'],
          ['Actionable guidance', 'See symptoms, prevention, and treatment context.'],
          ['Grape-focused', 'Optimized for grape leaf disease images only.'],
        ].map(([title, text]) => (
          <article className="card" key={title}>
            <h2>{title}</h2>
            <p>{text}</p>
          </article>
        ))}
      </section>

      <section className="overview card">
        <h2>How it works</h2>

        <ol>
          <li>Upload a sharp, well-lit image of a <strong>single grape leaf</strong>.</li>
          <li>The trained TensorFlow model evaluates its visual disease pattern.</li>
          <li>Review the predicted class and disease-management guidance.</li>
        </ol>

        <div
          style={{
            marginTop: '1rem',
            padding: '1rem',
            borderRadius: '1rem',
            background: 'rgba(245, 158, 11, 0.08)',
            border: '1px solid rgba(245, 158, 11, 0.2)',
            textAlign: 'left',
          }}
        >
          <p style={{ margin: 0, fontWeight: 700 }}>⚠️ Best results</p>
          <ul style={{ marginTop: '0.5rem', paddingLeft: '1.2rem' }}>
            <li>Use a single grape leaf</li>
            <li>Capture the leaf in good lighting</li>
            <li>Keep the leaf in focus</li>
            <li>Let the leaf fill most of the frame</li>
          </ul>
        </div>

        <p className="fine-print">
          Predictions are decision support, not a substitute for a local plant-health professional.
        </p>
      </section>
    </>
  );
}