import { Link } from 'react-router-dom';
export default function NotFoundPage() { return <section className="centered-page empty-state"><p className="eyebrow">404</p><h1>That leaf has drifted away.</h1><p>The page you requested does not exist.</p><Link to="/" className="button button-primary">Return home</Link></section>; }
