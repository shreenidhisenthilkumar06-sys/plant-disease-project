import { NavLink } from 'react-router-dom';

export default function Navbar({ theme, onThemeToggle }) {
  return <header className="navbar"><NavLink className="brand" to="/">Leaf<span>Lens</span></NavLink><nav aria-label="Main navigation"><NavLink to="/">Home</NavLink><NavLink to="/predict">Diagnose</NavLink><NavLink to="/research">Research</NavLink></nav><button className="theme-toggle" onClick={onThemeToggle} aria-label="Toggle dark mode">{theme === 'dark' ? '☀️' : '🌙'}</button></header>;
}
