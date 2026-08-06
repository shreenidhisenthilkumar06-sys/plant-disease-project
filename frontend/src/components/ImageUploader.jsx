import { useEffect, useRef, useState } from 'react';

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE = 10 * 1024 * 1024;

export default function ImageUploader({ onFileChange, file, disabled }) {
  const inputRef = useRef(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');
  useEffect(() => {
    if (!file) return undefined;
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);
  const choose = (selected) => {
    if (!selected) return;
    if (!ALLOWED_TYPES.includes(selected.type)) return setError('Choose a JPG, PNG, or WebP image.');
    if (selected.size > MAX_SIZE) return setError('Image must be 10 MB or smaller.');
    setError(''); onFileChange(selected);
  };
  return <section className="upload-section">
    <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files[0]); }}>
      <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => choose(event.target.files[0])} hidden />
      {file ? <img className="upload-preview" src={previewUrl} alt="Selected leaf" /> : <><div className="upload-icon">⌁</div><h3>Drop a leaf image here</h3><p>JPG, PNG, or WebP · up to 10 MB</p></>}
      <button type="button" className="button button-secondary" onClick={() => inputRef.current?.click()} disabled={disabled}>{file ? 'Choose another image' : 'Browse files'}</button>
    </div>
    {error && <p className="form-error" role="alert">{error}</p>}
  </section>;
}
