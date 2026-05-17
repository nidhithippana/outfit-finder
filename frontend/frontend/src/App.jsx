import { useState, useRef } from "react";

const BEIGE = "#F2EDE4";
const BEIGE_CARD = "#EDE8DF";
const BEIGE_BORDER = "#D9D2C5";
const BLACK = "#1A1A1A";
const GRAY = "#6B6560";
const ACCENT = "#8B7355";

function TshirtSVG({ size = 180, style = {} }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none"
      stroke="#1A1A1A" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
      style={style}>
      {/* sleeves + body */}
      <path d="M20 10 L5 30 L20 35 L20 85 L80 85 L80 35 L95 30 L80 10 Q70 20 50 20 Q30 20 20 10Z" />
    </svg>
  );
}

function DressSVG({ size = 160, style = {} }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 120" fill="none"
      stroke="#1A1A1A" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
      style={style}>
      <path d="M35 5 Q50 18 65 5 L75 35 Q60 40 55 50 L70 115 L30 115 L45 50 Q40 40 25 35 Z" />
    </svg>
  );
}

function PantsSVG({ size = 150, style = {} }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 120" fill="none"
      stroke="#1A1A1A" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
      style={style}>
      <path d="M15 5 L85 5 L85 15 L75 15 L75 60 Q72 90 65 115 L50 115 L45 60 L55 60 L50 115 L35 115 Q28 90 25 60 L25 15 L15 15 Z" />
    </svg>
  );
}

function BackgroundArt() {
  return (
    <div style={{ position: "fixed", inset: 0, overflow: "hidden", pointerEvents: "none", zIndex: 0 }}>
      <TshirtSVG size={220} style={{ position: "absolute", top: "4%",  left: "-3%",  opacity: 0.06, transform: "rotate(-12deg)" }} />
      <DressSVG  size={200} style={{ position: "absolute", top: "2%",  right: "2%",  opacity: 0.06, transform: "rotate(10deg)" }} />
      <PantsSVG  size={180} style={{ position: "absolute", top: "38%", left: "1%",   opacity: 0.05, transform: "rotate(8deg)" }} />
      <TshirtSVG size={160} style={{ position: "absolute", top: "35%", right: "-1%", opacity: 0.05, transform: "rotate(-6deg)" }} />
      <DressSVG  size={170} style={{ position: "absolute", bottom: "5%", left: "3%", opacity: 0.06, transform: "rotate(14deg)" }} />
      <TshirtSVG size={200} style={{ position: "absolute", bottom: "2%", right: "4%", opacity: 0.05, transform: "rotate(-8deg)" }} />
      <PantsSVG  size={150} style={{ position: "absolute", top: "68%",  left: "45%", opacity: 0.04, transform: "rotate(5deg)" }} />
    </div>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef();

  const handleFile = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleFileChange = (e) => handleFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) { alert("Please select an image first."); return; }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await fetch("http://127.0.0.1:8000/analyze", { method: "POST", body: formData });
      if (!response.ok) throw new Error("Server error. Please try again.");
      setResult(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", backgroundColor: BEIGE, fontFamily: "'Georgia', serif", color: BLACK, position: "relative" }}>
      <BackgroundArt />
      <div style={{ maxWidth: "960px", margin: "0 auto", padding: "3rem 2rem", position: "relative", zIndex: 1 }}>

        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: "3rem" }}>
          <h1 style={{ fontSize: "2.2rem", fontWeight: 700, letterSpacing: "0.08em", margin: 0, color: BLACK }}>
            OUTFIT FINDER
          </h1>
          <p style={{ fontSize: "0.9rem", color: GRAY, marginTop: "0.5rem", letterSpacing: "0.12em" }}>
            DISCOVER YOUR STYLE
          </p>
          <div style={{ width: "40px", height: "1px", backgroundColor: ACCENT, margin: "1rem auto 0" }} />
        </div>

        {/* Upload area */}
        <div
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          style={{
            border: `1.5px dashed ${dragOver ? ACCENT : BEIGE_BORDER}`,
            borderRadius: "12px",
            padding: "2.5rem",
            textAlign: "center",
            cursor: "pointer",
            backgroundColor: dragOver ? "#EDE5D8" : BEIGE_CARD,
            transition: "all 0.2s",
            marginBottom: "1.5rem",
          }}
        >
          <input ref={inputRef} type="file" accept="image/*" onChange={handleFileChange} style={{ display: "none" }} />
          {previewUrl ? (
            <img src={previewUrl} alt="Preview" style={{ maxHeight: "280px", maxWidth: "100%", borderRadius: "8px", objectFit: "contain" }} />
          ) : (
            <>
              <div style={{ fontSize: "2rem", marginBottom: "0.75rem", color: ACCENT }}>↑</div>
              <p style={{ margin: 0, color: GRAY, fontSize: "0.95rem" }}>Drop an image here, or click to browse</p>
            </>
          )}
        </div>

        {previewUrl && (
          <p style={{ textAlign: "center", fontSize: "0.8rem", color: GRAY, marginBottom: "1.5rem", letterSpacing: "0.05em" }}>
            {selectedFile?.name}
            <span
              onClick={() => { setSelectedFile(null); setPreviewUrl(null); setResult(null); }}
              style={{ marginLeft: "0.75rem", cursor: "pointer", color: ACCENT, textDecoration: "underline" }}
            >
              remove
            </span>
          </p>
        )}

        <div style={{ textAlign: "center", marginBottom: "2.5rem" }}>
          <button
            onClick={handleUpload}
            disabled={loading || !selectedFile}
            style={{
              padding: "0.75rem 2.5rem",
              backgroundColor: loading || !selectedFile ? BEIGE_BORDER : BLACK,
              color: loading || !selectedFile ? GRAY : BEIGE,
              border: "none",
              borderRadius: "6px",
              cursor: loading || !selectedFile ? "not-allowed" : "pointer",
              fontSize: "0.85rem",
              letterSpacing: "0.15em",
              fontFamily: "inherit",
              transition: "all 0.2s",
            }}
          >
            {loading ? (
              <>
                ANALYZING<span className="dot">.</span><span className="dot">.</span><span className="dot">.</span>
              </>
            ) : "ANALYZE OUTFIT"}
          </button>
        </div>

        {error && (
          <p style={{ color: "#8B4040", textAlign: "center", marginBottom: "2rem" }}>{error}</p>
        )}

        {/* Results */}
        {result && (
          <>
            {/* Detected items */}
            <div style={{ backgroundColor: BEIGE_CARD, borderRadius: "12px", padding: "1.5rem 2rem", marginBottom: "2.5rem", border: `1px solid ${BEIGE_BORDER}` }}>
              <h2 style={{ fontSize: "0.75rem", letterSpacing: "0.2em", margin: "0 0 1rem", color: GRAY }}>DETECTED ITEMS</h2>
              {result.detected_items?.map((item, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
                  <span style={{ fontWeight: 600, textTransform: "capitalize", fontSize: "1rem" }}>{item.category}</span>
                  {item.color && <span style={{ color: GRAY, fontSize: "0.9rem" }}>· {item.color}</span>}
                  {item.style_attributes?.length > 0 && (
                    <span style={{ color: GRAY, fontSize: "0.8rem" }}>· {item.style_attributes.join(", ")}</span>
                  )}
                </div>
              ))}
              {result.style && (
                <p style={{ marginTop: "0.75rem", marginBottom: 0, fontSize: "0.85rem", color: ACCENT, fontStyle: "italic" }}>
                  {result.style}
                </p>
              )}
            </div>

            {/* Recommendations */}
            <h2 style={{ fontSize: "0.75rem", letterSpacing: "0.2em", marginBottom: "1.5rem", color: GRAY }}>SHOP SIMILAR ON SHEIN</h2>

            {result.recommendations && Object.entries(result.recommendations).map(([category, products]) => (
              <div key={category} style={{ marginBottom: "2.5rem" }}>
                <h3 style={{ fontSize: "0.85rem", letterSpacing: "0.15em", textTransform: "capitalize", marginBottom: "1rem", color: BLACK, fontWeight: 400 }}>
                  {category}
                </h3>

                {products.length === 0 ? (
                  <p style={{ color: GRAY, fontSize: "0.9rem" }}>No matches found.</p>
                ) : (
                  <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                    {products.map((product, i) => (
                      <div key={i} style={{
                        backgroundColor: BEIGE_CARD,
                        border: `1px solid ${BEIGE_BORDER}`,
                        borderRadius: "10px",
                        padding: "0.875rem",
                        width: "165px",
                        display: "flex",
                        flexDirection: "column",
                        gap: "0.5rem",
                      }}>
                        {product.tag && (
                          <span style={{
                            backgroundColor: ACCENT, color: BEIGE,
                            fontSize: "0.65rem", fontWeight: 600,
                            padding: "2px 7px", borderRadius: "4px",
                            alignSelf: "flex-start", letterSpacing: "0.08em",
                          }}>
                            {product.tag}
                          </span>
                        )}
                        {product.image_url && (
                          <img
                            src={product.image_url}
                            alt={product.name}
                            style={{ width: "100%", height: "185px", objectFit: "cover", borderRadius: "6px" }}
                          />
                        )}
                        <p style={{ fontSize: "0.8rem", margin: 0, fontWeight: 500, lineHeight: "1.35", color: BLACK }}>
                          {product.name}
                        </p>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
                          {product.price && (
                            <span style={{ fontSize: "0.9rem", color: ACCENT, fontWeight: 600 }}>{product.price}</span>
                          )}
                          {product.original_price && (
                            <span style={{ fontSize: "0.75rem", color: GRAY, textDecoration: "line-through" }}>{product.original_price}</span>
                          )}
                        </div>
                        {product.product_url && (
                          <a
                            href={product.product_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{
                              display: "block",
                              textAlign: "center",
                              backgroundColor: BLACK,
                              color: BEIGE,
                              padding: "0.45rem 0",
                              borderRadius: "5px",
                              fontSize: "0.72rem",
                              letterSpacing: "0.1em",
                              textDecoration: "none",
                              marginTop: "auto",
                            }}
                          >
                            SHOP NOW
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
