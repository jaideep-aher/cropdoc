import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Leaf, AlertTriangle, CheckCircle, ChevronDown, X, Loader2, Sprout, FlaskConical, Bug, ShieldCheck } from 'lucide-react'

const SEVERITY_CONFIG = {
  mild:     { color: 'text-emerald-400', bg: 'bg-emerald-400/10 border-emerald-400/30', label: 'Mild',     icon: CheckCircle },
  moderate: { color: 'text-amber-400',   bg: 'bg-amber-400/10 border-amber-400/30',     label: 'Moderate', icon: AlertTriangle },
  severe:   { color: 'text-red-400',     bg: 'bg-red-400/10 border-red-400/30',          label: 'Severe',   icon: AlertTriangle },
}

function ConfidenceBar({ value, label, color = 'bg-leaf-500' }) {
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="w-52 truncate text-slate-400 text-xs">{label}</span>
      <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>
      <span className="w-12 text-right text-slate-300 text-xs">{(value * 100).toFixed(1)}%</span>
    </div>
  )
}

function TreatmentSection({ icon: Icon, title, items, colorClass }) {
  const [open, setOpen] = useState(true)
  if (!items?.length) return null
  return (
    <div className="border border-slate-800 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-900 hover:bg-slate-800 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon size={16} className={colorClass} />
          <span className="font-medium text-sm">{title}</span>
          <span className="text-xs text-slate-500 ml-1">({items.length})</span>
        </div>
        <ChevronDown size={16} className={`text-slate-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="px-4 py-3 space-y-2 bg-slate-900/50 overflow-hidden"
          >
            {items.map((item, i) => (
              <li key={i} className="flex gap-2 text-sm text-slate-300">
                <span className={`mt-1 shrink-0 w-1.5 h-1.5 rounded-full ${colorClass.replace('text-', 'bg-')}`} />
                {item}
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}

function ResultPanel({ result, onReset }) {
  const sev = SEVERITY_CONFIG[result.severity] || SEVERITY_CONFIG.mild
  const SevIcon = sev.icon

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-5"
    >
      {/* Header */}
      <div className={`flex items-start gap-4 p-5 rounded-2xl border ${sev.bg}`}>
        <div className="flex-1">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-widest mb-1">Detection Result</p>
          <h2 className="text-2xl font-bold text-white leading-tight">{result.display_name}</h2>
          <div className="flex items-center gap-3 mt-2">
            <div className="flex items-center gap-1.5">
              <SevIcon size={14} className={sev.color} />
              <span className={`text-sm font-semibold ${sev.color}`}>{sev.label} Severity</span>
            </div>
            <span className="text-slate-600">·</span>
            <span className="text-sm text-slate-400">{(result.confidence * 100).toFixed(1)}% confidence</span>
          </div>
        </div>
        <button onClick={onReset} className="p-2 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors">
          <X size={18} />
        </button>
      </div>

      {/* Note */}
      {result.treatment.note && (
        <div className="flex gap-3 p-4 bg-amber-400/5 border border-amber-400/20 rounded-xl text-sm text-amber-200">
          <AlertTriangle size={16} className="text-amber-400 shrink-0 mt-0.5" />
          {result.treatment.note}
        </div>
      )}

      {/* Treatment sections */}
      {!result.is_healthy && (
        <div className="space-y-3">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Treatment Plan</p>
          <TreatmentSection icon={FlaskConical} title="Chemical Controls" items={result.treatment.chemical_controls} colorClass="text-blue-400" />
          <TreatmentSection icon={Bug}          title="Biological Controls" items={result.treatment.biological_controls} colorClass="text-emerald-400" />
          <TreatmentSection icon={Sprout}       title="Cultural Practices"  items={result.treatment.cultural_controls}  colorClass="text-amber-400" />
          <TreatmentSection icon={ShieldCheck}  title="Prevention"          items={result.treatment.prevention}          colorClass="text-violet-400" />
        </div>
      )}

      {/* Confidence breakdown */}
      <div>
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">Confidence Breakdown</p>
        <div className="space-y-2 bg-slate-900/60 rounded-xl p-4 border border-slate-800">
          {result.top_k_predictions.map((p, i) => (
            <ConfidenceBar
              key={p.class}
              label={p.class.replace(/___/g, ' — ').replace(/_/g, ' ')}
              value={p.probability}
              color={i === 0 ? 'bg-leaf-500' : 'bg-slate-600'}
            />
          ))}
        </div>
      </div>
    </motion.div>
  )
}

function DropZone({ onFile }) {
  const inputRef = useRef()
  const [dragging, setDragging] = useState(false)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file?.type.startsWith('image/')) onFile(file)
  }, [onFile])

  return (
    <label
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`cursor-pointer flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-14 transition-all duration-200
        ${dragging ? 'border-leaf-400 bg-leaf-500/10' : 'border-slate-700 hover:border-leaf-600 hover:bg-leaf-500/5'}`}
    >
      <div className={`p-5 rounded-full transition-colors ${dragging ? 'bg-leaf-500/20' : 'bg-slate-800'}`}>
        <Upload size={32} className={dragging ? 'text-leaf-400' : 'text-slate-400'} />
      </div>
      <div className="text-center">
        <p className="font-semibold text-slate-200">Drop a leaf photo here</p>
        <p className="text-sm text-slate-500 mt-1">or click to browse — JPEG / PNG, max 10 MB</p>
      </div>
      <span className="px-4 py-2 rounded-lg bg-leaf-600 hover:bg-leaf-500 text-white text-sm font-medium transition-colors">
        Choose Image
      </span>
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={e => { const f = e.target.files[0]; if (f) onFile(f) }} />
    </label>
  )
}

export default function App() {
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFile = useCallback(async (file) => {
    setResult(null)
    setError(null)
    setPreview(URL.createObjectURL(file))
    setLoading(true)

    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch('/predict', { method: 'POST', body: form })
      if (!res.ok) throw new Error((await res.json()).detail || 'Server error')
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = () => { setPreview(null); setResult(null); setError(null) }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Nav */}
      <nav className="border-b border-slate-800 px-6 py-4 flex items-center gap-3">
        <div className="p-2 rounded-lg bg-leaf-600/20">
          <Leaf size={20} className="text-leaf-400" />
        </div>
        <div>
          <span className="font-bold text-lg tracking-tight">CropDoc</span>
          <span className="ml-2 text-xs text-slate-500 font-medium">Plant Disease AI</span>
        </div>
        <div className="ml-auto flex items-center gap-2 text-xs text-slate-500 border border-slate-800 rounded-lg px-3 py-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-leaf-500 animate-pulse" />
          38 disease classes · EfficientNetV2-S
        </div>
      </nav>

      {/* Main */}
      <main className="max-w-5xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold tracking-tight mb-3">
            Diagnose plant disease{' '}
            <span className="text-leaf-400">instantly</span>
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto">
            Upload a leaf photo — CropDoc identifies the disease and gives you a prioritized treatment plan in seconds.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: upload + preview */}
          <div className="space-y-5">
            {!preview ? (
              <DropZone onFile={handleFile} />
            ) : (
              <div className="relative rounded-2xl overflow-hidden border border-slate-800">
                <img src={preview} alt="Uploaded leaf" className="w-full object-cover max-h-80" />
                {loading && (
                  <div className="absolute inset-0 bg-slate-950/70 flex flex-col items-center justify-center gap-3">
                    <Loader2 size={32} className="text-leaf-400 animate-spin" />
                    <p className="text-sm text-slate-300 font-medium">Analyzing leaf...</p>
                  </div>
                )}
                {!loading && (
                  <button
                    onClick={reset}
                    className="absolute top-3 right-3 p-2 rounded-lg bg-slate-900/80 hover:bg-slate-900 text-slate-300 hover:text-white transition-colors backdrop-blur-sm"
                  >
                    <X size={16} />
                  </button>
                )}
              </div>
            )}

            {/* Supported crops */}
            <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-xs text-slate-500">
              <p className="font-semibold text-slate-400 mb-2">Supported crops</p>
              <p>Apple · Blueberry · Cherry · Corn · Grape · Orange · Peach · Bell pepper · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato</p>
            </div>
          </div>

          {/* Right: results */}
          <div>
            {error && (
              <div className="flex gap-3 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-sm text-red-300">
                <AlertTriangle size={16} className="text-red-400 shrink-0 mt-0.5" />
                {error}
              </div>
            )}
            {result && <ResultPanel result={result} onReset={reset} />}
            {!result && !error && !loading && (
              <div className="h-full flex flex-col items-center justify-center gap-4 py-16 text-center text-slate-600">
                <Leaf size={48} className="opacity-20" />
                <p className="text-sm">Results will appear here after you upload a photo</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="border-t border-slate-800 text-center py-6 text-xs text-slate-600">
        CropDoc · Built for AIPI 540 · PlantVillage dataset · EfficientNetV2-S · Not a substitute for agronomist advice
      </footer>
    </div>
  )
}
