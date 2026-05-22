import { useState, useEffect, useMemo, useRef } from 'react'
import RecsPanel from './RecsPanel'

// Número máximo de filas en estado: con 800k filas React se volvería lento
const MAX_ROWS = 500

function ClusterChart({ stats }) {
  if (!stats || stats.by_cluster.length === 0) return null
  const max = Math.max(...stats.by_cluster.map(c => c.n), 1)
  return (
    <div className="cluster-chart">
      <div className="chart-label">DISTRIBUCIÓN POR CLUSTER</div>
      {stats.by_cluster.map(({ cluster, n }) => (
        <div key={cluster} className="chart-row">
          <span className="chart-cl">{cluster}</span>
          <div className="chart-bar-wrap">
            <div className="chart-bar" style={{ width: `${(n / max) * 100}%` }} />
          </div>
          <span className="chart-n">{n}</span>
        </div>
      ))}
    </div>
  )
}

export default function LiveTab({ countries, prices }) {
  const [rows, setRows]               = useState([])
  const [connected, setConnected]     = useState(false)
  const [selectedId, setSelectedId]   = useState(null)
  const [selectedName, setSelectedName] = useState('')
  const [stats, setStats]             = useState(null)
  const [filterTotal, setFilterTotal] = useState(0)
  const [cities, setCities]           = useState([])
  const [cityInput, setCityInput]     = useState('')
  const [filters, setFilters]         = useState({
    country: 'Todos', city: 'Todos', price: 'Todos', min_rating: '0',
    veg: false, vegan: false, gf: false,
  })

  // Refs para que los handlers de SSE lean el valor actual sin cerrarse sobre el estado inicial
  const selectedIdRef = useRef(selectedId)
  useEffect(() => { selectedIdRef.current = selectedId }, [selectedId])

  const filtersRef = useRef(filters)
  useEffect(() => { filtersRef.current = filters }, [filters])

  // Cuando los filtros cambian, snapshotUrl cambia y dispara el re-fetch del snapshot.
  // El servidor filtra sobre los 800k; el cliente no puede hacerlo porque el snapshot
  // solo trae los últimos N cronológicos y pueden no incluir el país/ciudad buscado.
  const snapshotUrl = useMemo(() => {
    const p = new URLSearchParams({
      limit: 500,
      country:    filters.country,
      city:       filters.city,
      price:      filters.price,
      min_rating: filters.min_rating,
      veg:        filters.veg,
      vegan:      filters.vegan,
      gf:         filters.gf,
    })
    return `/api/stream/snapshot?${p}`
  }, [filters])

  useEffect(() => {
    fetch(snapshotUrl).then(r => r.json()).then(setRows)
  }, [snapshotUrl])

  // Ciudades disponibles para el país seleccionado
  useEffect(() => {
    if (filters.country === 'Todos') { setCities([]); return }
    fetch(`/api/cities?country=${encodeURIComponent(filters.country)}`)
      .then(r => r.json())
      .then(setCities)
  }, [filters.country])

  const handleCityInput = (value) => {
    setCityInput(value)
    if (value === '') setFilter('city', 'Todos')
    else if (cities.includes(value)) setFilter('city', value)
  }

  useEffect(() => {
    const load = () => {
      fetch('/api/stream/stats').then(r => r.json()).then(setStats)
      fetch('/api/filter/stats').then(r => r.json()).then(d => setFilterTotal(d.total))
    }
    load()
    const id = setInterval(load, 5000)
    return () => clearInterval(id)
  }, [])

  // SSE: solo toca rows, nunca selectedId. Diferencia clave con Streamlit, donde cualquier
  // rerender borraba la selección. Aquí el estado de selección es completamente independiente.
  useEffect(() => {
    const es = new EventSource('/api/stream/events')
    es.onopen = () => setConnected(true)
    es.onmessage = (e) => {
      const incoming = JSON.parse(e.data)
      if (!Array.isArray(incoming) || incoming.length === 0) return
      setRows(prev => {
        const existing = new Set(prev.map(r => r.row_id))
        const f    = filtersRef.current
        const minR = parseFloat(f.min_rating) || 0
        // Filtrar las filas del SSE antes de añadirlas: si entraran sin filtrar,
        // al podar a MAX_ROWS irían desplazando las filas del snapshot hasta vaciar la tabla
        const fresh = incoming
          .filter(r => !existing.has(r.row_id))
          .filter(r => {
            if (f.country !== 'Todos' && r.country !== f.country) return false
            if (f.city    !== 'Todos' && r.city    !== f.city)    return false
            if (f.price   !== 'Todos' && r.price   !== f.price)   return false
            if (f.veg   && !r.veg)   return false
            if (f.vegan && !r.vegan) return false
            if (f.gf    && !r.gf)    return false
            if (minR > 0 && (parseFloat(r.rating) || 0) < minR)  return false
            return true
          })
        if (fresh.length === 0) return prev
        const combined = [...fresh, ...prev]
        if (combined.length <= MAX_ROWS) return combined
        // Al podar, conservar siempre la fila seleccionada para que no desaparezca
        const selId = selectedIdRef.current
        if (!selId) return combined.slice(0, MAX_ROWS)
        const sel  = combined.find(r => r.row_id === selId)
        const rest = combined.filter(r => r.row_id !== selId).slice(0, MAX_ROWS - 1)
        return sel ? [sel, ...rest] : combined.slice(0, MAX_ROWS)
      })
    }
    es.onerror = () => { setConnected(false); es.close() }
    return () => es.close()
  }, [])

  // Filtrado cliente: cubre las filas del SSE añadidas después del último snapshot
  const filteredRows = useMemo(() => {
    const minR = parseFloat(filters.min_rating) || 0
    return rows.filter(row => {
      if (filters.country !== 'Todos' && row.country !== filters.country) return false
      if (filters.city    !== 'Todos' && row.city    !== filters.city)    return false
      if (filters.price   !== 'Todos' && row.price   !== filters.price)   return false
      if (filters.veg   && !row.veg)   return false
      if (filters.vegan && !row.vegan) return false
      if (filters.gf    && !row.gf)    return false
      if (minR > 0 && (parseFloat(row.rating) || 0) < minR) return false
      return true
    })
  }, [rows, filters])

  // La fila seleccionada puede quedar fuera de los primeros 200 al llegar nuevas filas;
  // se ancla al inicio para que siempre sea visible y el highlight no desaparezca
  const displayRows = useMemo(() => {
    const limited = filteredRows.slice(0, 200)
    if (!selectedId || limited.some(r => r.row_id === selectedId)) return limited
    const sel = rows.find(r => r.row_id === selectedId)
    return sel ? [sel, ...limited.slice(0, 199)] : limited
  }, [filteredRows, selectedId, rows])

  const setFilter = (key, value) => setFilters(f => ({ ...f, [key]: value }))

  const handleCountryChange = (value) => {
    setFilters(f => ({ ...f, country: value, city: 'Todos' }))
    setCityInput('')
  }

  const handleSelect = (row) => {
    setSelectedId(row.row_id)
    setSelectedName(row.name)
  }

  return (
    <div className="live">
      <div className="live-header">
        <div className="stats">
          <span className="stat-n">{filterTotal.toLocaleString()}</span>
          <span className="stat-l">procesados</span>
        </div>
        <div className="stats">
          <span className="stat-n">{(stats?.total ?? 0).toLocaleString()}</span>
          <span className="stat-l">scoreados</span>
        </div>
        <span className={`badge ${connected ? 'badge-live' : 'badge-off'}`}>
          {connected ? '● EN VIVO' : '○ desconectado'}
        </span>
      </div>

      <div className="filters">
        <select className="select" value={filters.country} onChange={e => handleCountryChange(e.target.value)}>
          <option value="Todos">Todos los países</option>
          {(countries || []).map(c => <option key={c}>{c}</option>)}
        </select>
        <input
          className="select"
          list="cities-live"
          placeholder="Ciudad…"
          value={cityInput}
          disabled={filters.country === 'Todos'}
          onChange={e => handleCityInput(e.target.value)}
        />
        <datalist id="cities-live">
          {cities.map(c => <option key={c} value={c} />)}
        </datalist>
        <select className="select" value={filters.price} onChange={e => setFilter('price', e.target.value)}>
          <option value="Todos">Cualquier precio</option>
          {(prices || []).map(p => <option key={p}>{p}</option>)}
        </select>
        <select className="select" value={filters.min_rating} onChange={e => setFilter('min_rating', e.target.value)}>
          <option value="0">Cualquier rating</option>
          <option value="3">≥ 3.0</option>
          <option value="3.5">≥ 3.5</option>
          <option value="4">≥ 4.0</option>
          <option value="4.5">≥ 4.5</option>
        </select>
        <label className="checkbox-label">
          <input type="checkbox" checked={filters.veg}
            onChange={e => setFilter('veg', e.target.checked)} />
          Vegetariano
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={filters.vegan}
            onChange={e => setFilter('vegan', e.target.checked)} />
          Vegano
        </label>
        <label className="checkbox-label">
          <input type="checkbox" checked={filters.gf}
            onChange={e => setFilter('gf', e.target.checked)} />
          Sin gluten
        </label>
        <span className="filter-count">
          {filteredRows.length.toLocaleString()} con filtros
        </span>
      </div>

      <div className="content-split">
        <div className="table-wrapper">
          {/* table-layout fixed evita que las columnas cambien de ancho al actualizar filas */}
          <table className="data-table live-table">
            <colgroup>
              <col style={{ width: '22%' }} />
              <col style={{ width: '11%' }} />
              <col style={{ width: '13%' }} />
              <col style={{ width: '7%' }} />
              <col style={{ width: '8%' }} />
              <col style={{ width: '12%' }} />
              <col style={{ width: '27%' }} />
            </colgroup>
            <thead>
              <tr>
                <th>Restaurante</th><th>País</th><th>Ciudad</th><th>Rating</th>
                <th>Cluster</th><th>Dist. centroide</th><th>Recibido</th>
              </tr>
            </thead>
            <tbody>
              {displayRows.map(row => (
                <tr
                  key={row.row_id}
                  onClick={() => handleSelect(row)}
                  className={row.row_id === selectedId ? 'selected' : ''}
                >
                  <td>{row.name}</td>
                  <td>{row.country}</td>
                  <td>{row.city}</td>
                  <td>{row.rating}</td>
                  <td>{row.cluster ?? '—'}</td>
                  <td>{typeof row.dist === 'number'
                        ? row.dist.toFixed(4) : '—'}</td>
                  <td className="muted small">{row.scored_at ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 &&
            <p className="muted">Esperando datos del stream…</p>
          }
        </div>

        <div className="side-panel">
          {selectedId &&
            <RecsPanel rowId={selectedId} name={selectedName} />
          }
          <ClusterChart stats={stats} />
        </div>
      </div>
    </div>
  )
}
