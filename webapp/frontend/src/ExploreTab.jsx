import { useState, useEffect } from 'react'
import RecsPanel from './RecsPanel'

export default function ExploreTab({ countries, prices }) {
  const [rows, setRows]         = useState([])
  const [loading, setLoading]   = useState(false)
  const [cities, setCities]     = useState([])
  const [cityInput, setCityInput] = useState('')
  const [selectedId, setSelectedId] = useState(null)
  const [selectedName, setSelectedName] = useState('')
  const [filters, setFilters]   = useState({
    country: 'Todos', city: 'Todos', search: '',
    veg: false, vegan: false, gf: false,
    min_rating: '0', price: 'Todos',
  })

  // Fetch cities when country changes
  useEffect(() => {
    setCityInput('')
    if (filters.country === 'Todos') {
      setCities([])
      setFilters(f => ({ ...f, city: 'Todos' }))
      return
    }
    fetch(`/api/cities?country=${encodeURIComponent(filters.country)}`)
      .then(r => r.json())
      .then(setCities)
    setFilters(f => ({ ...f, city: 'Todos' }))
  }, [filters.country])

  const handleCityInput = (value) => {
    setCityInput(value)
    if (value === '') setFilter('city', 'Todos')
    else if (cities.includes(value)) setFilter('city', value)
  }

  const search = () => {
    setLoading(true)
    const p = new URLSearchParams({ limit: 100, ...filters })
    fetch(`/api/restaurants?${p}`)
      .then(r => r.json())
      .then(data => { setRows(data); setLoading(false) })
  }

  useEffect(() => { search() }, [])

  const setFilter = (key, value) => setFilters(f => ({ ...f, [key]: value }))

  const handleSelect = (row) => {
    setSelectedId(row.row_id)
    setSelectedName(row.name)
  }

  return (
    <div className="explore">
      <div className="filters">
        <input
          className="input"
          placeholder="🔍  Buscar por nombre…"
          value={filters.search}
          onChange={e => setFilter('search', e.target.value)}
          onKeyDown={e => e.key === 'Enter' && search()}
        />
        <select className="select" value={filters.country}
          onChange={e => setFilter('country', e.target.value)}
        >
          <option value="Todos">Todos los países</option>
          {countries.map(c => <option key={c}>{c}</option>)}
        </select>
        <input
          className="select"
          list="cities-explore"
          placeholder="Ciudad…"
          value={cityInput}
          disabled={filters.country === 'Todos'}
          onChange={e => handleCityInput(e.target.value)}
        />
        <datalist id="cities-explore">
          {cities.map(c => <option key={c} value={c} />)}
        </datalist>
        <select className="select" value={filters.price}
          onChange={e => setFilter('price', e.target.value)}
        >
          <option value="Todos">Cualquier precio</option>
          {(prices || []).map(p => <option key={p}>{p}</option>)}
        </select>
        <select className="select" value={filters.min_rating}
          onChange={e => setFilter('min_rating', e.target.value)}
        >
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
        <button className="btn-primary" onClick={search}>Buscar</button>
      </div>

      <div className="content-split">
        <div className="table-wrapper">
          {loading
            ? <p className="muted">Cargando…</p>
            : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Restaurante</th><th>País</th><th>Ciudad</th>
                    <th>Rating</th><th>Precio</th><th>Cluster</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(row => (
                    <tr
                      key={row.row_id}
                      onClick={() => handleSelect(row)}
                      className={row.row_id === selectedId ? 'selected' : ''}
                    >
                      <td>{row.name}</td>
                      <td>{row.country}</td>
                      <td>{row.city}</td>
                      <td>{row.rating}</td>
                      <td>{row.price}</td>
                      <td>{row.cluster}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )
          }
          {!loading && rows.length === 0 &&
            <p className="muted">Sin resultados.</p>
          }
        </div>

        {selectedId &&
          <RecsPanel rowId={selectedId} name={selectedName} />
        }
      </div>
    </div>
  )
}
