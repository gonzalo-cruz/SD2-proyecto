import { useState, useEffect } from 'react'

export default function RecsPanel({ rowId, name }) {
  const [recs, setRecs]       = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetch(`/api/recommendations/${rowId}`)
      .then(r => r.json())
      .then(data => { setRecs(data); setLoading(false) })
  }, [rowId])

  return (
    <div className="recs-panel">
      <div className="recs-header">
        <span className="recs-title">Similar a</span>
        <em className="recs-name">{name}</em>
      </div>

      {loading
        ? <p className="muted">Cargando…</p>
        : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Restaurante</th><th>País</th><th>Rating</th><th>Similitud</th>
              </tr>
            </thead>
            <tbody>
              {recs.map((r) => (
                <tr key={r.row_id}>
                  <td>{r.name}</td>
                  <td>{r.country}</td>
                  <td>{r.rating}</td>
                  <td>
                    <div className="sim-bar">
                      <div className="sim-fill" style={{ width: `${r.similarity}%` }} />
                      <span>{r.similarity}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
    </div>
  )
}
