import { useState, useEffect, useRef } from 'react'

const POLL_INTERVAL_MS = 3000

export default function RecsPanel({ rowId, name }) {
  const [recs, setRecs]         = useState([])
  const [loading, setLoading]   = useState(true)
  const [pending, setPending]   = useState(false)
  const pollRef                 = useRef(null)

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  const fetchRecs = async (id) => {
    const r    = await fetch(`/api/recommendations/${id}`)
    const data = await r.json()

    // Array → scored, render immediately
    if (Array.isArray(data)) {
      setRecs(data)
      setLoading(false)
      setPending(false)
      stopPolling()
      return true   // done
    }

    // {status: 'pending'} → not yet scored
    if (data?.status === 'pending') {
      setPending(true)
      setLoading(false)
      return false  // keep polling
    }

    // {status: 'already_scored'} or unexpected → treat as done with empty recs
    setRecs([])
    setLoading(false)
    setPending(false)
    stopPolling()
    return true
  }

  useEffect(() => {
    stopPolling()
    setRecs([])
    setLoading(true)
    setPending(false)

    let cancelled = false

    const start = async () => {
      // First fetch
      const done = await fetchRecs(rowId)
      if (cancelled || done) return

      // If pending, enqueue in priority topic then start polling
      await fetch(`/api/priority/${rowId}`, { method: 'POST' })

      pollRef.current = setInterval(async () => {
        if (cancelled) { stopPolling(); return }
        await fetchRecs(rowId)
      }, POLL_INTERVAL_MS)
    }

    start()

    return () => {
      cancelled = true
      stopPolling()
    }
  }, [rowId])

  return (
    <div className="recs-panel">
      <div className="recs-header">
        <span className="recs-title">Similar a</span>
        <em className="recs-name">{name}</em>
      </div>

      {loading && <p className="muted">Cargando…</p>}

      {pending && !loading && (
        <div className="pending-state">
          <div className="pending-spinner" />
          <p className="muted pending-msg">
            Este restaurante aún no ha sido clasificado.<br />
            Procesando en cola de prioridad…
          </p>
        </div>
      )}

      {!loading && !pending && (
        recs.length > 0
          ? (
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
          : <p className="muted">No se encontraron recomendaciones.</p>
      )}
    </div>
  )
}
