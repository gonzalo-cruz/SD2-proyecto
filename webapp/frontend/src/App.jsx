import { useState, useEffect } from 'react'
import ExploreTab from './ExploreTab'
import LiveTab from './LiveTab'

export default function App() {
  const [tab, setTab] = useState('explore')
  const [countries, setCountries] = useState([])
  const [prices, setPrices]       = useState([])

  useEffect(() => {
    fetch('/api/countries').then(r => r.json()).then(setCountries)
    fetch('/api/prices').then(r => r.json()).then(setPrices)
  }, [])

  return (
    <div className="app">
      <header className="header">
        <span className="logo">Restaurant<em>Findr</em></span>
        <nav>
          <button
            className={tab === 'explore' ? 'tab active' : 'tab'}
            onClick={() => setTab('explore')}
          >
            Explorar
          </button>
          <button
            className={tab === 'live' ? 'tab active' : 'tab'}
            onClick={() => setTab('live')}
          >
            En vivo
          </button>
        </nav>
      </header>

      <main className="main">
        {tab === 'explore'
          ? <ExploreTab countries={countries} prices={prices} />
          : <LiveTab countries={countries} prices={prices} />
        }
      </main>
    </div>
  )
}
