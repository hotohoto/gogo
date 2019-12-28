import React from 'react'
import GameBoard from './GameBoard'
import InfoBoard from './InfoBoard'
import './App.scss'

const App: React.FC = () => {
  return (
    <div className="App">
      <div className="BoardArea">
        <GameBoard />
      </div>
      <div className="InfoArea">
        <InfoBoard />
      </div>
    </div>
  )
}

export default App
