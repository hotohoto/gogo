import React from 'react'
import './InfoBoard.scss'

enum StoneColor {
  Black,
  White,
}

type InfoBoardProps = {
  deadBlackStones: number;
  deadWhiteStones: number;
  currentTurn: StoneColor;
}

class InfoBoard extends React.Component<InfoBoardProps> {

  static defaultProps: InfoBoardProps = {
    deadBlackStones: 0,
    deadWhiteStones: 0,
    currentTurn: StoneColor.Black,
  }

  onClickSkip = () => {}
  onClickYield = () => {}

  render() {
    return (
      <div className="InfoBoard">
        <div>
          {`⚫: ${this.props.deadBlackStones}`}
          {this.props.currentTurn === StoneColor.Black &&
            (
              <span role="img" aria-labelledby="Current turn is black."> 👈</span>
            )
          }
        </div>
        <div>
          {`⚪: ${this.props.deadWhiteStones}`}
          {this.props.currentTurn === StoneColor.White &&
            (
              <span role="img" aria-labelledby="Current turn is white."> 👈</span>
            )
          }
        </div>
        <div>
          <button onClick={this.onClickSkip}>
            <span role="img" aria-labelledby="Skip">↪️</span> Skip
          </button>
        </div>
        <div>
          <button onClick={this.onClickYield}>
          <span role="img" aria-labelledby="Yield">🏳️</span> Yield
          </button>
        </div>
      </div>
    )
  }
}

export default InfoBoard;
