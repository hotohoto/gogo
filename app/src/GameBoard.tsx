import React, { MouseEvent } from 'react';
import './GameBoard.scss';

const WH_CANVAS = 1920
const LINE_WIDTH = 3
const DOT_RADIUS = LINE_WIDTH * 3
const PADDING = 8
const WH_GRID = WH_CANVAS - PADDING * 2

type GameBoardProps = {
  boardSize?: number;
  whiteStones: [number, number][];
  blackStones: [number, number][];
  lastStone?: [number, number];
}

type GameBoardState = {
  extraStones: [number, number][];
}

enum StoneColor {
  Black,
  White,
}

class GameBoard extends React.Component<GameBoardProps, GameBoardState> {

  gameBoardArea: any;
  gameBoardCanvas: any;
  stoneCanvas: any;

  static defaultProps: GameBoardProps = {
    boardSize: 19,
    whiteStones: [[5,6], [6,6], [5,5]],
    blackStones: [[7,6], [8,7], [7,7]],
    lastStone: [5,6],
  }

  constructor(props: GameBoardProps) {
    super(props)

    this.state = {
      extraStones: [[3,3]]
    }
  }

  getStep = () : number => {
    // @ts-ignore
    const boardSize = this.props.boardSize * 1
    return WH_GRID / boardSize
  }

  toPos = (idx: number) : number => {

    const step = this.getStep()
    const half = step / 2

    return PADDING + half + idx * step
  }

  toIdx = (pos: number) : number => {
    const step = this.getStep()
    const half = step / 2

    return (pos * WH_CANVAS / this.gameBoardArea.scrollWidth  - PADDING - half) / step
  }

  drawDot = (ctx: any, ix: number, iy: number) => {
    ctx.beginPath();
    ctx.arc(this.toPos(ix), this.toPos(iy), DOT_RADIUS, 0, 2 * Math.PI);
    ctx.fill();
  }

  drawStone = (ctx: any, color: StoneColor, ix: number, iy: number) => {
    const originalFillStyle = ctx.fillStyle
    const half = this.getStep() / 2
    const stoneRadius = half - LINE_WIDTH * 1.5

    const x = this.toPos(ix)
    const y = this.toPos(iy)

    ctx.beginPath();
    ctx.fillStyle = "#00000022"
    ctx.arc(x + LINE_WIDTH * 2.5, y + LINE_WIDTH * 2.5, stoneRadius, 0, 2 * Math.PI)
    ctx.fill()

    if (color === StoneColor.Black) {
      ctx.beginPath();
      ctx.fillStyle = "#000000"
      ctx.arc(x, y, stoneRadius, 0, 2 * Math.PI)
      ctx.fill()
      ctx.beginPath()
      ctx.fillStyle = "#292929"
      ctx.arc(x - stoneRadius/4, y - stoneRadius/4, stoneRadius/1.8, 0, 2 * Math.PI)
      ctx.fill()
    } else if (color === StoneColor.White) {
      ctx.beginPath()
      ctx.fillStyle = "#F8F8F8"
      ctx.arc(x, y, stoneRadius, 0, 2 * Math.PI)
      ctx.fill();
      ctx.beginPath()
      ctx.fillStyle = "#FFFFFF"
      ctx.arc(x - stoneRadius/3.3, y - stoneRadius/3.3, stoneRadius/2.5, 0, 2 * Math.PI)
      ctx.fill()
    } else {
      console.error("Unknown Color")
      return
    }

    ctx.fillStyle = originalFillStyle
  }

  drawLastStoneMarker = (ctx: any, ix: number, iy: number) => {
    const step = this.getStep()
    const markerSize = step / 4
    const x = this.toPos(ix)
    const y = this.toPos(iy)

    ctx.beginPath()
    ctx.strokeStyle = "#F6B73C"
    ctx.rect(x - markerSize, y - markerSize, 2 * markerSize, 2 * markerSize)
    ctx.stroke()
  }

  drawStones = (ctx: any) => {
    ctx.clearRect(0, 0, this.stoneCanvas.width, this.stoneCanvas.height);
    ctx.lineWidth = LINE_WIDTH

    for (let s of this.props.whiteStones) {
      this.drawStone(ctx, StoneColor.White, s[0], s[1])
    }
    for (let s of this.props.blackStones) {
      this.drawStone(ctx, StoneColor.Black, s[0], s[1])
    }
    for (let s of this.state.extraStones) {
      this.drawStone(ctx, StoneColor.Black, s[0], s[1])
    }
    if (this.props.lastStone) {
      this.drawLastStoneMarker(ctx, this.props.lastStone[0], this.props.lastStone[1])
    }
  }

  componentDidMount() {
    if (this.gameBoardCanvas) {
      const ctx = this.gameBoardCanvas.getContext("2d")
      ctx.imageSmoothingEnabled = true;
      ctx.lineWidth = LINE_WIDTH
      // @ts-ignore
      const boardSize = this.props.boardSize * 1

      // Draw grid
      for (let i=0; i <= boardSize; i++) {
        ctx.beginPath();
        ctx.moveTo(this.toPos(i), this.toPos(0));
        ctx.lineTo(this.toPos(i), this.toPos(boardSize - 1));
        ctx.stroke();
      }

      for (let i=0; i <= boardSize; i++) {
        ctx.beginPath();
        ctx.moveTo(this.toPos(0), this.toPos(i));
        ctx.lineTo(this.toPos(boardSize - 1), this.toPos(i));
        ctx.stroke();
      }

      // Draw dots
      if (boardSize === 19) {
        const positions = [3, 9, 15]
        for (let i = 0; i < positions.length; i++) {
          for (let j = 0; j < positions.length; j++) {
            this.drawDot(ctx, positions[i], positions[j])
          }
        }
      } else if (boardSize === 13) {
        const positions = [[3, 3], [3, 9], [9, 3], [9, 9], [6, 6]]
        for (let i = 0; i < positions.length; i++) {
          this.drawDot(ctx, positions[i][0], positions[i][1])
        }
      } else if (boardSize === 9) {
        const positions = [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]]
        for (let i = 0; i < positions.length; i++) {
          this.drawDot(ctx, positions[i][0], positions[i][1])
        }
      }
    }

    if (this.stoneCanvas) {
      const ctx = this.stoneCanvas.getContext("2d")
      this.drawStones(ctx)
    }
  }

  onMouseClick = (e: MouseEvent) => {
    const ix = Math.round(this.toIdx(e.clientX - this.gameBoardArea.offsetLeft))
    const iy = Math.round(this.toIdx(e.clientY - this.gameBoardArea.offsetTop))
    const newExtraStones: [number, number][] = [...this.state.extraStones, [ix, iy]]
    this.setState({extraStones: newExtraStones})
    // console.log(this.state.extraStones)
  }

  render() {

    if (this.stoneCanvas) {
      const ctx = this.stoneCanvas.getContext("2d")
      this.drawStones(ctx)
    }

    return (
      <div
        className="GameBoard"
        ref={ref => this.gameBoardArea = ref}
        onClick={this.onMouseClick}
      >
        <canvas
          className="GameBoardCanvas"
          ref={ref => this.gameBoardCanvas = ref}
          width={WH_CANVAS}
          height={WH_CANVAS}
        />
        <canvas
          className="StoneCanvas"
          ref={ref => this.stoneCanvas = ref}
          width={WH_CANVAS}
          height={WH_CANVAS}
        />
      </div>
    )
  }

}

export default GameBoard;
