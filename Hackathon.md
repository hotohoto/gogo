# AlphaZero

- modified MCTS backed by ResNet
- which means it chooses moves selected by MCTS
- plays against itself
- no hard-coding except the game rules
- but still there are a lot of hyper parameters

## Comparing to AlphaGoZero

(AlphaGoZero)

- binary output: win(1) or loss(0)
- keeps best NN for doing self-play against later within iteration
- rotating or flipping for data augmentation
- hyperparameter was tuned by Bayesian Optimization

(AlphaZero)

- 3 outputs: win(1) or draw(0) or loss(-1)
- maintains a single network and it is updated continually
- no rotating or flipping for data augmentation
- hyperparameter were the same for different games

## Other AlphaZero details

- training
  - 5000 v1 TPUs
  - 64 v2 TPUs
- inference
  - 4 TPUs

## AlphaZero_Gomoku

- 6 by 6 board
- requires 4 stones in a row to win the game
- no ResNet

## ResNet (deep residual neural network)

- outputs
  - estimated value $v$ of the position from 1 to -1
  - a vector of probabilities $p$ for playing next possible action
- layers
  - AlphaGoZero
    - inputs
      - 40 residual layer
        - policy head
        - value head

### loss function

$$
(p, v) = f_\theta(s)
$$

$$
l = (z - v)^2 - \pi^{T}\log p + c||\theta||^2
$$


## MCTS (Monte-Carlo Tree Search)




## TODO

- Add TensorBoardX
- Add ResNet
- Replace game logic with openai gym - gomoku
- Replace game logic with openai gym - go

## References

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- [Lessons From Implementing AlphaZero parts 1~6](https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191)
- [Monte Carlo Tree Search](https://www.youtube.com/watch?v=UXW2yZndl7U)
- [AlphaGo Zero Explained In One Diagram](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)
