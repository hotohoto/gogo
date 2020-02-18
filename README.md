# GOGO

a small project to mimic alphazero

## goal

- elo 2000 which is 1 kyu

### Terms

- group
  - adjacent stones
- liberty
  - empty place adjacent to a group
- neighbor
- diagonal
- ko
- superko
- eye
  - single liberty
- eyeish
- komi or komidashi
  - points added to the score of the player with the white stones as compensation for playing second

## TODO

(core)

- Add TensorBoardX

(server)

- flask
- placing stone

(client)

- add tslint
- make the board clickable
- add scss lint
- status
  - board size
    - 9, 13, 19
  - history
    - first stone color
    - action
      - move
      - skipping a move
      - surrender
  - current status
    - board
      - stones
      - last stone
    - info
      - whose turn
      - number of dead stones
- actions
  - move
  - skipping current turn
  - surrendering
  - time machine
    - with slide bar

## Setup development environment

```bash
python -m venv venv
. venv/bin/activate
pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## AlphaZero

- modified MCTS backed by ResNet
- which means it chooses moves selected by MCTS
- plays against itself
- no hard-coding except the game rules
- but still there are a lot of hyper parameters
- policy: 19x19 + 1
  - possible stone placements
  - pass move
- each MCTS used 800 simulations

### Questions and answers

- what data does it use for training?
  - real game states
  - actual winner, probs(from n_visits)
- what does it use for choosing action?
  - $\pi$ calculated only by n_visits for self play
- does it simulate with the monte carlo method at the leaf node?
  - no
- when the network is used?
  - For a leaf nodes to decide the value of the node and the probability of its children expanding the node
- how to train the model? what is the dataset like?
  - softmax of n_visits for each children of root
- how to control exploitation vs exploration trade-off?
  - the higher temperature the more exploration

### Comparing to AlphaGoZero

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

### Network architecture

- body
  - 19 residual blocks
    - 256-conv-filter - BN - ReLU - 256-conv-filter - BN - skip connection - ReLU
- policy head
- value head

### Other AlphaZero details

- training
  - 5000 v1 TPUs
  - 64 v2 TPUs
- inference
  - 4 TPUs

### AlphaZero_Gomoku

- 6 by 6 board
- requires 4 stones in a row to win the game
- no ResNet

### ResNet (deep residual neural network)

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

- $z$: actual winner of a single self-play game `-1` or `0` or `1`
- $v$: network prediction for winning -1~1
- $\pi$:


### MCTS (Monte-Carlo Tree Search)


- S_0 is leaf node?
  - yes (if it's leaf node)
    - n_0 == 0 (times sampled)
      - yes (has not been rolled out)
        - roll out and get value
        - backpropagate and update the total scores and number of visits n until reaching to the root
      - no (has been rolled out before)
        - add new available nodes
        - go to the first child node
          - roll out
          - backpropagate and update the total scores and number of visits n until reaching to the root
  - no
    - choose a child node that maximises the score

(how to calculate the score)

- UCB1(S_i)
  - v_i + C * sqrt(log(n_parent) / n_i)
- alpha zero version
  - Q + C * P * sqrt(n_parent) / (1 + n_i)

- Selection
  - start from root R and select successive child nodes until a leaf node L is reached.
  - The root is the current game state and a leaf is any node from which no simulation (playout) has yet been initiated.
  - The section below says more about a way of biasing choice of child nodes that lets the game tree expand towards the most promising moves, which is the essence of Monte Carlo tree search.
- Expansion
  - unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child nodes and choose node C from one of them. Child nodes are any valid moves from the game position defined by L.
- Simulation
  - complete one random playout from node C.
  - This step is sometimes also called playout or rollout.
  - A playout may be as simple as choosing uniform random moves until the game is decided.
- Backpropagation
  - use the result of the playout to update information in the nodes on the path from C to R.

## Discussion

- network doesn't know how many stones for each team were passed away
  - it's difficult to predict if it's gonna win or lose just by seeing the current stones on the board

## References

- [List of Go terms](https://en.wikipedia.org/wiki/List_of_Go_terms)
- [elo](https://en.wikipedia.org/wiki/Go_ranks_and_ratings#Elo_ratings_as_used_in_Go)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- [MCTS in alphazero](https://medium.com/@jonathan_hui/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a)
- [A simple alphazero tutorial](https://web.stanford.edu/~surag/posts/alphazero.html)
- [Lessons From Implementing AlphaZero parts 1~6](https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191)
- [AlphaZero Gomoku (original source code)](https://github.com/junxiaosong/AlphaZero_Gomoku)
- [Monte Carlo Tree Search](https://www.youtube.com/watch?v=UXW2yZndl7U)
- [AlphaGo Zero Explained In One Diagram](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)
- [MCTS on Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
