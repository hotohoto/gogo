# GOGO

## goal

- elo 2000 which 1 kyu

## TODO

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

(server)

- flask
- placing stone

(core)

- common
  - score counting
  - placing stone
    - add the stone
    - remove surrounded stones
- check out AI gym
  - API, ...

## setup

```bash
python -m venv venv
. venv/bin/activate
pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## References

- https://en.wikipedia.org/wiki/List_of_Go_terms
- https://en.wikipedia.org/wiki/Go_ranks_and_ratings#Elo_ratings_as_used_in_Go