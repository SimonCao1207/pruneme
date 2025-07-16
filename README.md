# pruneme

## Prunning Methods
- Currently support 3 distinct pruning methods:
    - `normal` : prune last `k` layers
    - `similarity-based`: prune `k` consecutive layers chosen by similarity-based algorithm
    - `prune-one` : prune only one layer specify in config

## Running
- Calculate `layer importance` score of each layer or compute `distances` between layers pairs `similarity-based`prunning

```bash
uv run python main.py
```

- Prune the model with specified method
```bash
uv run python prune.py
```

- Assess the pruned model's performance and save metrics (perplexity, loss, etc.) to metadata files:
```bash
uv run python evaluate.py
```

## Run tests
- Note: make sure prunned model is saved in `merged/Vanilla-3.2-8L/prune-one/6`
```bash
uv run pytest tests --disable-warnings
```

