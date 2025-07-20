# PruneMe

## Pruning Methods

Currently supports 3 distinct pruning methods:

- **`normal`**: Prune the last `k` layers
- **`similarity-based`**: Prune `k` consecutive layers chosen by similarity-based algorithm
- **`prune-one`**: Prune only one layer specified in config

## Getting Started

### Prerequisites

- Python with `uv` package manager

### Setup

1. **Initialize mergekit submodule:**
   ```bash
   git submodule update --init --recursive
   ```

2. **Calculate layer importance scores:**
   
   Compute layer importance scores or distances between layer pairs for similarity-based pruning:
   ```bash
   uv run python main.py
   ```

3. **Prune the model:**
   
   Apply the specified pruning method:
   ```bash
   uv run python prune.py
   ```

4. **Evaluate the pruned model:**
   
   Assess performance and save metrics (perplexity, loss, etc.) to metadata files:
   ```bash
   uv run python evaluate.py
   ```

## Testing

**Note:** Ensure the pruned model is saved in `merged/Vanilla-3.2-8L/prune-one/6` before running tests. (by running `scripts/prepare.sh`)

```bash
uv run pytest tests --disable-warnings
```

## Prepare Imagenet
- 
```bash
wget http://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
mkdir -p ./data/imagenet-1k/val
tar -xf ILSVRC2012_img_val.tar -C ./data/imagenet-1k/val
rm ILSVRC2012_img_val.tar
```

- Run `scripts/valprep.sh` to map file names to class index
```bash
bash scripts/valprep.sh
```


