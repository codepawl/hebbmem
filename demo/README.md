# hebbmem Visualization Demo

Interactive visualization of hebbmem's three mechanisms: decay, Hebbian learning, spreading activation.

## Run

```bash
# Install demo dependencies (not part of hebbmem)
uv pip install fastapi uvicorn

# Start server
uv run python demo/server.py

# Open http://localhost:8765
```

## What to try

1. Click "Load Sample" to load pre-built memories
2. Type "Project Atlas" and click Recall — watch activation spread through the graph
3. Click Step x5 — watch memories dim (decay)
4. Recall "Atlas" again — notice Hebbian bonds are stronger
5. Store a new memory and watch it auto-connect
6. Step x10 — trivial memories nearly disappear

## Recording for content

For best video: resize browser to 1280x720, use Load Sample, then execute this sequence:
1. Recall "Project Atlas" (shows spreading activation)
2. Step x5 (shows decay)
3. Recall "Rust programming" (shows separate cluster activating)
4. Recall "Atlas deadline" (shows Hebbian bonds from previous recalls)
5. Step x10 (shows trivial memories fading away)
