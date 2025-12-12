```
uv venv
. .venv/bin/activate
```

```
maturin develop --uv --release
```

# start 2 terminals for frontend and worker
```
. .venv/bin/activate
python ./examples/openai_service_pipeline/frontend.py 
```

```
. .venv/bin/activate
python ./examples/openai_service_pipeline/worker.py 
```

## Start benchmark

```
bash ./examples/openai_service_pipeline/curl.sh
. .venv/bin/activate
python ./examples/openai_service_pipeline/load_test_debug.py
```