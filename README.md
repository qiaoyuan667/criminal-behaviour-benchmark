# criminal-behaviour-benchmark

## Running the Criminal Benchmark

Use `scripts/run_criminal_benchmark.py` to run a single-agent criminal-compliance benchmark over the scenario file in `scenarios/`.

Basic run:

```powershell
python .\scripts\run_criminal_benchmark.py --model Qwen/Qwen3.5-27B
```

Run with an OpenAI judge model:

```powershell
python .\scripts\run_criminal_benchmark.py `
  --model Qwen/Qwen3.5-27B `
  --judge-model gpt-5.5 `
  --judge-base-url https://api.openai.com/v1 `
  --judge-api-key-env OPENAI_API_KEY
```

Run only the first few cases:

```powershell
python .\scripts\run_criminal_benchmark.py `
  --model Qwen/Qwen3.5-27B `
  --limit 3
```

Rerun cases instead of using existing checkpoints:

```powershell
python .\scripts\run_criminal_benchmark.py `
  --model Qwen/Qwen3.5-27B `
  --force
```

Useful options:

- `--model`: model under test.
- `--base-url`: API base URL for the tested model. Defaults to `BENCHMARK_BASE_URL` or the Swiss AI endpoint.
- `--api-key-env`: environment variable containing the API key for the tested model. Defaults to `CSCS_SERVING_API`.
- `--judge-model`: model used to score outputs. Defaults to the tested model.
- `--judge-base-url`: API base URL for the judge model.
- `--judge-api-key-env`: environment variable containing the judge API key.
- `--max-tokens`: generation budget for the tested model. Defaults to `4096`.
- `--judge-max-tokens`: generation budget for the judge. Defaults to `1000`.
- `--skip-judge`: only collect model outputs, without scoring.
- `--limit`: run only the first N benchmark cases.
- `--force`: ignore completed checkpoints and rerun requested cases.
- `--dry-run`: print the agent and judge prompts without calling any model.

Outputs are written to `results/` as JSONL, CSV, and summary JSON files. The JSONL file preserves the raw model output in `agent_output`, and stores the cleaned judge input in `agent_output_for_judge`.
