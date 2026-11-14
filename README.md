# Building Plan Compliance Pre-Check

A small multi-agent service for the ML System Design & AgentOps mini project. You give it the numbers for a building plan (plot size, road width, height, gaps to the plot edge and so on). It finds the bye-law rules that apply, checks the plan against each one, and double-checks its own answer against the rule text. The result is one of three outcomes: `clear-for-submission`, `revise-and-recheck` or `specialist-review`.

It never certifies a plan. "Clear" only means none of the rules we have were broken. A licensed architect or engineer still decides.

The rules come from real documents: Table 8 and the height notes of the Bengaluru Zonal Regulations Amendment 2025, plus extracts of the Karnataka Model Building Bye-Laws 2017. We do not have Table 9, Table 10 or the parking norms, so the service says "not checked" or sends the plan to a person for those.

## Running it

You need Docker Desktop.

```bash
cp .env.example .env            # the defaults use a local stand-in model, so no key is needed
docker compose up --build       # the first build is slow (it downloads torch and the embedding model)
```

On Windows PowerShell, use `copy .env.example .env`.

| Address | What it is |
|---|---|
| http://localhost:8000 | the app: fill in a plan and see which rules it breaks |
| http://localhost:8000/docs | API docs |
| http://localhost:3000 | Grafana (no login): speed, model calls, cost, quality |
| http://localhost:16686 | Jaeger: one check traced step by step (service `api`) |
| http://localhost:9090 | Prometheus |

`docker compose up --scale worker=3` runs more workers. After you pull new code, run `docker compose up --build` again.

## Using a real model

Any OpenAI-compatible provider works. We used Google Gemini (`gemini-3.5-flash-lite`, free tier). Put these in `.env`:

```
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_API_KEY=your-key-here
LLM_MODEL=gemini-3.5-flash-lite
LLM_JSON_MODE=false
LLM_REASONING_EFFORT=low
LLM_RATE_LIMIT_WAIT_S=65
LLM_MIN_INTERVAL_S=4.5
PROMPT_VERSION=v2
MAX_STEPS=14
MAX_TOOL_CALLS=14
MAX_RUN_SECONDS=900
```

Never commit `.env`. It is in `.gitignore`.

`docker compose restart` does not reread `.env`, so apply changes with:

```bash
docker compose up -d --build --force-recreate api worker
docker compose exec worker python -m precheck.cli check-llm    # does the model answer?
docker compose exec worker python -m precheck.cli probe        # runs the main example once and shows each step
```

The free tier allows 15 requests a minute and 500 a day. One check makes about 7 to 10 model calls, which is why `LLM_MIN_INTERVAL_S` spaces the calls out and why the full evaluation is run in parts (see below).

## Trying it

Open http://localhost:8000, or call the API. This is the hard case from the brief: a small plot on a narrow road, where the height limits sit in separate notes that the gap table never mentions.

```bash
curl -s localhost:8000/v1/runs -H 'content-type: application/json' -d '{"plot_area_sqm":200,"road_width_m":5.5,
 "building_height_m":13.5,"num_floors":4,"built_up_area_sqm":400,"ground_coverage_pct":60,"parking_spaces":1,
 "stilt_height_m":0,"plot_depth_m":20,"plot_width_m":10,"front_setback_m":0.8,"rear_setback_m":1,
 "left_setback_m":1,"right_setback_m":1,"building_use":"residential","zone":"residential","notes":""}'
```

That returns `202` and a `run_id`. Poll `GET /v1/runs/{run_id}` for the result. The other endpoints:

| Endpoint | What it does |
|---|---|
| `POST /v1/runs` | submit a plan; bad input gets `422` and is kept in quarantine |
| `GET /v1/runs/{id}` | status and result |
| `GET /v1/runs/{id}/trace` | every tool call, passage, prompt version, latency and cost |
| `POST /v1/runs/{id}/feedback` | `{verdict: satisfactory or unsatisfactory, reason}` |
| `GET /v1/quality/summary` | automatic quality scores for a sample of live runs |
| `/v1/feedback/summary`, `/v1/quarantine`, `/v1/contract`, `/health`, `/metrics` | housekeeping |

## How it works

```
client -> api (validate, rate-limit, cache) -> redis queue -> worker -> result in redis
                                                              |
              supervisor (LangGraph): classify -> research -> assess -> verify -> route
              researcher: decide and call tools, in a loop
              assessor: one finding per rule, with a verbatim quote
              verifier: plain code, checks citations and coverage, can send it back for another try
              if the model is down: retrieval only, goes to a person
```

- **Tools the researcher can use:** `search_corpus`, `get_clause`, `list_table`.
- **Two agents, one hand-off.** The assessor can only cite passages the researcher retrieved, so it cannot invent a rule.
- **The verifier is code, not a model.** Every citation must be a passage that was retrieved, every quote must appear word for word, and the required rules must be covered. If the checks fail twice, the plan goes to a person with no findings.
- **Some rules are decided in code.** The Table 8 row for the plot size, the height limits, the open-parking (stilt) floor rule and the "Table 9 is not in our documents" case are computed in `precheck/agents.py`. The cheap model got these wrong in testing, so the model now only explains them.
- **Guardrails.** Notes that look like prompt injection are thrown away and the plan goes to a person. Notes are wrapped as untrusted text, and the output is filtered so it never says "approved" or "certified".
- **Why async.** A check takes tens of seconds, so the API replies `202` right away and the client polls. Workers scale separately from the API.

## Adding more rules

`corpus/` holds the rule text. To add more (Table 9, Table 10, parking norms), drop text or PDFs in and run:

```bash
docker compose run --rm indexer && docker compose restart api worker
```

PDF splitting uses a regex (`PDF_CLAUSE` in `precheck/index.py`), so check `index/chunks.jsonl` afterwards. A table that extracts badly should be converted by hand to the `[ID] ...` format, one row per chunk, as in `corpus/Bengaluru_zoning_amendment_2025_and_bye-law_extracts.txt`. Scanned PDFs need OCR first.

## Evaluation

There are 35 scenarios with an answer key that is separate from the system (`eval/reference.py`). Nine are invalid inputs that never reach the model. With a real model, run them in parts to stay inside the free tier:

```bash
docker compose exec -T worker python -m eval.run --only s01,s02,s03,s04,s05,s06,s07,s08,s09 --workers 1
docker compose exec -T worker python -m eval.run --only s10,s11,s12,s13,s14,s15,s16,s17,s18 --workers 1
docker compose exec -T worker python -m eval.run --only s19,s20,s21,s22,s23,s24,s25,s26 --workers 1
docker compose exec -T worker python -m eval.run --only s27,s28,s29,s30,s31,s32,s33,s34,s35 --workers 1
```

On Windows, `run-gemini-eval.bat` runs all four parts and writes `results/gemini-part1.txt` to `results/gemini-part4.txt`. The results and what went wrong along the way are in `docs/REPORT.md`. `eval/README.md` explains each metric.

## Tests

```bash
pip install -r requirements.txt pytest fakeredis
python -m pytest -q                 # stand-in model, no Docker or network needed
python tests/devserver.py           # the whole app in one process on :8000, for a quick look
```

## Other tools

```bash
docker compose exec worker python -m precheck.cli stub-mode down   # stand-in model only: break the AI on purpose for the demo
docker compose exec worker python -m precheck.cli index-info       # when the rule index was built
locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 -t 3m --csv results/run   # load test
```

Settings you may want in `.env`: `WORKER_CONCURRENCY` (checks per worker at once), `SCORE_SAMPLE_RATE` (share of live runs that get scored) and `RATE_LIMIT_PER_MIN` (raise it for a load test).

## Docs

- `docs/REPORT.md`: the project report
- `docs/DEMO_SCRIPT.md`: how to explain it in plain words
- `docs/SLO.md`: the speed target and load-test numbers
- `docs/RUNBOOK.md`: what to do when it breaks
- `docs/INCIDENTS.md`: failures we hit
- `ROADMAP.md`: requirement status and what is left
