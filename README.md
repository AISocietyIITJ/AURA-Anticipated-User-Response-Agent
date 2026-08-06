# Customer-Care AI Co-pilot (Cloud Deployment Version)

Real-time co-pilot for human customer-care agents on live calls.

## Hosted Application
- **Live Demo:** [https://aura-anticipated-user-response-agent.onrender.com]
- **Original Repository:** [https://github.com/aquaphoenix69-prog/copilot]

Pipeline matches the project diagram:

```
Live call → NVIDIA NIM Parakeet STT → Transcript
                                      ├─→ HF Serverless Interpreter (intent + customer tag)
                                      └─→ HF Serverless Sentiment
                                                      ↓
                                                   State
                                                      ↓
                                          Neo4j AuraDB Planner Agent
                                                      ↓
                                      HF Serverless LLM → Coaching reply
                                                      ↑
                                        (fallback path: skip planner if cold state)
```

## Stack

- **STT**: NVIDIA NIM Cloud API (`nvidia/parakeet-tdt-0.6b-v2`)
- **Interpreter API**: Hugging Face Serverless Inference API (Instruct LLM)
- **Sentiment API**: Hugging Face Serverless Inference API (Instruct LLM)
- **Graph DB**: Neo4j AuraDB (Cloud)
- **LLM**: Hugging Face Serverless Inference API (`meta-llama/Llama-3.1-8B-Instruct`)
- **API**: FastAPI (REST + WebSocket) deployed on Render

## Why APIs?
Originally, this project utilized local inference via vLLM and local ML models. For this cloud deployment on Render, we shifted to cloud APIs (Hugging Face Serverless Inference and NVIDIA NIM API) to remove heavy GPU requirements and accommodate scaling in a serverless/PaaS environment.

## Why no 400 errors anymore

The dataset is loaded **once** and projected into Neo4j AuraDB. At runtime the planner asks Cypher for the top-k agent_tags conditioned on the current `(intent, customer_tag, sent_bucket)` state — typically a handful of nodes, kilobytes of context, well under any LLM input limit.

## Graph model

```
(:State {key, intent, customer_tag, sent_bucket})
   -[:HAS_INTENT]-> (:Intent)
   -[:HAS_CUSTOMER_TAG]-> (:CustomerTag)
   -[:USED_AGENT_TAG {count}]-> (:AgentTag)
   -[:LED_TO_OUTCOME {count}]-> (:Outcome)
   -[:NEXT {count}]-> (:State)
```

Sentiment buckets: `neg < -0.33`, `pos > 0.33`, else `neu`.

## Fallback mechanism

If the planner finds no graph match for the current state (cold node), it backs off:
`(intent, customer_tag, sentiment)` → `(intent, customer_tag)` → `(intent)` → no-graph fallback. When the LLM call fails or planner returns nothing, the API still returns a degraded reply so the human agent never gets a blank screen.

## Planner confidence

The Neo4j graph is the **primary** planner. Every graph match is scored
(modular, in `app/agents/confidence.py`) using three signals:

- **Graph score** — dominance of the top agent: `top_count / Σ(top-k counts)` (the
  original `confidence` formula, preserved for backward compatibility).
- **Occurrence frequency** — total historical edge counts, normalized
  (`evidence_volume`, capped at `EVIDENCE_NORM`).
- **Match specificity** — how specific the backoff level was
  (`MatchLevel`: exact = 1.0, intent+tag = 0.9, intent-only = 0.75).

The effective score is:

```
planner_confidence = level × (0.5 × graph_share + 0.5 × evidence_volume)
```

`PlannerSuggestion.confidence` keeps the original graph score; the new
`planner_confidence` field carries the effective decisioning confidence used
for the fallback threshold. When there is no graph data, `planner_confidence`
is `0.0`.

## Fallback threshold

`PLANNER_CONFIDENCE_THRESHOLD` (default `0.60`, see Configuration below) decides
whether the graph recommendation is trusted:

```
IF planner_confidence >= threshold   -> return graph recommendation
ELSE                                  -> invoke LLM planner fallback
```

## LLM fallback flow

```
User query
   ↓
Interpreter Agent
   ↓
Planner Agent
   ↓
Query Neo4j → compute confidence
   ↓
IF confidence >= threshold: return graph recommendation
ELSE:
   Invoke LLM planner (app/agents/llm_fallback.py)
     inputs: user query, intent, customer tag, sentiment,
             previous graph candidates, available agent names
     output: JSON {"agents": ["history", "persona", "planner", "critic"]}
   ↓
Validate every agent against the registered-agent registry
   (Neo4j AgentTag nodes, cached — app/agents/planner_utils.py)
   ↓
Return generated plan  (used_fallback=True, used_llm_fallback=True)
```

The LLM uses the **same** OpenAI-compatible endpoint as the rest of the app
(`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`) — no new provider. On any failure
(timeout, provider error, invalid JSON, parse error, or hallucinated agents
that fail validation) the planner returns the existing empty fallback and
**never crashes** (the whole `plan()` call is also wrapped so a Neo4j outage
degrades gracefully instead of 500-ing).

Example structured logs:

```
Planner confidence: 0.83
Planner confidence 0.42 below threshold 0.60. Invoking LLM fallback...
LLM generated plan: history → persona → planner → critic
```

## Configuration

Planner settings (env-driven via `app/core/config.py`, all optional):

| Env var                        | Default  | Purpose                                  |
|--------------------------------|----------|------------------------------------------|
| `PLANNER_CONFIDENCE_THRESHOLD` | `0.60`   | Graph confidence needed to skip LLM      |
| `PLANNER_LLM_MODEL`            | (unset)  | Override model for the planner LLM call  |
| `PLANNER_LLM_TEMPERATURE`      | `0.0`    | Temperature for the planner LLM call     |
| `PLANNER_LLM_MAX_TOKENS`       | `256`    | Max tokens for the planner LLM call      |
| `PLANNER_LLM_TIMEOUT`          | `60.0`   | Timeout (seconds) for the planner LLM call |

Confidence weights (`TOP_SHARE_WEIGHT`, `EVIDENCE_WEIGHT`, `EVIDENCE_NORM`) and
match-level weights are module constants in `app/agents/confidence.py` so they
can be tuned in one place.

## Project layout

```
copilot/
├── app/
│   ├── main.py                  # FastAPI app
│   ├── static/                  # Frontend UI
│   │   └── index.html           
│   ├── core/
│   │   ├── config.py            # Settings (env-driven)
│   │   └── schemas.py           # Pydantic models
│   ├── graph/driver.py          # Neo4j AuraDB driver singleton
│   └── agents/                  # Cloud API integrations
│       ├── stt.py               # NVIDIA NIM Parakeet
│       ├── interpreter.py       # Hugging Face LLM-based intent + customer tag
│       ├── sentiment.py         # Hugging Face LLM-based sentiment
│       ├── state.py             # State aggregator
│       ├── planner.py           # Neo4j-backed planner (+ LLM fallback orchestration)
│       ├── confidence.py        # Planner confidence scoring (modular)
│       ├── llm_fallback.py      # LLM-generated next-agent plan (low confidence only)
│       ├── planner_utils.py     # Registered-agent registry + validation
│       └── llm.py               # Hugging Face LLM client
├── requirements.txt             # Python API dependencies
├── requirements-dev.txt         # Dev / test dependencies (pytest)
├── Procfile                     # Render startup command
├── tests/test_planner.py        # Planner + LLM fallback tests
└── conftest.py                  # Test env bootstrap
```
