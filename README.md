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
│       ├── planner.py           # Neo4j-backed planner
│       └── llm.py               # Hugging Face LLM client
├── requirements.txt             # Python API dependencies
└── Procfile                     # Render startup command
```
