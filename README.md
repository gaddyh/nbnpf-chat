# NbN Patients & Families Chat

A grounded medication information chatbot for the **NbN (Neuroscience-based Nomenclature) Patients & Families** drug corpus. Patients ask natural-language questions about their medications; the system answers using **only** curated NbN P&F source material — designed to minimize unsupported answers and outside-knowledge leakage.

> The Neuroscience-based Nomenclature (NbN) is an initiative aimed at updating the classification of psychiatric medication.

## Why this exists

LLMs are good at sounding confident, which is dangerous in a medical context. This project constrains the LLM to a narrow role: **interpret and rephrase only the supplied curated evidence.** The architecture makes grounding visible — every answer links back to the exact source field and drug ID it was drawn from.

## How it works

```
User question
    ↓
QueryParser          LLM extracts (drug_name, intent) via Structured Outputs
    ↓                 drug_name falls back to current_drug deterministically
DrugRepository       Maps intent → evidence field(s), returns an EvidenceBundle
    ↓                 100% deterministic lookup, no LLM in this step
AnswerGenerator      LLM rephrases the evidence into a patient-friendly answer
    ↓                 hard rule: use ONLY the provided evidence
Streamlit UI         Chat + sidebar showing drug / intent / source / drug ID
                      + expandable source text under each answer
```

The LLM touches two points only: **parsing** the question and **rephrasing** the answer. Everything in between is deterministic code.

### Evidence bundles

Some questions can't be answered from a single field. For example, *"Why would it help with OCD?"* needs both the indication (OCD is a treated condition) and the mechanism (serotonin reuptake). Each intent maps to one or more evidence fields:

| Intent | Fields retrieved | Rationale |
|---|---|---|
| `approved_use` | `approved_uses` | |
| `efficacy` | `additional_efficacy` | |
| `side_effects` | `side_effects` | |
| `addiction` | `addiction` + `timeline.abrupt_discontinuation` | "not addictive" + "but don't stop abruptly" |
| `science` | `science` + `approved_uses` + `additional_efficacy` | Connect mechanism to indication |
| `pharmacology` | `pharmacology` | |
| `how_it_works` | `how_it_works` | |
| `timeline_onset` | `timeline.onset` | |
| `timeline_maintenance` | `timeline.maintenance` | |
| `discontinuation` | `timeline.abrupt_discontinuation` | |

This is a static mapping — no RAG, no agent planning, no vector search.

## Project structure

```
nbnpf-chat/
├── app.py                      Streamlit UI + session state + grounding sidebar
├── conftest.py                 Pytest fixtures, LLM marker, API-key skip logic
├── data/
│   └── nbnpf_drugs.json        50 drugs, curated NbN P&F content
├── src/
│   ├── models.py               Intent enum, Query, Evidence, EvidenceBundle, ParsedQuery
│   ├── repository.py           DrugRepository — deterministic lookup + evidence bundles
│   ├── query_parser.py         QueryParser — OpenAI Structured Outputs (Pydantic)
│   └── answer_generator.py     AnswerGenerator — grounded rephrase with source footer
├── tests/
│   └── test_scenarios.py       18 tests: repository, parser, generator (2 demo scenarios)
├── requirements.txt
└── .env                        OPENAI_API_KEY (not committed)
```

## Setup

```bash
# Clone
git clone https://github.com/gaddyh/nbnpf-chat.git
cd nbnpf-chat

# Create venv and install deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run
.venv/bin/streamlit run app.py
```

The app loads at `http://localhost:8501`.

## Usage

Type questions in natural language. The sidebar shows what the system detected; the expander under each answer shows the raw source text.

**Scenario 1 — starting a new medication:**

| You ask | System detects | Answer grounds on |
|---|---|---|
| "I was prescribed Prozac. What is it for?" | Prozac → Fluoxetine / approved_use | `approved_uses` |
| "Why would it help with OCD?" | Fluoxetine (context) / science | `science` + `approved_uses` + `additional_efficacy` |
| "How long does it take to work?" | Fluoxetine (context) / timeline_onset | `timeline.onset` |
| "Is it addictive?" | Fluoxetine (context) / addiction | `addiction` + `timeline.abrupt_discontinuation` |

**Scenario 2 — side effects and stopping:**

| You ask | System detects | Answer grounds on |
|---|---|---|
| "I started taking Cymbalta and I feel very sleepy. Could it be related?" | Cymbalta → Duloxetine / side_effects | `side_effects` |
| "Does it also affect sexual function?" | Duloxetine (context) / side_effects | `side_effects` |
| "What is Cymbalta actually doing in the brain?" | Cymbalta / science | `science` + `approved_uses` + `additional_efficacy` |
| "Can I stop it if the side effects bother me?" | Duloxetine (context) / discontinuation | `timeline.abrupt_discontinuation` |

When evidence is missing (e.g. Duloxetine has no discontinuation data), the system says so honestly rather than inventing guidance.

## Tests

```bash
# Deterministic only (no API key needed, runs in <1s)
.venv/bin/pytest tests/ -m "not llm" -v

# Full suite (makes real OpenAI calls)
set -a && . ./.env && set +a && .venv/bin/pytest tests/ -v

# Single scenario
.venv/bin/pytest tests/ -k scenario1 -v
```

18 tests across three layers:

| Layer | Count | API? | What it checks |
|---|---|---|---|
| Repository | 11 | No | Drug resolution, evidence bundle structure, companion fields, data gaps |
| Parser | 3 | Yes | Intent extraction per scenario, deterministic context fallback |
| Generator | 4 | Yes | Key wording, source footer, no-drug message, no-invention on missing data |

LLM tests are auto-skipped when `OPENAI_API_KEY` is not set.

## Data

`data/nbnpf_drugs.json` contains 50 usable P&F medication records. Each record has:

`id`, `name`, `brand_names`, `pharmacology`, `how_it_works`, `approved_uses`, `additional_efficacy`, `side_effects`, `addiction`, `timeline` (onset, maintenance, abrupt_discontinuation, important_aids), `why_take_it`, `science`, `_source`, `_quality_flags`

## Design principles

1. **LLM rephrases, never invents.** The generator's system prompt forbids outside knowledge. If evidence is missing, it says so.
2. **Context fallback is deterministic.** The parser returns `drug_name=null` for "it"/"this medication"; Python code fills in `current_drug`. The LLM never holds conversation context.
3. **Grounding is visible.** Sidebar + source expanders make it immediately clear which field and drug ID each answer came from — important for clinical trust.
4. **Companion evidence prevents forced inference.** Bundling `science` + `approved_uses` for "why would it help with X?" means the model connects mechanism to indication using two grounded texts, not by inventing the link.
5. **Static mapping over RAG.** With 50 drugs and 10 intents, a deterministic field map is simpler, faster, and more auditable than vector search. The right architecture for this corpus size.

## Current scope

This is a focused demo, not a clinical decision-support system.

It can:
- answer questions about medication information already present in NbN P&F
- maintain simple drug context across follow-up questions
- show the source evidence behind each answer

It does not:
- recommend which medication a patient should take
- personalize treatment or dosing
- use external medical knowledge
- replace a clinician

## Tech stack

- **Streamlit** — UI, session state, chat interface
- **OpenAI** (`gpt-4o-mini`) — Structured Outputs for parsing, chat completions for generation
- **Pydantic** — schema enforcement for parser output
- **pytest** — scenario tests with LLM marker and auto-skip
