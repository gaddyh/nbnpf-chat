"""Demo scenario tests for the NbN P&F chatbot.

Two scenarios, each exercising a different conversation arc:
  Scenario 1 (Prozac/Fluoxetine): identity -> indication -> mechanism
      -> timeline -> addiction
  Scenario 2 (Cymbalta/Duloxetine): symptom -> side effect -> follow-up
      -> mechanism -> discontinuation

Three test layers:
  - repository_* : deterministic, no API key required (always run)
  - parser_*     : real OpenAI call, asserts (drug_name, intent)
  - generator_*  : real OpenAI call, asserts key wording in the answer
"""
from __future__ import annotations

import pytest

from src.models import Evidence


# ---------------------------------------------------------------------------
# Scenario 1 — Prozac / Fluoxetine
#   "I was prescribed Prozac. What is it for?"        -> approved_use
#   "Why would it help with OCD?"                     -> science
#   "How long does it take to work?"                  -> timeline_onset
#   "Is it addictive?"                                -> addiction
# ---------------------------------------------------------------------------

SCENARIO_1 = [
    {
        "question": "I was prescribed Prozac. What is it for?",
        "drug_name": "Prozac",
        "resolved": "Fluoxetine",
        "intent": "approved_use",
        "must_contain": ["Fluoxetine"],
    },
    {
        "question": "Why would it help with OCD?",
        "drug_name": None,  # context fallback to Fluoxetine
        "resolved": "Fluoxetine",
        "intent": "science",
        "must_contain": ["serotonin"],
    },
    {
        "question": "How long does it take to work?",
        "drug_name": None,
        "resolved": "Fluoxetine",
        "intent": "timeline_onset",
        "must_contain": ["week"],
    },
    {
        "question": "Is it addictive?",
        "drug_name": None,
        "resolved": "Fluoxetine",
        "intent": "addiction",
        "must_contain": ["not", "addictive"],
    },
]


# ---------------------------------------------------------------------------
# Scenario 2 — Cymbalta / Duloxetine
#   "I started taking Cymbalta and I feel very sleepy. Could it be related?"
#       -> side_effects
#   "Does it also affect sexual function?"            -> side_effects
#   "What is Cymbalta actually doing in the brain?"   -> science
#   "Can I stop it if the side effects bother me?"    -> discontinuation
#       (NOTE: Duloxetine has no abrupt_discontinuation data, so the
#        generator must say it doesn't have that info — not invent it.)
# ---------------------------------------------------------------------------

SCENARIO_2 = [
    {
        "question": "I started taking Cymbalta and I feel very sleepy. Could it be related?",
        "drug_name": "Cymbalta",
        "resolved": "Duloxetine",
        "intent": "side_effects",
        "must_contain": ["sleep"],
    },
    {
        "question": "Does it also affect sexual function?",
        "drug_name": None,
        "resolved": "Duloxetine",
        "intent": "side_effects",
        "must_contain": ["sexual"],
    },
    {
        "question": "What is Cymbalta actually doing in the brain?",
        "drug_name": "Cymbalta",
        "resolved": "Duloxetine",
        "intent": "science",
        "must_contain": ["serotonin"],
    },
    {
        "question": "Can I stop it if the side effects bother me?",
        "drug_name": None,
        "resolved": "Duloxetine",
        "intent": "discontinuation",
        # Duloxetine has no abrupt_discontinuation text -> honest "no info".
        "must_contain": ["don't have"],
        "must_not_contain": ["gradual dose reduction"],  # not in the data
    },
]


SCENARIOS = [
    ("scenario1", SCENARIO_1),
    ("scenario2", SCENARIO_2),
]


# ===========================================================================
# Layer 1: repository (deterministic, no API)
# ===========================================================================


@pytest.mark.parametrize("name,resolved", [
    ("Prozac", "Fluoxetine"),
    ("Cymbalta", "Duloxetine"),
    ("Fluoxetine", "Fluoxetine"),
    ("Duloxetine", "Duloxetine"),
])
def test_repository_resolves_brand_and_generic(repo, name, resolved):
    drug = repo.resolve_drug(name)
    assert drug is not None
    assert drug["name"] == resolved


def test_repository_unknown_drug_returns_none(repo):
    assert repo.resolve_drug("Asdfgh") is None
    assert repo.resolve_drug(None) is None


@pytest.mark.parametrize("scenario_name,steps", SCENARIOS)
def test_repository_evidence_for_each_step(repo, scenario_name, steps):
    for step in steps:
        drug = repo.resolve_drug(step["drug_name"] or step["resolved"])
        ev = repo.get_evidence(drug=drug, intent=step["intent"])
        assert isinstance(ev, Evidence)
        assert ev.drug_name == step["resolved"]
        assert ev.field == step["intent"]
        assert ev.source == "NbN P&F"


def test_repository_duloxetine_discontinuation_is_none(repo):
    """Documents the known data gap: Duloxetine has no abrupt_discontinuation."""
    drug = repo.resolve_drug("Duloxetine")
    ev = repo.get_evidence(drug=drug, intent="discontinuation")
    assert ev is not None
    assert ev.text is None


def test_repository_unknown_intent_returns_none(repo):
    drug = repo.resolve_drug("Fluoxetine")
    assert repo.get_evidence(drug=drug, intent="bogus") is None
    assert repo.get_evidence(drug=drug, intent=None) is None


# ===========================================================================
# Layer 2: parser intent (LLM)
# ===========================================================================


@pytest.mark.llm
@pytest.mark.parametrize("scenario_name,steps", SCENARIOS)
def test_parser_intents_across_scenario(parser, scenario_name, steps):
    current_drug = None
    for step in steps:
        q = parser.parse(step["question"], current_drug=current_drug)
        # drug_name: either the named drug, or falls back to current_drug.
        if step["drug_name"] is not None:
            assert q.drug_name == step["drug_name"], (
                f"{scenario_name}: {step['question']!r} -> "
                f"drug_name={q.drug_name!r}, expected {step['drug_name']!r}"
            )
        assert q.intent == step["intent"], (
            f"{scenario_name}: {step['question']!r} -> "
            f"intent={q.intent!r}, expected {step['intent']!r}"
        )
        # Advance context: the resolved drug becomes current for the next turn.
        if q.drug_name:
            from src.repository import DrugRepository
            resolved = DrugRepository("data/nbnpf_drugs.json").resolve_drug(q.drug_name)
            current_drug = resolved["name"] if resolved else current_drug


@pytest.mark.llm
def test_parser_context_fallback_is_deterministic(parser, repo):
    """'Is it addictive?' with no named drug should fall back to current_drug."""
    q = parser.parse("Is it addictive?", current_drug="Fluoxetine")
    assert q.drug_name == "Fluoxetine"
    assert q.intent == "addiction"


# ===========================================================================
# Layer 3: generator wording (LLM)
# ===========================================================================


@pytest.mark.llm
@pytest.mark.parametrize("scenario_name,steps", SCENARIOS)
def test_generator_wording_across_scenario(parser, repo, generator, scenario_name, steps):
    current_drug = None
    for step in steps:
        q = parser.parse(step["question"], current_drug=current_drug)
        drug = repo.resolve_drug(q.drug_name)
        if drug:
            current_drug = drug["name"]
        ev = repo.get_evidence(drug=drug, intent=q.intent)
        answer = generator.generate(step["question"], drug, ev)

        assert answer, f"empty answer for {step['question']!r}"
        lower = answer.lower()
        for word in step["must_contain"]:
            assert word.lower() in lower, (
                f"{scenario_name}: {step['question']!r} -> "
                f"answer missing {word!r}\n--- answer ---\n{answer}"
            )
        for word in step.get("must_not_contain", []):
            assert word.lower() not in lower, (
                f"{scenario_name}: {step['question']!r} -> "
                f"answer should not contain {word!r}\n--- answer ---\n{answer}"
            )


@pytest.mark.llm
def test_generator_no_drug_message(generator):
    answer = generator.generate("hello", drug=None, evidence=None)
    assert "couldn't identify" in answer.lower()


@pytest.mark.llm
def test_generator_source_footer_when_evidence_present(parser, repo, generator):
    q = parser.parse("Is Prozac addictive?")
    drug = repo.resolve_drug(q.drug_name)
    ev = repo.get_evidence(drug=drug, intent=q.intent)
    answer = generator.generate("Is Prozac addictive?", drug, ev)
    assert "Source: NbN P&F" in answer
