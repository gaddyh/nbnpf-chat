from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Make `src` importable when running pytest from the repo root.
sys.path.insert(0, str(Path(__file__).parent))


REPO_PATH = "data/nbnpf_drugs.json"


@pytest.fixture(scope="session")
def repo():
    from src.repository import DrugRepository

    return DrugRepository(REPO_PATH)


@pytest.fixture(scope="session")
def parser():
    from src.query_parser import QueryParser

    return QueryParser()


@pytest.fixture(scope="session")
def generator():
    from src.answer_generator import AnswerGenerator

    return AnswerGenerator()


def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


# Marker for tests that make real OpenAI API calls.
def pytest_configure(config):
    config.addinivalue_line("markers", "llm: marks tests that call the OpenAI API")


def pytest_collection_modifyitems(config, items):
    skip_llm = pytest.mark.skip(reason="OPENAI_API_KEY not set; skipping LLM tests")
    if not has_openai_key():
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip_llm)
