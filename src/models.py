from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel


class Intent(str, Enum):
    PHARMACOLOGY = "pharmacology"
    HOW_IT_WORKS = "how_it_works"
    APPROVED_USE = "approved_use"
    EFFICACY = "efficacy"
    SIDE_EFFECTS = "side_effects"
    ADDICTION = "addiction"
    SCIENCE = "science"
    TIMELINE_ONSET = "timeline_onset"
    TIMELINE_MAINTENANCE = "timeline_maintenance"
    DISCONTINUATION = "discontinuation"

    @classmethod
    def values(cls) -> list[str]:
        return [i.value for i in cls]


class Timeline(TypedDict, total=False):
    onset: str
    maintenance: str
    important_aids: str
    abrupt_discontinuation: str
    other: object


class Source(TypedDict, total=False):
    source: str
    drug_id: int


class Drug(TypedDict, total=False):
    id: int
    name: str
    brand_names: list[str]
    pharmacology: str
    how_it_works: str
    approved_uses: str
    additional_efficacy: str
    side_effects: str
    addiction: str
    timeline: Timeline
    why_take_it: str
    science: str
    _source: Source
    _quality_flags: list


@dataclass
class Query:
    drug_name: str | None
    intent: str | None

    @property
    def intent_enum(self) -> Intent | None:
        try:
            return Intent(self.intent)
        except (ValueError, TypeError):
            return None


@dataclass
class Evidence:
    drug_name: str
    field: str
    text: str | None
    source: str
    drug_id: int | None


class ParsedQuery(BaseModel):
    """Schema enforced by OpenAI Structured Outputs."""

    drug_name: str | None = None
    intent: str | None = None
