from __future__ import annotations

import json

from src.models import Drug, Evidence, Intent


# intent -> (top-level field, timeline sub-key or None)
INTENT_FIELD_MAP: dict[str, tuple[str, str | None]] = {
    Intent.APPROVED_USE.value: ("approved_uses", None),
    Intent.EFFICACY.value: ("additional_efficacy", None),
    Intent.SIDE_EFFECTS.value: ("side_effects", None),
    Intent.ADDICTION.value: ("addiction", None),
    Intent.SCIENCE.value: ("science", None),
    Intent.PHARMACOLOGY.value: ("pharmacology", None),
    Intent.HOW_IT_WORKS.value: ("how_it_works", None),
    Intent.TIMELINE_ONSET.value: ("timeline", "onset"),
    Intent.TIMELINE_MAINTENANCE.value: ("timeline", "maintenance"),
    Intent.DISCONTINUATION.value: ("timeline", "abrupt_discontinuation"),
}


class DrugRepository:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.drugs: list[Drug] = json.load(f)

        self.by_alias: dict[str, Drug] = {}

        for drug in self.drugs:
            self.by_alias[drug["name"].lower()] = drug
            for brand in drug.get("brand_names", []):
                self.by_alias[brand.lower()] = drug

    def resolve_drug(self, name: str | None) -> Drug | None:
        if not name:
            return None
        return self.by_alias.get(name.lower())

    def get_evidence(self, drug: Drug | None, intent: str | None) -> Evidence | None:
        if not drug:
            return None

        mapping = INTENT_FIELD_MAP.get(intent or "")
        if not mapping:
            return None

        field, sub_key = mapping
        source_info = drug.get("_source", {})
        drug_id = source_info.get("drug_id", drug.get("id"))

        if sub_key:
            timeline = drug.get(field) or {}
            text = timeline.get(sub_key) if isinstance(timeline, dict) else None
        else:
            text = drug.get(field)

        return Evidence(
            drug_name=drug["name"],
            field=intent or "",
            text=text,
            source=source_info.get("source", "NbN P&F"),
            drug_id=drug_id,
        )
