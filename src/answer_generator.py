from __future__ import annotations

from openai import OpenAI

from src.models import Drug, Evidence


SYSTEM_PROMPT = """\
You are a helpful assistant for the NbN Patients & Families medication chatbot.

You answer the patient's question using ONLY the provided evidence about the drug.
Rules:
- Use only the facts in <evidence>. Never invent, assume, or pull in outside knowledge.
- If the evidence is empty, null, or does not cover the question, say you don't have
  that information and suggest asking their clinician.
- Write in plain, reassuring language for a patient/family member. Keep it concise.
- Do not give dosing instructions or medical advice beyond what the evidence states.
- End with a short line: "Source: NbN P&F" when evidence was used."""


class AnswerGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self._client: OpenAI | None = None
        self.model = model

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()  # reads OPENAI_API_KEY from env
        return self._client

    def generate(
        self,
        question: str,
        drug: Drug | None,
        evidence: Evidence | None,
    ) -> str:
        if not drug:
            return (
                "I couldn't identify which medication you're asking about. "
                "Could you name the drug (generic or brand name)?"
            )

        if evidence and evidence.text:
            evidence_str = evidence.text
        else:
            evidence_str = "(no information available for this topic)"

        user_msg = (
            f"Drug: {drug['name']}\n\n"
            f"<evidence>\n{evidence_str}\n</evidence>\n\n"
            f"Question: {question}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )

        return response.choices[0].message.content or ""
