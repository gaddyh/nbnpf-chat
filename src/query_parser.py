from __future__ import annotations

from openai import OpenAI

from src.models import Intent, ParsedQuery, Query


SYSTEM_PROMPT = """\
You are a query parser for a medication information chatbot (NbN Patients & Families).

From the user's question, extract:
- drug_name: the medication the user is asking about, by generic or brand name.
  Return null if the question does not name a drug (even if it refers to "it" or
  "this medication" — the caller resolves that to the current drug).
- intent: exactly one of {intents}. Return null if the question does not map to
  any of them.

Do not infer a drug from context. Only set drug_name when the user explicitly
names one in this question.

Map natural patient questions to intents using these examples:
- approved_use: "What is it for?", "What does it treat?", "Why was I prescribed this?"
- efficacy: "Does it actually work?", "How well does it work for anxiety?"
- side_effects: "What are the side effects?", "I feel sleepy, could it be related?", \
"Does it affect sexual function?", "Any side effects I should know about?"
- addiction: "Is it addictive?", "Will I get dependent?", "Can I get hooked on it?"
- science: "Why would it help with OCD?", "What is it doing in the brain?", \
"How does it work mechanically?", "What's the science behind it?"
- pharmacology: "What neurotransmitters does it affect?", "What does it do to serotonin?"
- how_it_works: "How does it work?", "What does it do in the body?" (plain-language)
- timeline_onset: "How long until it works?", "When will I feel better?", \
"How long does it take to kick in?"
- timeline_maintenance: "How long do I need to take it?", "How long is the treatment?"
- discontinuation: "Can I stop it?", "What happens if I stop?", "Is it safe to stop abruptly?"

When in doubt, prefer the closest match over null. A question about a specific symptom \
the drug might cause is side_effects. A question about why/how the drug works biologically \
is science."""


# Few-shot examples as user/assistant turns. These are stronger than
# examples listed in the system prompt because the model sees the exact
# input/output mapping it must produce.
FEW_SHOT_EXAMPLES: list[dict] = [
    {"role": "user", "content": "I was prescribed Prozac. What is it for?"},
    {"role": "assistant", "content": '{"drug_name":"Prozac","intent":"approved_use"}'},
    {"role": "user", "content": "Why would it help with OCD?"},
    {"role": "assistant", "content": '{"drug_name":null,"intent":"science"}'},
    {"role": "user", "content": "I feel very sleepy. Could it be related?"},
    {"role": "assistant", "content": '{"drug_name":null,"intent":"side_effects"}'},
    {"role": "user", "content": "Does it also affect sexual function?"},
    {"role": "assistant", "content": '{"drug_name":null,"intent":"side_effects"}'},
    {"role": "user", "content": "How long does it take to work?"},
    {"role": "assistant", "content": '{"drug_name":null,"intent":"timeline_onset"}'},
    {"role": "user", "content": "Is it addictive?"},
    {"role": "assistant", "content": '{"drug_name":null,"intent":"addiction"}'},
    {"role": "user", "content": "What is it doing in the brain?"},
    {"role": "assistant", "content": '{"drug_name":null,"intent":"science"}'},
    {"role": "user", "content": "Can I stop it if the side effects bother me?"},
    {"role": "assistant", "content": '{"drug_name":null,"intent":"discontinuation"}'},
]


class QueryParser:
    def __init__(self, model: str = "gpt-4o-mini"):
        self._client: OpenAI | None = None
        self.model = model

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()  # reads OPENAI_API_KEY from env
        return self._client

    def parse(self, prompt: str, current_drug: str | None = None) -> Query:
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            response_format=ParsedQuery,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(intents=Intent.values()),
                },
                *FEW_SHOT_EXAMPLES,
                {"role": "user", "content": prompt},
            ],
        )

        parsed: ParsedQuery | None = response.choices[0].message.parsed
        drug_name = parsed.drug_name if parsed else None
        intent = parsed.intent if parsed else None

        # Deterministic fallback: never trust the LLM with context.
        if drug_name is None:
            drug_name = current_drug

        if intent and intent not in Intent.values():
            intent = None

        return Query(drug_name=drug_name, intent=intent)
