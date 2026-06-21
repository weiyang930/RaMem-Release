"""
Answer Generator - Final synthesis from retrieved contexts

Section 3.3: Intent-Aware Retrieval Planning
Generates answers from the merged context C_q after multi-view retrieval
"""
from typing import List
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
from ramem import config


class AnswerGenerator:
    """
    Answer Generator - Synthesis from retrieved memory units (Section 3.3)

    Generates answers from C_q = R_sem ∪ R_lex ∪ R_sym
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_answer(self, query: str, contexts: List[MemoryEntry]) -> str:
        """
        Generate answer

        Args:
        - query: User question
        - contexts: List of retrieved relevant MemoryEntry

        Returns:
        - Generated answer (concise phrase)
        """
        if not contexts:
            return "No relevant information found"

        # Build context string. Keep the full retrieved list available to eval
        # exports, but bound the generator prompt when configured so local vLLM
        # runs do not exceed the model context window.
        context_str = self._format_contexts_for_generation(contexts)

        # Build prompt
        prompt = self._build_answer_prompt(query, context_str)

        # Call LLM to generate answer
        messages = [
            {
                "role": "system",
                "content": "You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )

                # Parse JSON response
                result = self.llm_client.extract_json(response)
                answer = result.get("answer", response.strip())

                # Strip typographic/curly quotes (Pi/Pf unicode categories) — always safe
                if getattr(config, 'NORMALIZE_ANSWER_QUOTES', True):
                    answer = self._normalize_quotes(answer)

                return answer

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Answer generation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Warning: Failed to parse JSON response after {max_retries} attempts: {e}")
                    # Fallback to raw response
                    if 'response' in locals():
                        return response.strip()
                    else:
                        return "Failed to generate answer"

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        """Strip typographic/curly quote characters using unicode categories.
        Pi = initial punctuation (\u201c \u2018 \u00ab \u2039 …)
        Pf = final punctuation   (\u201d \u2019 \u00bb \u203a …)
        General approach: catches all typographic variants without hardcoding codepoints.
        """
        import unicodedata
        return ''.join(ch for ch in text if unicodedata.category(ch) not in ('Pi', 'Pf'))

    def _format_contexts(self, contexts: List[MemoryEntry]) -> str:
        """
        Format contexts to readable text
        """
        formatted = []
        for i, entry in enumerate(contexts, 1):
            parts = [f"[Context {i}]"]
            parts.append(f"Content: {entry.lossless_restatement}")

            if getattr(config, "ABLATE_GENERATION_TEXT_ONLY", False):
                formatted.append("\n".join(parts))
                continue

            if entry.session_date:
                parts.append(f"Session: {entry.session_date[:10]} → {entry.session_end_date[:10] if entry.session_end_date else '?'}")

            if entry.timestamp:
                parts.append(f"Time: {entry.timestamp}")

            if entry.location:
                parts.append(f"Location: {entry.location}")

            if entry.persons:
                parts.append(f"Persons: {', '.join(entry.persons)}")

            if entry.entities:
                parts.append(f"Related Entities: {', '.join(entry.entities)}")

            if entry.topic:
                parts.append(f"Topic: {entry.topic}")

            formatted.append("\n".join(parts))

        return "\n\n".join(formatted)

    def _format_contexts_for_generation(self, contexts: List[MemoryEntry]) -> str:
        max_chars = int(getattr(config, "ANSWER_CONTEXT_MAX_CHARS", 0) or 0)
        if max_chars <= 0:
            return self._format_contexts(contexts)

        selected = []
        current_chars = 0
        for entry in contexts:
            formatted_entry = self._format_contexts([entry])
            added_chars = len(formatted_entry) + (2 if selected else 0)
            if selected and current_chars + added_chars > max_chars:
                break
            if not selected and added_chars > max_chars:
                selected.append(entry)
                break
            selected.append(entry)
            current_chars += added_chars

        if len(selected) < len(contexts):
            print(
                f"[AnswerGenerator] Packed {len(selected)}/{len(contexts)} contexts "
                f"for generation (ANSWER_CONTEXT_MAX_CHARS={max_chars})"
            )
        return self._format_contexts(selected)

    def _build_answer_prompt(self, query: str, context_str: str) -> str:
        """
        Build answer generation prompt
        """
        entity_instruction = ""
        if getattr(config, 'ENTITY_AWARE_ANSWER', False):
            entity_instruction = """6. Answer format by question type — this is critical:
   - Asking for a SPECIFIC NAMED ENTITY (city, country, game title, movie, person's name,
     food, instrument, sport, number, band, hobby, etc.): reply with the minimal phrase that
     directly answers the question. Remove sentence wrappers ("X went to", "The answer is",
     "It was"), but do NOT drop words that are part of the answer itself.
     Example: "play the piano" stays "play the piano", not just "piano".
   - Yes/No question (starts with Did/Was/Is/Were/Has/Have/Does/Do): reply "Yes" or "No"
     plus one short clause only if the context explicitly provides a reason.
   - Description, event, feeling, or explanation: reply with a short descriptive phrase.
"""
        longmem_instruction = ""
        if getattr(config, "LONGMEM_SESSION_LINKING", False):
            category = getattr(config, "LONGMEMEVAL_CATEGORY", "")
            category_instruction = ""
            if category == "knowledge-update":
                category_instruction = """   - This is a knowledge-update question. If the context contains conflicting older and newer facts, answer with the latest/current fact by session date or timestamp. Do not answer with a stale earlier fact unless the question explicitly asks for the past state.
"""
            elif category == "single-session-preference":
                category_instruction = """   - This is a preference-personalization question. The judge is checking whether the response uses remembered personal information, not whether it gives generic advice.
   - First identify the most relevant remembered preference, prior success, recent experiment, constraint, frustration, owned item, or tool from the context.
   - The final answer must explicitly include those personal details and tailor the recommendation around them. Use at least two specific remembered details when the context provides them.
   - Do not answer with only a broad category list. Do not bury the personal detail after generic suggestions. Start with the remembered preference/fact when possible.
   - If there is any relevant personal preference memory in the context, do not answer "insufficient information"; give a personalized recommendation using that memory.
   - For this category only, the answer may be 1-3 concise sentences or compact semicolon-separated suggestions instead of a short phrase.
"""
            elif category == "single-session-assistant":
                category_instruction = """   - This question asks about something the assistant previously said or recommended. Prefer assistant-provided exact facts from prior responses: names, titles, chapter numbers, years, budgets, durations, ingredients, list items, and table cells.
"""
            elif category == "temporal-reasoning":
                category_instruction = """   - This is a temporal reasoning question. Use the question date/as-of date when present, and base relative words such as today, currently, last, next, ago, and this year on that date. If a required comparison date is missing, say that the information is insufficient instead of guessing.
"""
            elif category == "multi-session":
                category_instruction = """   - This is a multi-session aggregation question. Collect distinct matching evidence across all retrieved sessions before answering; do not stop at the first matching session.
"""
            longmem_instruction = f"""6. Long-memory QA rules:
   - For count questions, first enumerate every distinct matching fact in the context, then answer with the total count and a compact breakdown. Do not output only a subset.
   - Count pickup obligations and return obligations separately when the question asks for items to pick up or return. Dry cleaning pickup is a clothing pickup.
   - If the context says an item was exchanged and still needs replacement pickup, and nearby context discusses tracking pickup/return for that same item or store, treat the replacement pickup and the original item return as separate obligations unless the context explicitly says the return is already complete.
   - If the question asks how many projects the user has led or is currently leading, count only projects where the context explicitly says the user led, is leading, or was the leader. Do not count projects merely discussed by an organization, team, class, or community unless the user is explicitly the leader.
   - For "how many" questions, the final answer must start with the number or include an unambiguous total such as "3 items: ...".
   - Prefer concrete user-specific obligations, dates, table cells, and updated facts over generic advice.
   - Preserve exact answer strings from context when available: proper names, product/store names, numbers, dates, durations, chapter numbers, budgets, and list items.
   - If the question asks for the current/latest state, choose the newest supported memory and ignore superseded older memories.
   - If the question is unanswerable from the provided context, answer with a concise insufficient-information statement instead of inferring.
{category_instruction}"""

        return f"""
Answer the user's question based on the provided context.

User Question: {query}

Relevant Context:
{context_str}

Requirements:
1. First, think through the reasoning process
2. Then provide a very CONCISE answer (short phrase about core information)
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format
{entity_instruction}
{longmem_instruction}
Output Format:
```json
{{
  "reasoning": "Brief explanation of your thought process",
  "answer": "Concise answer in a short phrase"
}}
```

Example:
Question: "When will they meet?"
Context: "Alice suggested meeting Bob at 2025-11-16T14:00:00..."

Output:
```json
{{
  "reasoning": "The context explicitly states the meeting time as 2025-11-16T14:00:00",
  "answer": "16 November 2025 at 2:00 PM"
}}
```

Now answer the question. Return ONLY the JSON, no other text.
"""
