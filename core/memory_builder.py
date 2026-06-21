"""
Memory Builder
Stage 1: Semantic Structured Compression (Section 3.1)
& Stage 2: Online Semantic Synthesis (Section 3.2)
Implements:
- Implicit semantic density gating: Φ_gate(W) → {m_k} (filters low-density windows)
- Sliding window processing for dialogue segmentation
- Generates compact memory units with resolved coreferences and absolute timestamps
"""
from typing import List, Optional, Tuple
from models.memory_entry import MemoryEntry, Dialogue
from utils.llm_client import LLMClient
from database.vector_store import VectorStore
from ramem import config
import json
import asyncio
import concurrent.futures
from functools import partial
import os


def _extract_session_spans(dialogues: List[Dialogue]) -> List[Tuple[str, str]]:
    """
    Return an ordered, deduplicated list of (session_date, session_end_date)
    tuples from the dialogues in this window.
    Each unique session_date appears once, preserving order.
    """
    seen = set()
    spans = []
    for d in dialogues:
        if d.session_date and d.session_date not in seen:
            seen.add(d.session_date)
            spans.append((d.session_date, d.session_end_date or "2099-12-31T00:00:00"))
    return spans


def _match_session_span(
    timestamp: Optional[str],
    spans: Optional[List[Tuple[str, str]]]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Given an entry's LLM-extracted timestamp and the window's session spans,
    return the (session_date, session_end_date) whose range contains the timestamp.

    Falls back to the first span if no match or no timestamp.
    """
    if not spans:
        return None, None

    fallback = spans[0]

    if not timestamp:
        return fallback

    # Normalise timestamp for string comparison (ISO 8601 sorts lexicographically)
    ts = timestamp[:19]  # trim sub-seconds if any

    for session_date, session_end_date in spans:
        if session_date <= ts < session_end_date:
            return session_date, session_end_date

    # Timestamp outside all spans in window — use closest span by start date
    # (handles cases where LLM inferred a slightly different time)
    closest = min(spans, key=lambda s: abs(
        _iso_to_int(s[0]) - _iso_to_int(ts)
    ))
    return closest


def _looks_like_iso_timestamp(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    return bool(__import__("re").match(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$", value.strip()))


def _normalize_optional_string(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts) if parts else None
    return str(value).strip() or None


def _normalize_string_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value).strip()] if str(value).strip() else []


def _normalize_memory_item(item: dict) -> dict:
    normalized = dict(item)
    timestamp = _normalize_optional_string(normalized.get("timestamp"))
    normalized["timestamp"] = timestamp if _looks_like_iso_timestamp(timestamp) else None
    normalized["location"] = _normalize_optional_string(normalized.get("location"))
    normalized["topic"] = _normalize_optional_string(normalized.get("topic"))
    normalized["keywords"] = _normalize_string_list(normalized.get("keywords"))
    normalized["persons"] = _normalize_string_list(normalized.get("persons"))
    normalized["entities"] = _normalize_string_list(normalized.get("entities"))
    normalized["lossless_restatement"] = str(normalized.get("lossless_restatement", "")).strip()
    return normalized


def _iso_to_int(iso: str) -> int:
    """Convert ISO date string to comparable integer (strip non-numeric chars)."""
    return int(iso[:19].replace("-", "").replace("T", "").replace(":", ""))


class MemoryBuilder:
    """
    Memory Builder - Semantic Structured Compression (Section 3.1)

    Core Functions:
    1. Sliding window segmentation
    2. Implicit semantic density gating: Φ_gate(W) → {m_k}
    3. Multi-view indexing: I(m_k) = {s_k, l_k, r_k}
    4. Intra-session consolidation during write (Section 3.2): by generating enough memory entries to ensure ALL information is captured
    """
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        window_size: int = None,
        enable_parallel_processing: bool = True,
        max_parallel_workers: int = 3
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.window_size = window_size or config.WINDOW_SIZE
        self.overlap_size = getattr(config, 'OVERLAP_SIZE', 0)
        # step_size is how far the window advances each iteration; overlap retains
        # the last overlap_size dialogues so the next window has continuity context
        self.step_size = max(1, self.window_size - self.overlap_size)

        # Use config values as default if not explicitly provided
        self.enable_parallel_processing = enable_parallel_processing if enable_parallel_processing is not None else getattr(config, 'ENABLE_PARALLEL_PROCESSING', True)
        self.max_parallel_workers = max_parallel_workers if max_parallel_workers is not None else getattr(config, 'MAX_PARALLEL_WORKERS', 4)

        # Dialogue buffer
        self.dialogue_buffer: List[Dialogue] = []
        self.processed_count = 0

        # Previous window entries (for context)
        self.previous_entries: List[MemoryEntry] = []

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True):
        """
        Add a dialogue to the buffer
        """
        self.dialogue_buffer.append(dialogue)

        # Auto process
        if auto_process and len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def add_dialogues(self, dialogues: List[Dialogue], auto_process: bool = True):
        """
        Batch add dialogues with optional parallel processing
        """
        if self.enable_parallel_processing and len(dialogues) > self.window_size * 2:
            # Use parallel processing for large batches
            self.add_dialogues_parallel(dialogues)
        else:
            # Use sequential processing for smaller batches
            for dialogue in dialogues:
                self.add_dialogue(dialogue, auto_process=False)

            # Process complete windows
            if auto_process:
                while len(self.dialogue_buffer) >= self.window_size:
                    self.process_window()
    
    def add_dialogues_parallel(self, dialogues: List[Dialogue]):
        """
        Add dialogues using parallel processing for better performance
        """
        # Snapshot pre-existing buffer items so the fallback can restore them
        # if the buffer is cleared mid-way through parallel processing
        pre_existing = list(self.dialogue_buffer)
        windows_to_process = []
        try:
            # Add all dialogues to buffer first
            self.dialogue_buffer.extend(dialogues)

            # Group into windows using step_size so that each window retains
            # overlap_size dialogues of context from the previous window
            pos = 0
            while pos + self.window_size <= len(self.dialogue_buffer):
                window = self.dialogue_buffer[pos:pos + self.window_size]
                windows_to_process.append(window)
                pos += self.step_size

            # Add remaining dialogues as a smaller batch (no need to process separately)
            remaining = self.dialogue_buffer[pos:]
            if remaining:
                windows_to_process.append(remaining)
            self.dialogue_buffer = []  # Clear buffer since we're processing all

            if windows_to_process:
                print(f"\n[Parallel Processing] Processing {len(windows_to_process)} batches in parallel with {self.max_parallel_workers} workers")
                print(f"Batch sizes: {[len(w) for w in windows_to_process]}")

                # Process all windows/batches in parallel (including remaining dialogues)
                self._process_windows_parallel(windows_to_process)

        except Exception as e:
            print(f"[Parallel Processing] Failed: {e}. Falling back to sequential processing...")
            # Fallback: overlapping windows cannot be re-stacked naively.
            # If the buffer was cleared (exception after line 107), restore the full
            # original state: pre-existing items that were already in the buffer
            # PLUS the new dialogues we were asked to process.
            # If the buffer was NOT cleared (exception before line 107), it already
            # contains pre_existing + dialogues, so leave it as-is.
            if not self.dialogue_buffer:
                self.dialogue_buffer = pre_existing + list(dialogues)
            # process_window() uses step_size, so overlap is handled correctly here
            while len(self.dialogue_buffer) >= self.window_size:
                self.process_window()

    def process_window(self):
        """
        Process current window dialogues - Core logic
        """
        if not self.dialogue_buffer:
            return

        # Extract window; advance by step_size to retain overlap_size dialogues
        # at the tail so the next window has continuity context
        window = self.dialogue_buffer[:self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.step_size:]

        print(f"\nProcessing window: {len(window)} dialogues (processed {self.processed_count} so far)")

        # Call LLM to generate memory entries
        entries = self._generate_memory_entries(window)

        # Store to database
        if entries:
            self.vector_store.add_entries(entries)
            self.previous_entries = entries  # Save as context
            self.processed_count += len(window)

        print(f"Generated {len(entries)} memory entries")

    def process_remaining(self):
        """
        Process remaining dialogues (fallback method, normally handled in parallel)
        """
        if self.dialogue_buffer:
            print(f"\nProcessing remaining dialogues: {len(self.dialogue_buffer)} (fallback mode)")
            entries = self._generate_memory_entries(self.dialogue_buffer)
            if entries:
                self.vector_store.add_entries(entries)
                self.processed_count += len(self.dialogue_buffer)
            self.dialogue_buffer = []
            print(f"Generated {len(entries)} memory entries")

    def _generate_memory_entries(self, dialogues: List[Dialogue]) -> List[MemoryEntry]:
        """
        Implicit Semantic Density Gating (Section 3.1)
        Φ_gate(W) → {m_k}, generates compact memory units from dialogue window
        """
        # Build dialogue text
        dialogue_text = "\n".join([str(d) for d in dialogues])
        dialogue_ids = [d.dialogue_id for d in dialogues]

        # Build an ordered list of unique session spans from this window.
        # A window spans multiple sessions; we need per-session spans so each
        # generated entry can be matched to the session its timestamp belongs to.
        window_session_spans = _extract_session_spans(dialogues)

        # Build context
        context = ""
        if self.previous_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for entry in self.previous_entries[:3]:  # Only show first 3
                context += f"- {entry.lossless_restatement}\n"

        # Build prompt
        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, context)

        # Call LLM
        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times if parsing fails
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
                    response_format=response_format,
                    max_tokens=getattr(config, "MEMORY_EXTRACTION_MAX_TOKENS", None),
                )

                # Parse response — stamp correct session span per entry
                entries = self._parse_llm_response(
                    response, dialogue_ids,
                    window_session_spans
                )
                return entries

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed to parse LLM response: {e}")
                    print(f"Retrying...")
                else:
                    print(f"All {max_retries} attempts failed to parse LLM response: {e}")
                    print(f"Raw response: {response[:500] if 'response' in locals() else 'No response'}")
                    return []

    def _build_extraction_prompt(
        self,
        dialogue_text: str,
        dialogue_ids: List[int],
        context: str
    ) -> str:
        """
        Build LLM extraction prompt
        """
        model_name = (
            os.getenv("QWEN_LLM_MODEL")
            or getattr(config, "LLM_MODEL", "")
            or ""
        ).lower()
        no_think_prefix = "/no_think\n" if "qwen3" in model_name else ""
        output_format = """
[Output Format]
Return a JSON object with a "memory_entries" array. The array must contain one object per memory entry:

```json
{
  "memory_entries": [
    {
      "lossless_restatement": "Complete unambiguous restatement (must include all subjects, objects, time, location, etc.)",
      "keywords": ["keyword1", "keyword2"],
      "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
      "location": "location name or null",
      "persons": ["name1", "name2"],
      "entities": ["entity1", "entity2"],
      "topic": "topic phrase"
    }
  ]
}
```
"""
        example_output = """
Output:
```json
{
  "memory_entries": [
    {
      "lossless_restatement": "Alice suggested at 2025-11-15T14:30:00 to meet with Bob at Starbucks on 2025-11-16T14:00:00 to discuss the new product.",
      "keywords": ["Alice", "Bob", "Starbucks", "new product", "meeting"],
      "timestamp": "2025-11-16T14:00:00",
      "location": "Starbucks",
      "persons": ["Alice", "Bob"],
      "entities": ["new product"],
      "topic": "Product discussion meeting arrangement"
    },
    {
      "lossless_restatement": "Bob agreed to attend the meeting and committed to prepare relevant materials.",
      "keywords": ["Bob", "prepare materials", "agree"],
      "timestamp": null,
      "location": null,
      "persons": ["Bob"],
      "entities": [],
      "topic": "Meeting preparation confirmation"
    }
  ]
}
```
"""
        longmem_fact_instruction = ""
        max_entries = 12
        if getattr(config, "LONGMEM_SESSION_LINKING", False):
            if getattr(config, "LONGMEMEVAL_MODE", False):
                max_entries = 20 if getattr(config, "LONGMEMEVAL_MEMORY_LAYOUT", "") == "session" else 35
            else:
                max_entries = 20
            longmem_fact_instruction = """
7. **Preserve Same-Session Links**:
   - When later turns refer to an item, coupon, purchase, plan, preference, or event and earlier turns in the same dialogue establish the store, app, place, person, or organization, include that linked context in the later memory entry.
   - Example: if the user says they use Target's Cartwheel app and later says they redeemed a $5 coupon on coffee creamer, write a memory such as "The user redeemed a $5 coupon on coffee creamer at Target using the Cartwheel app" rather than omitting Target.
   - Do not invent links across unrelated topics; only preserve links that are supported by nearby turns in the current dialogue window.
8. **Preserve QA-Critical Details**:
   - Preserve dated personal events as standalone facts, especially visits, appointments, purchases, returns, pickups, trips, classes, competitions, and exhibits. Include the exact event name and date in the same entry.
   - For museum/gallery/exhibit visits, preserve the user's visit as a standalone dated event even when the rest of the dialogue is about generic recommendations. Example: "I attended the 'Ancient Civilizations' exhibit at the Metropolitan Museum of Art today" must become "The user attended the 'Ancient Civilizations' exhibit at the Metropolitan Museum of Art on <session date>."
   - Preserve countable obligations as separate entries. For example, if the user needs to pick up dry cleaning and pick up exchanged boots, write separate entries for each item instead of a broad closet-organization summary.
   - Preserve exchange and return obligations explicitly. If an exchange implies both receiving a replacement item and returning the original item, create separate entries for the replacement pickup and the original return, including item names, store, size/color, and due timing when available.
   - Preserve table, schedule, checklist, and roster cells as answerable facts. If a table assigns a person to a day, shift, role, or time, extract entries such as "Admon is assigned to the 8 am-4 pm shift on Sunday" instead of only saying a rotation sheet was created.
   - Prefer concrete personal/user-specific facts over generic advice when the window contains both.
9. **LongMemEval Fact Recall**:
   - Preserve tiny but answerable facts even when they appear incidental: pet names, family names, doctors, schools, restaurants, product/store/app names, current locations, hobbies, diagnoses, allergies, medications, contact names, and relationship facts.
   - Preserve exact numbers and units as standalone facts: counts, prices, budgets, percentages, years, chapter numbers, durations, distances, serving sizes, sizes, colors, and appointment times.
   - Preserve assistant-provided facts from recommendations, recipes, travel plans, budgets, legal/history explanations, ranked lists, checklists, and tables. If the assistant names an item, chapter, year, budget line, dish, ingredient, or duration, extract it exactly.
   - Preserve user preferences and constraints as direct memory entries: likes, dislikes, avoided options, preferred languages/tools/styles, past successes, frustrations, and requirements for future recommendations.
   - For updates or corrections, preserve both the old and new facts with the session timestamp, and explicitly mark the newer/current state when the dialogue states that something changed.
   - Avoid using entry budget on generic advice unless it contains an exact fact that the user may later ask about.
"""
        final_instruction = 'Now process the above dialogues. Return ONLY the JSON object with the "memory_entries" array, no other explanations.'

        return f"""
{no_think_prefix}
Your task is to extract all valuable information from the following dialogues and convert them into structured memory entries.

{context}

[Current Window Dialogues]
{dialogue_text}

[Requirements]
1. **Complete Coverage**: Generate enough memory entries to ensure ALL information in the dialogues is captured
2. **Force Disambiguation**: Absolutely PROHIBIT using pronouns (he, she, it, they, this, that) and relative time (yesterday, today, last week, tomorrow)
3. **Lossless Information**: Each entry's lossless_restatement must be a complete, independent, understandable sentence
4. **Precise Extraction**:
   - keywords: Core keywords (names, places, entities, topic words)
   - timestamp: Absolute time in ISO 8601 format (if explicit time mentioned in dialogue)
   - location: Specific location name (if mentioned)
   - persons: All person names mentioned
   - entities: Companies, products, organizations, etc.
   - topic: The topic of this information
5. **Entry Granularity**:
   - Create separate entries for separate facts, preferences, plans, events, locations, relationships, health details, work details, and media references
   - Do not compress an entire 40-dialogue window into one summary entry
   - Return 6 to {max_entries} high-value entries when enough information is present
   - Never return more than {max_entries} entries for one window; if there are more candidate facts, keep only the most useful facts for future question answering
   - The final JSON is invalid for this task if "memory_entries" contains more than {max_entries} objects
6. **Strict JSON**:
   - Return valid JSON only
   - Use double quotes for all keys and string values
   - Do not include markdown fences, comments, ellipses, or trailing commas in the final answer
{longmem_fact_instruction}

{output_format}

[Example]
Dialogues:
[2025-11-15T14:30:00] Alice: Bob, let's meet at Starbucks tomorrow at 2pm to discuss the new product
[2025-11-15T14:31:00] Bob: Okay, I'll prepare the materials

{example_output}

{final_instruction}
"""

    def _parse_llm_response(
        self,
        response: str,
        dialogue_ids: List[int],
        session_spans: Optional[List[tuple]] = None,
    ) -> List[MemoryEntry]:
        """
        Parse LLM response to MemoryEntry list.

        session_spans: ordered list of (session_date_iso, session_end_date_iso)
        covering the window. Each entry's timestamp is matched to the span that
        contains it; if no match (or no timestamp), the first span is used.
        """
        # Extract JSON. Local small/medium LLMs occasionally hit the output
        # token cap after producing many complete entries but before closing
        # the final JSON object. Recover complete entries instead of wasting
        # retries on the same long response shape.
        try:
            data = self.llm_client.extract_json(response)
        except ValueError:
            data = self._recover_partial_memory_entries(response)
            if not data:
                raise
        data = self._normalize_memory_entry_payload(data)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array but got: {type(data)}")

        entries = []
        for item in data:
            item = _normalize_memory_item(item)
            if not item["lossless_restatement"]:
                continue
            timestamp = item.get("timestamp")
            session_date, session_end_date = _match_session_span(timestamp, session_spans)

            entry = MemoryEntry(
                lossless_restatement=item["lossless_restatement"],
                keywords=item.get("keywords", []),
                timestamp=timestamp,
                location=item.get("location"),
                persons=item.get("persons", []),
                entities=item.get("entities", []),
                topic=item.get("topic"),
                session_date=session_date,
                session_end_date=session_end_date,
            )
            entries.append(entry)

        return entries

    def _recover_partial_memory_entries(self, response: str):
        """
        Recover complete memory entry objects from a truncated JSON response.

        This is intentionally narrow: it only scans inside a memory_entries
        array, or a top-level array fallback, and uses json.JSONDecoder so
        braces inside strings are handled correctly.
        """
        if not response:
            return []

        decoder = json.JSONDecoder()
        text = response.strip()
        array_start = -1

        key_pos = text.find('"memory_entries"')
        if key_pos != -1:
            array_start = text.find("[", key_pos)
        if array_start == -1:
            array_start = text.find("[")
        if array_start == -1:
            return []

        entries = []
        pos = array_start + 1
        while pos < len(text):
            while pos < len(text) and text[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(text) or text[pos] == "]":
                break
            if text[pos] != "{":
                pos += 1
                continue

            try:
                item, end = decoder.raw_decode(text[pos:])
            except json.JSONDecodeError:
                break

            if isinstance(item, dict) and "lossless_restatement" in item:
                entries.append(item)
            pos += end

        if entries:
            print(
                f"[MemoryBuilder] Warning: recovered {len(entries)} complete "
                "entries from a truncated JSON response"
            )
        return entries

    def _normalize_memory_entry_payload(self, data):
        """
        Accept a few common LLM wrapper shapes around the memory-entry list.

        The preferred format is still a top-level JSON array, but some models
        occasionally return an object such as {"memory_entries": [...]} or a
        single entry object. Normalize those into the list shape expected by
        the rest of the parser.
        """
        if isinstance(data, list):
            return data

        if not isinstance(data, dict):
            return data

        wrapper_keys = (
            "memory_entries",
            "memory_entry",
            "memory",
            "entries",
            "memories",
            "items",
            "results",
            "data",
        )
        for key in wrapper_keys:
            candidate = data.get(key)
            if isinstance(candidate, list):
                print(
                    f"[MemoryBuilder] Warning: LLM returned object with '{key}' list; "
                    "normalizing to memory-entry array"
                )
                return candidate
            if isinstance(candidate, dict) and "lossless_restatement" in candidate:
                print(
                    f"[MemoryBuilder] Warning: LLM returned object with '{key}' single entry; "
                    "wrapping it in a list"
                )
                return [candidate]

        if "lossless_restatement" in data:
            print(
                "[MemoryBuilder] Warning: LLM returned a single memory-entry object; "
                "wrapping it in a list"
            )
            return [data]

        list_values = [value for value in data.values() if isinstance(value, list)]
        if len(list_values) == 1:
            print(
                "[MemoryBuilder] Warning: LLM returned an object containing a single list value; "
                "using that list as the memory-entry array"
            )
            return list_values[0]

        entry_values = [
            value
            for value in data.values()
            if isinstance(value, dict) and "lossless_restatement" in value
        ]
        if entry_values:
            print(
                "[MemoryBuilder] Warning: LLM returned memory entries as object values; "
                "using those values as the memory-entry array"
            )
            return entry_values

        return data
    
    def _process_windows_parallel(self, windows: List[List[Dialogue]]):
        """
        Process multiple windows in parallel using ThreadPoolExecutor
        """
        all_entries = []
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all window processing tasks
            future_to_window = {}
            for i, window in enumerate(windows):
                dialogue_ids = [d.dialogue_id for d in window]
                future = executor.submit(self._generate_memory_entries_worker, window, dialogue_ids, i+1)
                future_to_window[future] = (window, i+1)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_window):
                window, window_num = future_to_window[future]
                try:
                    entries = future.result()
                    all_entries.extend(entries)
                    print(f"[Parallel Processing] Window {window_num} completed: {len(entries)} entries")
                except Exception as e:
                    print(f"[Parallel Processing] Window {window_num} failed: {e}")
        
        # Store all entries to database in batch
        if all_entries:
            print(f"\n[Parallel Processing] Storing {len(all_entries)} entries to database...")
            self.vector_store.add_entries(all_entries)
            self.processed_count += sum(len(window) for window in windows)
            
            # Update previous entries (use last window's entries for context)
            if all_entries:
                self.previous_entries = all_entries[-10:]  # Keep last 10 entries for context
        
        print(f"[Parallel Processing] Completed processing {len(windows)} windows")
    
    def _generate_memory_entries_worker(self, window: List[Dialogue], dialogue_ids: List[int], window_num: int) -> List[MemoryEntry]:
        """
        Worker function for parallel processing of a single batch (full window or remaining dialogues)
        """
        batch_size = len(window)
        batch_type = "full window" if batch_size == self.window_size else f"remaining batch"
        print(f"[Worker {window_num}] Processing {batch_type} with {batch_size} dialogues")

        # Build per-session spans for this worker's window
        worker_session_spans = _extract_session_spans(window)
        
        # Build dialogue text
        dialogue_text = "\n".join([str(d) for d in window])
        
        # Build context (shared across all workers - this is fine for parallel processing)
        context = ""
        if self.previous_entries:
            context = "\n[Previous Window Memory Entries (for reference to avoid duplication)]\n"
            for entry in self.previous_entries[:3]:  # Only show first 3
                context += f"- {entry.lossless_restatement}\n"

        # Build prompt
        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, context)

        # Call LLM
        messages = [
            {
                "role": "system",
                "content": "You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times if parsing fails
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
                    response_format=response_format,
                    max_tokens=getattr(config, "MEMORY_EXTRACTION_MAX_TOKENS", None),
                )

                # Parse response — stamp correct session span per entry
                entries = self._parse_llm_response(
                    response, dialogue_ids,
                    worker_session_spans
                )
                print(f"[Worker {window_num}] Generated {len(entries)} entries")
                return entries

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Worker {window_num}] Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"[Worker {window_num}] All {max_retries} attempts failed: {e}")
                    return []
