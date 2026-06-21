"""
Vector Store - Multi-View Indexing (Section 3.1)

Implements three-layer indexing I(m_k):
- Semantic Layer: s_k = E_dense(m_k) - Dense vector similarity
- Lexical Layer: l_k = E_sparse(m_k) - BM25 keyword matching (Tantivy FTS)
- Symbolic Layer: r_k = E_sym(m_k) - Metadata filtering (SQL)
"""
from typing import List, Optional, Dict, Any
import lancedb
import pyarrow as pa
from models.memory_entry import MemoryEntry
from utils.embedding import EmbeddingModel
from ramem import config
import os
import re


class VectorStore:
    """
    Multi-View Indexing - Storage and retrieval for memory units (Section 3.1)

    Three-layer indexing I(m_k):
    1. Semantic Layer: Dense embeddings for conceptual similarity
    2. Lexical Layer: Tantivy FTS for exact keyword matching
    3. Symbolic Layer: SQL-based metadata filtering
    """

    def __init__(
        self,
        db_path: str = None,
        embedding_model: EmbeddingModel = None,
        table_name: str = None,
        storage_options: Optional[Dict[str, Any]] = None
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self.table = None
        self._fts_initialized = False

        # Detect if using cloud storage (GCS, S3, Azure)
        self._is_cloud_storage = self.db_path.startswith(("gs://", "s3://", "az://"))

        # Connect to database
        if self._is_cloud_storage:
            self.db = lancedb.connect(self.db_path, storage_options=storage_options)
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)

        self._init_table()

    def _init_table(self):
        """Initialize table schema and FTS index."""
        schema = pa.schema([
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("session_date", pa.string()),
            pa.field("session_end_date", pa.string()),
            pa.field("mention_date", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing table: {self.table_name}")

    def _init_fts_index(self):
        """Initialize Full-Text Search index on lossless_restatement column."""
        if self._fts_initialized:
            return

        try:
            if self._is_cloud_storage:
                # Use native FTS for cloud storage (Tantivy only works with local filesystem)
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=False,
                    replace=True
                )
                print("FTS index created (native mode for cloud storage)")
            else:
                # Use Tantivy FTS for local storage (better performance)
                self.table.create_fts_index(
                    "lossless_restatement",
                    use_tantivy=True,
                    tokenizer_name="en_stem",
                    replace=True
                )
                print("FTS index created (Tantivy mode)")
            self._fts_initialized = True
        except Exception as e:
            print(f"FTS index creation skipped: {e}")

    def _results_to_entries(self, results: List[dict]) -> List[MemoryEntry]:
        """Convert LanceDB results to MemoryEntry objects."""
        entries = []
        for r in results:
            try:
                entries.append(MemoryEntry(
                    entry_id=r["entry_id"],
                    lossless_restatement=r["lossless_restatement"],
                    keywords=list(r.get("keywords") or []),
                    timestamp=r.get("timestamp") or None,
                    location=r.get("location") or None,
                    persons=list(r.get("persons") or []),
                    entities=list(r.get("entities") or []),
                    topic=r.get("topic") or None,
                    session_date=r.get("session_date") or None,
                    session_end_date=r.get("session_end_date") or None,
                    mention_date=r.get("mention_date") or None,
                ))
            except Exception as e:
                print(f"Warning: Failed to parse result: {e}")
                continue
        return entries

    def add_entries(self, entries: List[MemoryEntry]):
        """Batch add memory entries."""
        if not entries:
            return

        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)

        data = []
        for entry, vector in zip(entries, vectors):
            data.append({
                "entry_id": entry.entry_id,
                "lossless_restatement": entry.lossless_restatement,
                "keywords": entry.keywords,
                "timestamp": entry.timestamp or "",
                "location": entry.location or "",
                "persons": entry.persons,
                "entities": entry.entities,
                "topic": entry.topic or "",
                "session_date": entry.session_date or "",
                "session_end_date": entry.session_end_date or "",
                "mention_date": getattr(entry, "mention_date", None) or "",
                "vector": vector.tolist()
            })

        self.table.add(data)
        print(f"Added {len(entries)} memory entries")

        # Initialize FTS index after first data insertion
        if not self._fts_initialized:
            self._init_fts_index()

    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """
        Semantic Layer Search - Dense vector similarity (Section 3.1)
        s_k = E_dense(m_k)
        """
        try:
            if self.table.count_rows() == 0:
                return []

            query_vector = self.embedding_model.encode_single(query, is_query=True)
            results = self.table.search(query_vector.tolist()).limit(top_k).to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

    def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[MemoryEntry]:
        """
        Lexical Layer Search - BM25 keyword matching (Section 3.1)
        l_k = E_sparse(m_k)
        """
        try:
            if not keywords or self.table.count_rows() == 0:
                return []

            # LanceDB/Tantivy treats punctuation such as apostrophes and colons
            # as query syntax. Clean natural-language keywords into plain terms.
            query = self._sanitize_fts_query(" ".join(keywords))
            if not query:
                return []
            results = self.table.search(query).limit(top_k).to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during keyword search: {e}")
            return []

    def keyword_search_with_filter(
        self,
        keywords: List[str],
        where_clause: str,
        top_k: int = 25
    ) -> List[MemoryEntry]:
        """
        BM25 keyword search pre-filtered by a metadata condition (e.g. session overlap).
        Mirrors keyword_search() exactly, but adds a SQL-style where prefilter.
        Used for SSS-BM25 comparison in debug_gt_rank.py.
        """
        try:
            if not keywords or self.table.count_rows() == 0:
                return []
            query   = self._sanitize_fts_query(" ".join(keywords))
            if not query:
                return []
            results = (
                self.table
                .search(query)
                .where(where_clause, prefilter=True)
                .limit(top_k)
                .to_list()
            )
            return self._results_to_entries(results)
        except Exception as e:
            print(f"Error during filtered keyword search: {e}")
            return []

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        query = re.sub(r"[^A-Za-z0-9_\s-]+", " ", str(query))
        query = re.sub(r"\s+", " ", query).strip()
        return query

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        session_overlap_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Symbolic Layer Search - Metadata filtering (Section 3.1)
        r_k = E_sym(m_k), filters by timestamps, entities, persons

        session_overlap_range: (start_iso, end_iso) — matches any entry whose
        session span [session_date, session_end_date] overlaps the query window.
        Preferred over timestamp_range for temporal questions.
        """
        try:
            if self.table.count_rows() == 0:
                return []

            if not any([persons, timestamp_range, session_overlap_range, location, entities]):
                return []

            conditions = []

            if persons:
                values = ", ".join([f"'{p}'" for p in persons])
                conditions.append(f"array_has_any(persons, make_array({values}))")

            if location:
                safe_location = location.replace("'", "''")
                conditions.append(f"location LIKE '%{safe_location}%'")

            if entities:
                values = ", ".join([f"'{e}'" for e in entities])
                conditions.append(f"array_has_any(entities, make_array({values}))")

            if session_overlap_range:
                # Overlap: session starts before range ends AND session ends after range starts
                start_time, end_time = session_overlap_range
                conditions.append(
                    f"session_date <= '{end_time}' AND session_end_date >= '{start_time}'"
                )
            elif timestamp_range:
                start_time, end_time = timestamp_range
                conditions.append(f"timestamp >= '{start_time}' AND timestamp <= '{end_time}'")

            where_clause = " AND ".join(conditions)
            query = self.table.search().where(where_clause, prefilter=True)

            if top_k:
                query = query.limit(top_k)

            results = query.to_list()
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during structured search: {e}")
            return []

    def semantic_search_with_filter(
        self,
        query: str,
        where_clause: str,
        top_k: int = 5
    ) -> List[MemoryEntry]:
        """
        Semantic search pre-filtered by a metadata condition.
        Combines vector similarity with a SQL-style where clause in one
        LanceDB call — used for temporal overlap + semantic ranking.
        """
        try:
            if self.table.count_rows() == 0:
                return []

            query_vector = self.embedding_model.encode_single(query, is_query=True)
            results = (
                self.table
                .search(query_vector.tolist())
                .where(where_clause, prefilter=True)
                .limit(top_k)
                .to_list()
            )
            return self._results_to_entries(results)

        except Exception as e:
            print(f"Error during filtered semantic search: {e}")
            return []

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all memory entries."""
        results = self.table.to_arrow().to_pylist()
        return self._results_to_entries(results)

    def optimize(self):
        """Optimize table after bulk insertions for better query performance."""
        self.table.optimize()
        print("Table optimized")

    def clear(self):
        """Clear all data and reinitialize table."""
        self.db.drop_table(self.table_name)
        self._fts_initialized = False
        self._init_table()
        print("Database cleared")
