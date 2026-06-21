"""
Embedding utilities - Generate vector embeddings using SentenceTransformers
Supports Qwen3 Embedding models through SentenceTransformers interface
"""
from pathlib import Path
from typing import List, Optional
import numpy as np
from ramem import config
import os


class EmbeddingModel:
    """
    Embedding model using SentenceTransformers (supports Qwen3 and other models)
    """
    def __init__(self, model_name: str = None, use_optimization: bool = True):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_optimization = use_optimization
        self.device = os.environ.get("EMBEDDING_DEVICE", "auto").strip().lower()
        self.batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "8"))
        self.resolved_model_path = self._resolve_local_model_path(self.model_name)
        
        print(f"Loading embedding model: {self.model_name}")
        print(f"Embedding device: {self.device}")
        print(f"Embedding batch size: {self.batch_size}")
        if self.resolved_model_path:
            print(f"Resolved local embedding model path: {self.resolved_model_path}")
        
        # Check if it's a Qwen3 model (through SentenceTransformers)
        if self._is_qwen3_model(self.model_name):
            self._init_qwen3_sentence_transformer()
        else:
            self._init_standard_sentence_transformer()

    def _is_qwen3_model(self, model_name: str) -> bool:
        normalized = (model_name or "").lower()
        return normalized.startswith("qwen3") or "qwen3-embedding" in normalized

    def _get_model_load_target(self) -> str:
        return self.resolved_model_path or self.model_name

    def _resolve_local_model_path(self, model_name: str) -> Optional[str]:
        """
        Resolve a local snapshot path for offline Hugging Face usage.

        Supports:
        - direct local directory paths
        - EMBEDDING_MODEL_PATH env override
        - Hugging Face cache layouts under HF_HUB_CACHE / HF_HOME / defaults
        """
        direct_path = self._normalize_model_dir(Path(os.path.expandvars(os.path.expanduser(model_name))))
        if direct_path:
            return str(direct_path)

        override = os.environ.get("EMBEDDING_MODEL_PATH")
        if override:
            override_path = self._normalize_model_dir(
                Path(os.path.expandvars(os.path.expanduser(override)))
            )
            if override_path:
                print("Using EMBEDDING_MODEL_PATH override for embedding model")
                return str(override_path)
            print(f"Warning: EMBEDDING_MODEL_PATH does not exist or is invalid: {override}")

        if "/" not in model_name:
            return None

        repo_cache_dir = f"models--{model_name.replace('/', '--')}"
        searched_paths = []

        for cache_root in self._candidate_hf_cache_roots():
            repo_dir = cache_root / repo_cache_dir
            searched_paths.append(str(repo_dir))
            local_snapshot = self._normalize_model_dir(repo_dir)
            if local_snapshot:
                return str(local_snapshot)

        if searched_paths:
            print("No local embedding snapshot found in:")
            for path in searched_paths:
                print(f"  - {path}")

        return None

    def _candidate_hf_cache_roots(self) -> List[Path]:
        roots: List[Path] = []

        def add_candidate(raw_path: Optional[str], append_hub: bool = False) -> None:
            if not raw_path:
                return
            base = Path(os.path.expandvars(os.path.expanduser(raw_path)))
            candidates = [base / "hub"] if append_hub else [base]
            for candidate in candidates:
                if candidate not in roots:
                    roots.append(candidate)

        add_candidate(os.environ.get("HF_HUB_CACHE"))
        add_candidate(os.environ.get("HUGGINGFACE_HUB_CACHE"))
        add_candidate(os.environ.get("HF_HOME"), append_hub=True)
        add_candidate("~/.cache/huggingface", append_hub=True)

        return roots

    def _normalize_model_dir(self, path: Path) -> Optional[Path]:
        """
        Convert either a snapshot path or a Hugging Face repo cache directory
        into the actual model directory SentenceTransformer should load from.
        """
        if not path.is_dir():
            return None

        if (path / "config.json").exists():
            return path

        snapshots_dir = path / "snapshots"
        if not snapshots_dir.is_dir():
            return None

        ref_path = path / "refs" / "main"
        if ref_path.is_file():
            snapshot_name = ref_path.read_text().strip()
            snapshot_path = snapshots_dir / snapshot_name
            if (snapshot_path / "config.json").exists():
                return snapshot_path

        snapshots = sorted(
            [candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir()],
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if (snapshot / "config.json").exists():
                return snapshot

        return None

    def _init_qwen3_sentence_transformer(self):
        """Initialize Qwen3 model using SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Map model names to actual model paths
            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B", 
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
            }
            
            model_path = self._get_model_load_target()
            if not self.resolved_model_path:
                model_path = qwen3_models.get(self.model_name, self.model_name)
            print(f"Loading Qwen3 model via SentenceTransformers: {model_path}")

            if self.device == "cpu":
                self.model = SentenceTransformer(
                    model_path,
                    trust_remote_code=True,
                    device="cpu"
                )
                print("Qwen3 loaded on CPU for embedding generation")
            # Initialize with optimization settings
            elif self.use_optimization:
                try:
                    # Try to use flash_attention_2 and left padding for better performance
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2", 
                            "device_map": "auto"
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True
                    )
                    print("Qwen3 loaded with flash_attention_2 optimization")
                except Exception as e:
                    print(f"Flash attention failed ({e}), using standard loading...")
                    self.model = SentenceTransformer(
                        model_path,
                        trust_remote_code=True,
                        device=None if self.device == "auto" else self.device
                    )
            else:
                self.model = SentenceTransformer(
                    model_path,
                    trust_remote_code=True,
                    device=None if self.device == "auto" else self.device
                )
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3_sentence_transformer"
            
            # Check if Qwen3 supports query prompts
            self.supports_query_prompt = hasattr(self.model, 'prompts') and 'query' in getattr(self.model, 'prompts', {})
            
            print(f"Qwen3 model loaded successfully with dimension: {self.dimension}")
            if self.supports_query_prompt:
                print("Query prompt support detected")
                
        except Exception as e:
            print(f"Failed to load Qwen3 model: {e}")
            print("Falling back to default SentenceTransformers model...")
            self._fallback_to_sentence_transformer()

    def _init_standard_sentence_transformer(self):
        """Initialize standard SentenceTransformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                self._get_model_load_target(),
                trust_remote_code=True,
                device=None if self.device == "auto" else self.device
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "sentence_transformer"
            self.supports_query_prompt = False
            print(f"SentenceTransformer model loaded with dimension: {self.dimension}")
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _fallback_to_sentence_transformer(self):
        """Fallback to default SentenceTransformer model"""
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using fallback model: {fallback_model}")
        self.model_name = fallback_model
        self._init_standard_sentence_transformer()

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to vectors
        
        Args:
        - texts: List of texts to encode
        - is_query: Whether these are query texts (for Qwen3 prompt optimization)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Use query prompt for Qwen3 models when encoding queries
        if self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self._encode_with_query_prompt(texts)
        else:
            return self._encode_standard(texts)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode single text
        
        Args:
        - text: Text to encode
        - is_query: Whether this is a query text (for Qwen3 prompt optimization)
        """
        return self.encode([text], is_query=is_query)[0]
    
    def encode_query(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries with optimal settings for Qwen3
        """
        return self.encode(queries, is_query=True)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents (no query prompt)
        """
        return self.encode(documents, is_query=False)
    
    def _encode_with_query_prompt(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3 query prompt"""
        try:
            embeddings = self.model.encode(
                texts, 
                prompt_name="query",  # Use Qwen3's query prompt
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            print(f"Query prompt encoding failed: {e}, falling back to standard encoding")
            return self._encode_standard(texts)
    
    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        """Encode texts using standard method"""
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings
