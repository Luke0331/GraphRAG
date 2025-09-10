"""
Comprehensive embedding solution for RAG project
Handles Qwen3 embedding model issues and provides fallbacks
"""

import os
import sys
import torch
from typing import List, Optional, Any
from pydantic import Field

# Try to import llama_index embeddings
try:
    from llama_index.core.embeddings import BaseEmbedding
    LLAMA_INDEX_AVAILABLE = True
    print("✓ Llama-index embeddings available")
except ImportError as e:
    LLAMA_INDEX_AVAILABLE = False
    print(f"✗ Llama-index embeddings not available: {e}")

# Try to import transformers
try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    import transformers
    TRANSFORMERS_AVAILABLE = True
    print(f"✓ Transformers version: {transformers.__version__}")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("✗ Transformers not available")

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✓ Sentence-transformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("✗ Sentence-transformers not available")

class Qwen3Embedding(BaseEmbedding if LLAMA_INDEX_AVAILABLE else object):
    """Qwen3 embedding model wrapper with error handling"""
    
    model_path: str = Field()
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)

    def __init__(self, model_path: str):
        if LLAMA_INDEX_AVAILABLE:
            super().__init__(model_path=model_path)
        object.__setattr__(self, "tokenizer", None)
        object.__setattr__(self, "model", None)
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen3 model with error handling"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        
        print(f"Loading Qwen3 model from: {self.model_path}")
        
        try:
            # Method 1: Try with trust_remote_code=True
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                padding_side='right'
            )
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.float32
            ).eval()
            print("✓ Qwen3 model loaded successfully (Method 1)")
            
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            
            try:
                # Method 2: Try with config first
                config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    self.model_path, 
                    config=config,
                    trust_remote_code=True, 
                    torch_dtype=torch.float32
                ).eval()
                print("✓ Qwen3 model loaded successfully (Method 2)")
                
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                
                try:
                    # Method 3: Try without trust_remote_code
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.model = AutoModel.from_pretrained(
                        self.model_path, 
                        torch_dtype=torch.float32
                    ).eval()
                    print("✓ Qwen3 model loaded successfully (Method 3)")
                    
                except Exception as e3:
                    print(f"Method 3 failed: {e3}")
                    raise RuntimeError(f"All methods failed to load Qwen3 model: {e3}")
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for input text"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                self.model = self.model.cuda()
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                embedding = hidden_states.mean(dim=1).squeeze().cpu().tolist()
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    # Llama-index compatibility methods
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding for llama-index compatibility"""
        return self.get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding for llama-index compatibility"""
        return self.get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings for llama-index compatibility"""
        return [self.get_text_embedding(text) for text in texts]

    # Async methods for newer llama-index versions
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously"""
        return self.get_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously"""
        return self.get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings asynchronously"""
        return [self.get_text_embedding(text) for text in texts]

class SentenceTransformerEmbedding(BaseEmbedding if LLAMA_INDEX_AVAILABLE else object):
    """Sentence transformer embedding as fallback"""
    
    model: Any = Field(default=None)

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if LLAMA_INDEX_AVAILABLE:
            super().__init__(model=None)
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence-transformers not available")
        object.__setattr__(self, "model", None)
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✓ SentenceTransformer loaded: {model_name}")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            try:
                self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                print("✓ SentenceTransformer loaded: paraphrase-MiniLM-L6-v2")
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                raise
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding using sentence transformers"""
        return self.model.encode(text).tolist()

    # Llama-index compatibility methods
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding for llama-index compatibility"""
        return self.get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding for llama-index compatibility"""
        return self.get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings for llama-index compatibility"""
        return self.model.encode(texts).tolist()

    # Async methods for newer llama-index versions
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously"""
        return self.get_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously"""
        return self.get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings asynchronously"""
        return self.model.encode(texts).tolist()

class DummyEmbedding(BaseEmbedding if LLAMA_INDEX_AVAILABLE else object):
    """Dummy embedding for testing purposes"""
    
    dimension: int = Field(default=768)

    def __init__(self, dimension: int = 768):
        if LLAMA_INDEX_AVAILABLE:
            super().__init__(dimension=dimension)
        object.__setattr__(self, "dimension", dimension)
        print(f"✓ Dummy embedding initialized with dimension {dimension}")
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate dummy embedding (zeros)"""
        import random
        # Generate random embedding for testing
        return [random.uniform(-1, 1) for _ in range(self.dimension)]

    # Llama-index compatibility methods
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding for llama-index compatibility"""
        return self.get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding for llama-index compatibility"""
        return self.get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings for llama-index compatibility"""
        return [self.get_text_embedding(text) for text in texts]

    # Async methods for newer llama-index versions
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding asynchronously"""
        return self.get_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding asynchronously"""
        return self.get_text_embedding(text)
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get multiple text embeddings asynchronously"""
        return [self.get_text_embedding(text) for text in texts]

def create_embedding_model(model_path: Optional[str] = None, fallback: bool = True):
    """
    Create embedding model with automatic fallback
    
    Args:
        model_path: Path to Qwen3 model (optional)
        fallback: Whether to use fallback models if Qwen3 fails
    
    Returns:
        Embedding model instance
    """
    
    # Try Qwen3 first if path is provided
    if model_path:
        try:
            return Qwen3Embedding(model_path)
        except Exception as e:
            print(f"Qwen3 failed: {e}")
            if not fallback:
                raise
    
    # Try sentence-transformers
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            return SentenceTransformerEmbedding()
        except Exception as e:
            print(f"Sentence-transformers failed: {e}")
            if not fallback:
                raise
    
    # Use dummy embedding as last resort
    print("Using dummy embedding as fallback")
    return DummyEmbedding()

# Test function
def test_embedding():
    """Test the embedding functionality"""
    print("="*60)
    print("EMBEDDING MODEL TEST")
    print("="*60)
    
    # Test Qwen3 path
    qwen3_path = r"F:\Intern\EDF\EmbeddingModels\Qwen3-Embedding-0.6B"
    
    print(f"Qwen3 path: {qwen3_path}")
    print(f"Path exists: {os.path.exists(qwen3_path)}")
    
    # Create embedding model
    try:
        embed_model = create_embedding_model(qwen3_path, fallback=True)
        print(f"✓ Embedding model created: {type(embed_model).__name__}")
        
        # Test embedding generation
        test_text = "Silicon battery technology is promising."
        embedding = embed_model.get_text_embedding(test_text)
        print(f"✓ Embedding generated successfully")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        return embed_model
        
    except Exception as e:
        print(f"✗ Failed to create embedding model: {e}")
        return None

if __name__ == "__main__":
    # Test the embedding solution
    embed_model = test_embedding()
    
    if embed_model:
        print("\n" + "="*60)
        print("SUCCESS: Embedding model is ready to use!")
        print("="*60)
        
        # Example usage with llama-index
        try:
            from llama_index.core import Settings
            Settings.embed_model = embed_model
            print("✓ Embedding model set as default for llama-index")
        except ImportError:
            print("llama-index not available, but embedding model works")
    else:
        print("\n" + "="*60)
        print("FAILED: Could not create embedding model")
        print("Please check your dependencies and model path")
        print("="*60) 