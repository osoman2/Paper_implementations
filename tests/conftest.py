"""
Set SMOKE_FAKE_EMBEDDINGS=1 to replace SentenceTransformer with a deterministic stub.
Use it where huggingface.co is unreachable (agent sandboxes). CI runs with the real model.
"""

import os

import numpy as np

# The local Gradio server must never be reached through an HTTP(S) proxy.
for _var in ("NO_PROXY", "no_proxy"):
    os.environ[_var] = ",".join(filter(None, [os.environ.get(_var, ""), "127.0.0.1", "localhost"]))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

if os.environ.get("SMOKE_FAKE_EMBEDDINGS") == "1":
    import sentence_transformers

    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.dim = 384

        def get_sentence_embedding_dimension(self):
            return self.dim

        def encode(self, sentences, *args, normalize_embeddings=False, convert_to_numpy=True, **kwargs):
            single = isinstance(sentences, str)
            items = [sentences] if single else list(sentences)
            out = np.stack([
                np.random.default_rng(abs(hash(s)) % (2**32)).standard_normal(self.dim).astype("float32")
                for s in items
            ]) if items else np.zeros((0, self.dim), dtype="float32")
            if normalize_embeddings and len(out):
                out /= np.linalg.norm(out, axis=1, keepdims=True)
            return out[0] if single else out

    sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
