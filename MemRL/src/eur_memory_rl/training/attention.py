"""Explicit HF attention kernels, preserving the model and policy distribution."""
from contextlib import contextmanager


@contextmanager
def hf_attention_backend(name="efficient"):
    import transformers.integrations.sdpa_attention as sdpa
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if name not in {"efficient", "flash-gqa"}:
        raise ValueError("Unknown HF attention backend")
    original = sdpa.use_gqa_in_sdpa
    try:
        # Efficient SDPA cannot handle grouped heads without explicit repetition.
        # Flash SDPA retains the native guarded GQA path (no custom attention math).
        if name == "efficient":
            sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION if name == "efficient"
                         else SDPBackend.FLASH_ATTENTION):
            yield
    finally:
        sdpa.use_gqa_in_sdpa = original
