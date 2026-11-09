"""Flash attention functions resolved through transformers.

``_lazy_imports`` caches nothing: without the ``flash_attn`` package each call loads the
hub-kernel fallback (e.g. kernels-community/flash-attn2) again, and ``kernels`` checks the
publisher's trust status against the Hub API on every load. transformers keeps what it
resolved for the model in ``lazy_import_flash_attention``'s module globals, so reuse that.
"""

from __future__ import annotations

from functools import lru_cache


def flash_attn_funcs(attn_implementation: str):
    """``(flash_attn_func, flash_attn_varlen_func)`` for ``attn_implementation``."""
    loaded = loaded_flash_attn_funcs(attn_implementation)
    return loaded if loaded is not None else _lazy_flash_attn_funcs(attn_implementation)


def loaded_flash_attn_funcs(attn_implementation: str):
    """The functions transformers loaded for ``attn_implementation`` or its hub-kernel
    fallback, or None when it loaded another implementation or none."""
    from transformers import modeling_flash_attention_utils as fau

    loaded = getattr(fau, "_loaded_implementation", None)
    fallback = getattr(fau, "FLASH_ATTN_KERNEL_FALLBACK", {}).get(attn_implementation)
    if loaded is None or loaded not in (attn_implementation, fallback):
        return None
    # Asking for the implementation already loaded returns transformers' cache untouched.
    funcs, _ = fau.lazy_import_flash_attention(loaded)
    fn, varlen_fn = funcs[0], funcs[1]
    # Before 5.23 transformers marks an implementation loaded even when its import failed.
    return None if fn is None and varlen_fn is None else (fn, varlen_fn)


@lru_cache(maxsize=4)
def _lazy_flash_attn_funcs(attn_implementation: str):
    from transformers.modeling_flash_attention_utils import _lazy_imports

    fn, varlen_fn, *_ = _lazy_imports(attn_implementation)
    return fn, varlen_fn
