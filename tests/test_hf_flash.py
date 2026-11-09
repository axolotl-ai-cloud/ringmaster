"""Ring/Ulysses flash lookups reuse the kernel transformers loaded for the model instead of
loading the hub kernel again, which re-checks publisher trust against the Hub API."""

import sys
import types

import pytest

fau = pytest.importorskip("transformers.modeling_flash_attention_utils")
hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")

from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

from ringmaster import hf_flash  # noqa: E402
from ringmaster.ring import kernels, varlen_blocks  # noqa: E402
from ringmaster.strategies import ulysses  # noqa: E402

FA2 = "flash_attention_2"
HUB_FA2 = fau.FLASH_ATTN_KERNEL_FALLBACK[FA2]
HUB_FA3 = fau.FLASH_ATTN_KERNEL_FALLBACK["flash_attention_3"]
_FAU_STATE = (
    "_loaded_implementation",
    "_flash_fn",
    "_flash_varlen_fn",
    "_flash_with_kvcache_fn",
    "_pad_fn",
    "_unpad_fn",
    "_process_flash_kwargs_fn",
    "_process_with_kvcache_kwargs_fn",
)


def _fake_kernel(name):
    mod = types.ModuleType(name)

    def flash_attn_func(*args, **kwargs):
        raise AssertionError("not called")

    def flash_attn_varlen_func(*args, **kwargs):
        raise AssertionError("not called")

    for fn in (flash_attn_func, flash_attn_varlen_func):
        fn.__module__ = name
        setattr(mod, fn.__name__, fn)
    mod._flash_attn_varlen_forward = object()
    mod._flash_attn_varlen_backward = object()
    return mod


@pytest.fixture
def hub(monkeypatch):
    """No flash_attn package; every hub-kernel load is recorded instead of fetched."""
    loads = []

    def get_kernel(repo_id, **kwargs):
        loads.append(repo_id)
        mod = _fake_kernel(f"_ringmaster_test_kernel_{len(loads)}")
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
        return mod

    monkeypatch.setattr(hub_kernels, "get_kernel", get_kernel)
    monkeypatch.setattr(hub_kernels, "is_kernels_available", lambda: True)
    for name in (
        "is_flash_attn_2_available",
        "is_flash_attn_3_available",
        "is_flash_attn_4_available",
        "is_torch_npu_available",
    ):
        monkeypatch.setattr(fau, name, lambda: False)
    for name in _FAU_STATE:
        monkeypatch.setattr(fau, name, None, raising=False)
    for registry in (ALL_ATTENTION_FUNCTIONS, ALL_MASK_ATTENTION_FUNCTIONS):
        cls = type(registry)
        monkeypatch.setattr(cls, "_global_mapping", dict(cls._global_mapping))
    hf_flash._lazy_flash_attn_funcs.cache_clear()
    yield loads
    hf_flash._lazy_flash_attn_funcs.cache_clear()


def test_reuses_the_kernel_transformers_loaded(hub):
    # What from_pretrained(attn_implementation="flash_attention_2") does without flash_attn.
    fau.lazy_import_flash_attention(HUB_FA2)
    assert hub == [HUB_FA2]
    kernel = sys.modules[fau._flash_fn.__module__]

    for name in (FA2, HUB_FA2):  # Ulysses passes the inner name, the ring loop the model's
        assert ulysses._flash_varlen_fn(name) is kernel.flash_attn_varlen_func
    assert kernels._flash_fn(FA2) is kernel.flash_attn_func
    assert varlen_blocks._flash_varlen_ops(FA2) == (
        kernel._flash_attn_varlen_forward,
        kernel._flash_attn_varlen_backward,
    )
    assert hub == [HUB_FA2]
    assert fau._loaded_implementation == HUB_FA2


def test_loads_once_itself_when_transformers_has_not(hub):
    first = ulysses._flash_varlen_fn(FA2)
    assert ulysses._flash_varlen_fn(FA2) is first
    assert kernels._flash_fn(FA2).__module__ == first.__module__
    assert hub == [HUB_FA2]
    assert fau._loaded_implementation is None


def test_other_loaded_implementation_is_not_reused(hub):
    fau.lazy_import_flash_attention(HUB_FA3)
    fa3 = fau._flash_varlen_fn
    assert ulysses._flash_varlen_fn(FA2) is not fa3
    assert hub == [HUB_FA3, HUB_FA2]
    assert fau._flash_varlen_fn is fa3
