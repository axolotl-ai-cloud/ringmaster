"""Ring attention: our own loop, pluggable block kernel.

The ring algorithm (shard sequence, attend each query block to every causal KV
block, merge via online softmax) is kernel-independent. We own the loop and make
the per-block kernel swappable, so Ring is not limited to torch SDPA/flex.
"""

from ringmaster.ring.loop import ring_attention

__all__ = ["ring_attention"]
