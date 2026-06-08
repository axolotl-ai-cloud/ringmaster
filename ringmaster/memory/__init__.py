"""Memory optimizations are deferred to the host — ringmaster doesn't own them.

ALST-style techniques are orthogonal to the distribution backend; axolotl provides
them: tiled MLP (monkeypatch.tiled_mlp), tiled loss (cut_cross_entropy), activation
checkpoint/offload (gradient-checkpointing config). The Ulysses/Ring/USP backends
compose with all of them unchanged. Standalone hosts enable their own (e.g.
``torch.autograd.graph.save_on_cpu``).
"""
