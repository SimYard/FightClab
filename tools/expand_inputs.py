"""Widen a checkpoint's actor and critic by N inputs with zero weights.

    uv run python tools/expand_inputs.py CHECKPOINT N OUT.pt

The policy acts and values as before until training uses them. For a task that gained a last observation term
(``k1/duel.py``'s SEEN) and a run resumed from OUT.pt. The new inputs' normalizer starts at mean 0, std 1 and, after
the checkpoint's billions of samples, stays there: the term itself keeps them near 1. Adam's moments for the new
weights start at zero, so their first few updates run a few times the learning rate.
"""

import sys
from pathlib import Path

import torch

checkpoint, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
widened = []
for net in ("actor_state_dict", "critic_state_dict"):
    state = saved[net]
    widened.append(state["mlp.0.weight"].shape)
    state["mlp.0.weight"] = torch.cat((state["mlp.0.weight"], state["mlp.0.weight"].new_zeros(widened[-1][0], n)), 1)
    for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        stat = state[f"obs_normalizer.{key}"]
        state[f"obs_normalizer.{key}"] = torch.cat((stat, stat.new_full((1, n), fill)), 1)
moments = [s for s in saved["optimizer_state_dict"]["state"].values() if s["exp_avg"].shape in widened]
assert len(moments) == len(widened), "Adam's moments of the input layers not found by their shape"
for s in moments:
    for key in ("exp_avg", "exp_avg_sq"):
        s[key] = torch.cat((s[key], s[key].new_zeros(s[key].shape[0], n)), 1)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(saved, out)
print(f"{checkpoint} -> {out}: inputs {[s[1] for s in widened]} + {n}")
