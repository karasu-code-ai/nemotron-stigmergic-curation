r"""Compatibility shim so NVIDIA-Nemotron-3-Nano-30B (Nemotron-H hybrid) loads on
this ARM64 + Blackwell (sm_121) Spark WITHOUT building mamba-ssm / causal-conv1d.

Why this exists
---------------
`modeling_nemotron_h.py` (vendored via trust_remote_code) has exactly ONE hard,
unconditional dependency:

 from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn # raises if missing

Every other CUDA kernel it uses (selective_state_update, mamba_chunk_scan_combined,
mamba_split_conv1d_scan_combined, causal_conv1d_fn/_update) is already optional:
when absent, `is_fast_path_available` is False and the model runs its pure-PyTorch
`torch_forward` slow path. mamba-ssm has NO aarch64 wheel, so installing it means a
fragile from-source CUDA build on sm_121. Instead we provide JUST `rmsnorm_fn` as a
faithful pure-PyTorch function, injected into sys.modules BEFORE the model is
imported.

Key subtlety: we do NOT make `mamba_ssm` look "installed" to transformers.
`transformers.utils.import_utils.is_mamba_2_ssm_available` checks package
*metadata* (importlib.metadata), which a sys.modules injection does not provide, so
it stays False -> the fast-path kernels remain None -> torch_forward is used ->
only our rmsnorm_fn is ever called (via MambaRMSNormGated). We verify that
invariant in `apply` and refuse to proceed if it ever flips True (which would
mean the model tries to import kernels we didn't supply).

rmsnorm_fn contract (from mamba_ssm, as called by MambaRMSNormGated):
 rmsnorm_fn(x, weight, bias=None, z=gate, eps, group_size, norm_before_gate=False)
 with norm_before_gate=False and z given:
 x = x * silu(z) # gate first
 then group-wise RMSNorm over the last dim in chunks of `group_size`:
 x = x / sqrt(mean(x**2, group) + eps) * weight (+ bias)
 Computation upcasts to fp32 then casts back to the input dtype (matches the
 mamba-ssm kernel's upcast=True default and the "converted back to float32" note
 in MambaRMSNormGated).
"""
from __future__ import annotations
import sys
import types


def _make_rmsnorm_fn:
 import torch
 import torch.nn.functional as F

 def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-6, group_size=None,
 norm_before_gate=False, upcast=True):
 in_dtype = x.dtype
 if upcast:
 x = x.float
 weight = weight.float
 if bias is not None:
 bias = bias.float
 if z is not None:
 z = z.float

 # Gating. norm_before_gate=False -> gate BEFORE normalizing (Nemotron-H
 # uses this). norm_before_gate=True -> normalize, then gate.
 if z is not None and not norm_before_gate:
 x = x * F.silu(z)

 if group_size is None:
 var = x.pow(2).mean(dim=-1, keepdim=True)
 out = x * torch.rsqrt(var + eps)
 else:
 shape = x.shape
 d = shape[-1]
 assert d % group_size == 0, f"last dim {d} not divisible by group_size {group_size}"
 xg = x.reshape(*shape[:-1], d // group_size, group_size)
 var = xg.pow(2).mean(dim=-1, keepdim=True)
 xg = xg * torch.rsqrt(var + eps)
 out = xg.reshape(shape)

 out = out * weight
 if bias is not None:
 out = out + bias

 if z is not None and norm_before_gate:
 out = out * F.silu(z)

 return out.to(in_dtype)

 return rmsnorm_fn


def apply -> None:
 """Inject the fake mamba_ssm.ops.triton.layernorm_gated module providing
 rmsnorm_fn. Idempotent. Call once before loading the model."""
 # Don't clobber a real install.
 try:
 from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn # noqa: F401
 return # real mamba-ssm present; nothing to do
 except Exception:
 pass

 rmsnorm_fn = _make_rmsnorm_fn

 import importlib.machinery

 def _mod(name):
 m = sys.modules.get(name)
 if m is None:
 m = types.ModuleType(name)
 m.__path__ = [] # mark as package so submodule imports resolve
 # A real __spec__ is required: transformers' _is_package_available
 # calls importlib.util.find_spec(name), which raises ValueError if
 # the module exists in sys.modules with __spec__ is None.
 spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
 m.__spec__ = spec
 m.__loader__ = None
 sys.modules[name] = m
 return m

 _mod("mamba_ssm")
 _mod("mamba_ssm.ops")
 _mod("mamba_ssm.ops.triton")
 lng = _mod("mamba_ssm.ops.triton.layernorm_gated")
 lng.rmsnorm_fn = rmsnorm_fn

 # Force the fast-path detectors to return False. The model's modeling file
 # does `if is_mamba_2_ssm_available: from mamba_ssm... import <fast kernels>`
 # UNGUARDED at module import time. We only shim rmsnorm_fn, NOT those kernels,
 # so the detector must say "unavailable" -> model uses the pure-torch
 # torch_forward path. We patch it to a hard False (rather than relying on the
 # metadata probe, which raises InvalidVersion on our spec-only fake module).
 # Patch BEFORE the modeling file's `from ... import is_mamba_2_ssm_available`
 # binds, so it picks up our version. Also clear lru_cache if any.
 import transformers.utils.import_utils as _iu
 for _name in ("is_mamba_2_ssm_available", "is_causal_conv1d_available",
 "is_mamba_ssm_available"):
 _fn = getattr(_iu, _name, None)
 if _fn is not None:
 try:
 if hasattr(_fn, "cache_clear"):
 _fn.cache_clear
 except Exception:
 pass
 setattr(_iu, _name, lambda *a, **k: False)


if __name__ == "__main__":
 apply
 # smoke-test rmsnorm_fn against a direct reference
 import torch
 import torch.nn.functional as F
 from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
 torch.manual_seed(0)
 x = torch.randn(2, 8, 64)
 g = torch.randn(2, 8, 64)
 w = torch.randn(64)
 gs = 32
 out = rmsnorm_fn(x=x, weight=w, bias=None, z=g, eps=1e-5, group_size=gs,
 norm_before_gate=False)
 # reference
 xr = (x.float * F.silu(g.float))
 xr = xr.reshape(2, 8, 64 // gs, gs)
 xr = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-5)
 xr = xr.reshape(2, 8, 64) * w.float
 assert torch.allclose(out, xr.to(x.dtype), atol=1e-5), (out - xr).abs.max
 print("nemotron_compat: rmsnorm_fn shim OK; is_mamba_2_ssm_available stays False")
