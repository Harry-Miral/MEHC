"""
SPAttention_GappedBands.py - Ablation Experiment C (Incompletely Covered Structured Bands)
Deliberately create information blind spots to verify the necessity of complete coverage
"""

import torch
import time
import gc
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from enum import Enum
from typing import Dict, Optional


class LengthMode(Enum):
    """Length mode enumeration"""
    SHORT = "1-1024"  # 512 to 1024
    LONG = "1-4096"  # 1024 to 4096


class SPAttentionCache:
    """Sparse attention mask cache manager - Configuration 4: Incompletely covered structured bands"""

    def __init__(self, n_heads: int, length_mode: LengthMode, device: str = 'cuda'):
        """
        Initialize cache manager

        Args:
            n_heads: Number of attention heads, must be 8 or 16
            length_mode: Length mode, SHORT(256-1024) or LONG(256-4096)
            device: Device type
        """
        assert n_heads in [8, 16], f"n_heads must be 8 or 16, got {n_heads}"
        assert torch.cuda.is_available() if device == 'cuda' else True

        self.n_heads = n_heads
        self.length_mode = length_mode
        self.device = device
        self.masks: Dict[int, torch.Tensor] = {}

        # Set length range
        if length_mode == LengthMode.SHORT:
            self.min_len, self.max_len = 1, 1024
        else:
            self.min_len, self.max_len = 1, 4096

        print(f"Initializing SPAttention cache [Configuration 4 - Incomplete Coverage Bands]: {n_heads} heads, {length_mode.value} length range")

    def _create_balanced_mask(self, seq_len: int) -> torch.Tensor:
        """Create band mask with information blind spots (Ablation Experiment C)"""
        # Calculate total region width each head is responsible for (including computed area and blind spots)
        base_region_width = seq_len // self.n_heads
        extra_tokens = seq_len % self.n_heads

        def mask_fn(b, h, i, j):
            # 1. Causal condition, remains unchanged
            causal = j <= i

            # 2. Calculate distance for current token pair
            distance = i - j

            # 3. Calculate total region width that current head h should be responsible for
            # First extra_tokens heads have width base_region_width + 1, others have base_region_width
            head_region_width = torch.where(
                h < extra_tokens,
                base_region_width + 1,
                base_region_width
            )

            # 4. Calculate starting distance of region responsible by current head h
            # How many of the first h heads need +1
            extra_before = torch.minimum(h, torch.tensor(extra_tokens, dtype=h.dtype))
            region_start = h * base_region_width + extra_before

            # 5. Within each region, only compute the first half as active computation band, the second half becomes blind spot
            active_band_width = head_region_width // 2

            # 6. Calculate start and end distances of active computation band
            active_band_start = region_start
            active_band_end = region_start + active_band_width

            # 7. Determine if current token pair distance falls within active computation band
            is_in_active_band = (distance >= active_band_start) & (distance < active_band_end)

            # 8. Return final mask: causal AND within active band
            # Note: Token pairs with distance in [active_band_end, region_start + head_region_width)
            # will become permanent "information blind spots", not computed by any head
            return causal & is_in_active_band

        return create_block_mask(
            mask_fn,
            B=None,
            H=self.n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=self.device
        )

    def precompile(self, verbose: bool = True) -> None:
        """Precompile all masks"""
        gc.collect()
        torch.cuda.empty_cache() if self.device == 'cuda' else None

        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() / 1024 ** 3 if self.device == 'cuda' else 0

        print(f"Starting precompilation of masks from {self.min_len} to {self.max_len}...")

        count = 0
        for seq_len in range(self.min_len, self.max_len + 1):
            self.masks[seq_len] = self._create_balanced_mask(seq_len)
            count += 1

            if verbose and seq_len % 100 == 0:
                print(f"  Compiled up to length {seq_len}")

        compile_time = time.time() - start_time
        final_memory = torch.cuda.memory_allocated() / 1024 ** 3 if self.device == 'cuda' else 0
        memory_used = final_memory - initial_memory

        print(f"\nPrecompilation complete [Configuration 4 - Incomplete Coverage Bands]:")
        print(f"  Number of masks: {count}")
        print(f"  Memory usage: {memory_used:.3f} GB")
        print(f"  Compilation time: {compile_time:.1f} seconds")
        print(f"  Average per mask: {memory_used * 1024 / count:.2f} MB")

    def get_mask(self, seq_len: int) -> torch.Tensor:
        """Get mask for specified length"""
        if seq_len not in self.masks:
            if self.min_len <= seq_len <= self.max_len:
                print(f"Warning: Length {seq_len} not precompiled, creating dynamically...")
                self.masks[seq_len] = self._create_balanced_mask(seq_len)
            else:
                return self._create_balanced_mask(seq_len)
        return self.masks[seq_len]

    def clear_cache(self) -> None:
        """Clear cache"""
        self.masks.clear()
        gc.collect()
        torch.cuda.empty_cache() if self.device == 'cuda' else None

    def get_cache_memory_usage(self) -> float:
        """Calculate memory usage of cache (GB)"""
        total_memory = 0
        for seq_len, mask in self.masks.items():
            # BlockMask contains multiple tensor attributes
            if hasattr(mask, 'kv_num_blocks') and mask.kv_num_blocks is not None:
                total_memory += mask.kv_num_blocks.element_size() * mask.kv_num_blocks.nelement()
            if hasattr(mask, 'kv_indices') and mask.kv_indices is not None:
                total_memory += mask.kv_indices.element_size() * mask.kv_indices.nelement()
            if hasattr(mask, 'q_num_blocks') and mask.q_num_blocks is not None:
                total_memory += mask.q_num_blocks.element_size() * mask.q_num_blocks.nelement()
            if hasattr(mask, 'q_indices') and mask.q_indices is not None:
                total_memory += mask.q_indices.element_size() * mask.q_indices.nelement()
        return total_memory / 1024 ** 3  # Convert to GB


def sparse_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_cache: SPAttentionCache
) -> torch.Tensor:
    """
    Execute sparse attention computation

    Args:
        q, k, v: Tensors with shape [batch_size, n_heads, seq_len, d_head]
        mask_cache: SPAttentionCache instance

    Returns:
        Attention output, same shape as input
    """
    seq_len = q.shape[2]
    mask = mask_cache.get_mask(seq_len)
    return flex_attention(q, k, v, block_mask=mask)


# Convenience functions
def create_sp_attention_cache(n_heads: int = 8, mode: str = "short") -> SPAttentionCache:
    """
    Convenience function to create SPAttention cache

    Args:
        n_heads: 8 or 16
        mode: "short" (256-1024) or "long" (256-4096)
    """
    length_mode = LengthMode.SHORT if mode == "short" else LengthMode.LONG
    cache = SPAttentionCache(n_heads, length_mode)
    return cache

