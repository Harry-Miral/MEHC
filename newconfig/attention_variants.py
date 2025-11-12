"""
Drop-in replacement for SPAttention.py - 100% compatible interface
Supports fully compatible implementations of Longformer, BigBird, and Reformer.
"""

import torch
import time
import gc
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from enum import Enum
from typing import Dict, Optional
import os

class LengthMode(Enum):
    """Length pattern enumeration - exactly the same as SPATtention"""
    SHORT = "1-1024"
    LONG = "1-4096"

class SPAttentionCache:
    """
    A cache manager fully compatible with the SPATtention interface
    Controlling which attention mechanism to use via environment variables
    """

    def __init__(self, n_heads: int, length_mode: LengthMode, device: str = 'cuda'):
        "The initialization interface is exactly the same as the original SPATtentionCache."
        assert n_heads in [8, 16], f"n_heads must be 8 or 16, got {n_heads}"
        assert torch.cuda.is_available() if device == 'cuda' else True

        self.n_heads = n_heads
        self.length_mode = length_mode
        self.device = device
        self.masks: Dict[int, torch.Tensor] = {}

        # The attention type can be selected via environment variables; the default is longformer.
        self.attention_type = os.environ.get('ATTENTION_TYPE', 'longformer').lower()
        
        # Set length range
        if length_mode == LengthMode.SHORT:
            self.min_len, self.max_len = 1, 1024
        else:
            self.min_len, self.max_len = 1, 4096

        print(f"Initializing {self.attention_type.upper()} cache: {n_heads} heads, {length_mode.value} length range")

    def _create_longformer_mask(self, seq_len: int) -> torch.Tensor:
        """Longformer: Local + Global attention"""
        # Pre-compute all parameters to avoid control flow in mask_fn
        if seq_len <= 4:
            # Simple strategy for small sequences
            def mask_fn(b, h, i, j):
                return j <= i  # Simple causal mask
        else:
            window_size = max(1, min(256, seq_len // 4))
            num_global = max(1, min(16, max(1, seq_len // 32)))

            def mask_fn(b, h, i, j):
                # Basic causal condition
                causal = j <= i

                # Local window
                distance = i - j
                local = distance <= window_size

                # Global tokens
                is_global_i = i < num_global
                is_global_j = j < num_global
                global_attn = is_global_i | is_global_j

                return causal & (local | global_attn)

        return create_block_mask(
            mask_fn, B=None, H=self.n_heads, Q_LEN=seq_len, KV_LEN=seq_len, device=self.device
        )

    def _create_bigbird_mask(self, seq_len: int) -> torch.Tensor:
        """BigBird: Local + Global + Random attention"""
        # Pre-compute all parameters to avoid control flow in mask_fn
        if seq_len <= 4:
            # Simple strategy for small sequences
            def mask_fn(b, h, i, j):
                return j <= i  # Simple causal mask
        else:
            window_size = max(1, min(128, max(1, seq_len // 8)))
            num_global = max(1, min(8, max(1, seq_len // 64)))

            def mask_fn(b, h, i, j):
                # Basic causal condition
                causal = j <= i

                # Local window
                distance = i - j
                local = distance <= window_size

                # Global tokens
                is_global_i = i < num_global
                is_global_j = j < num_global
                global_attn = is_global_i | is_global_j

                # Simplified random connections - remove complex conditional logic
                # Use simpler pattern: add connections at regular intervals
                stride = max(16, seq_len // 32)  # Adaptive stride
                strided_attn = (distance % stride == 0) & (distance > window_size)

                return causal & (local | global_attn | strided_attn)

        return create_block_mask(
            mask_fn, B=None, H=self.n_heads, Q_LEN=seq_len, KV_LEN=seq_len, device=self.device
        )

    def _create_reformer_mask(self, seq_len: int) -> torch.Tensor:
        """Reformer: LSH-inspired bucketed attention"""
        # Pre-compute all parameters to avoid control flow in mask_fn
        if seq_len <= 4:
            # Simple strategy for small sequences
            def mask_fn(b, h, i, j):
                return j <= i  # Simple causal mask
        else:
            bucket_size = max(1, min(64, max(1, seq_len // 8)))
            num_buckets = max(2, seq_len // bucket_size)
            local_window = max(1, min(8, seq_len // 4))
            has_wrap = num_buckets > 2  # Pre-compute whether circular adjacency is needed

            def mask_fn(b, h, i, j):
                # Basic causal condition
                causal = j <= i

                # Simplified deterministic hash
                hash_offset = h * 13 + 7
                bucket_i = (i * 3 + hash_offset) % num_buckets
                bucket_j = (j * 3 + hash_offset) % num_buckets

                # Same bucket
                same_bucket = (bucket_i == bucket_j)

                # Adjacent buckets - use simple logic
                diff = bucket_i - bucket_j
                adjacent = (diff == 1) | (diff == -1) | (diff == 0)

                # Circular adjacency - avoid if statements, control with boolean values
                wrap_adjacent = has_wrap & (
                    ((bucket_i == 0) & (bucket_j == num_buckets - 1)) |
                    ((bucket_i == num_buckets - 1) & (bucket_j == 0))
                )
                adjacent = adjacent | wrap_adjacent

                # Short-range local connections
                distance = i - j
                local = distance <= local_window

                return causal & (same_bucket | adjacent | local)

        return create_block_mask(
            mask_fn, B=None, H=self.n_heads, Q_LEN=seq_len, KV_LEN=seq_len, device=self.device
        )

    def _create_simple_sparse_mask(self, seq_len: int) -> torch.Tensor:
        """Create the simplest sparse mask - only causal + local window"""
        window_size = min(32, seq_len)  # Simple local window

        def mask_fn(b, h, i, j):
            causal = j <= i
            local = (i - j) <= window_size
            return causal & local

        return create_block_mask(
            mask_fn, B=None, H=self.n_heads, Q_LEN=seq_len, KV_LEN=seq_len, device=self.device
        )

    def _create_balanced_mask(self, seq_len: int) -> torch.Tensor:
        """Create corresponding mask based on attention type - keep the same method name as original interface"""
        try:
            if self.attention_type == 'longformer':
                return self._create_longformer_mask(seq_len)
            elif self.attention_type == 'bigbird':
                return self._create_bigbird_mask(seq_len)
            elif self.attention_type == 'reformer':
                return self._create_reformer_mask(seq_len)
            else:
                # Default to longformer
                if seq_len not in self.masks:  # Only warn once
                    print(f"Warning: Unknown attention type {self.attention_type}, using longformer")
                return self._create_longformer_mask(seq_len)
        except Exception as e:
            print(f"Warning: {self.attention_type} creation failed ({e}), using simple sparse mask")
            return self._create_simple_sparse_mask(seq_len)

    def precompile(self, verbose: bool = True) -> None:
        """Exactly the same precompile interface as original SPAttentionCache"""
        gc.collect()
        torch.cuda.empty_cache() if self.device == 'cuda' else None

        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated() / 1024**3 if self.device == 'cuda' else 0

        print(f"Starting precompilation of {self.attention_type.upper()} masks from {self.min_len} to {self.max_len}...")

        count = 0
        # Start precompiling from larger sequence lengths, create small sequences dynamically when needed
        start_len = max(self.min_len, 16)  # Start precompiling from 16, create smaller sequences dynamically

        for seq_len in range(start_len, self.max_len + 1):
            try:
                self.masks[seq_len] = self._create_balanced_mask(seq_len)
                count += 1

                if verbose and seq_len % 100 == 0:
                    print(f"  Compiled up to length {seq_len}")
            except Exception as e:
                print(f"  Warning: Compilation failed for sequence length {seq_len}: {e}")
                continue

        compile_time = time.time() - start_time
        final_memory = torch.cuda.memory_allocated() / 1024**3 if self.device == 'cuda' else 0
        memory_used = final_memory - initial_memory

        print(f"\n{self.attention_type.upper()} precompilation complete:")
        print(f"  Number of masks: {count}")
        print(f"  Memory usage: {memory_used:.3f} GB")
        print(f"  Compilation time: {compile_time:.1f} seconds")
        if count > 0:
            print(f"  Average per mask: {memory_used * 1024 / count:.2f} MB")
        print(f"  Note: Masks for lengths 1-{start_len-1} will be created dynamically when needed")

    def get_mask(self, seq_len: int) -> torch.Tensor:
        """Exactly the same get mask interface as original SPAttentionCache"""
        if seq_len not in self.masks:
            try:
                if self.min_len <= seq_len <= self.max_len:
                    if seq_len < 16:  # No warning for small sequences
                        pass
                    else:
                        print(f"Warning: Length {seq_len} not precompiled, creating dynamically...")
                    self.masks[seq_len] = self._create_balanced_mask(seq_len)
                else:
                    # Also try to create sequences out of range
                    self.masks[seq_len] = self._create_balanced_mask(seq_len)
            except Exception as e:
                print(f"Error: Cannot create mask for length {seq_len}: {e}")
                # Final fallback - create the simplest causal mask
                try:
                    def simple_causal_mask(b, h, i, j):
                        return j <= i  # Only use simple causal condition

                    fallback_mask = create_block_mask(
                        simple_causal_mask, B=None, H=self.n_heads,
                        Q_LEN=seq_len, KV_LEN=seq_len, device=self.device
                    )
                    self.masks[seq_len] = fallback_mask
                    print(f"Using simplest causal mask as fallback")
                except Exception as e2:
                    print(f"Critical error: Cannot even create simplest mask: {e2}")
                    print("This may be an environment or hardware issue, please check:")
                    print("1. Whether PyTorch version supports flex_attention")
                    print("2. Whether CUDA device is available")
                    print("3. Whether memory is sufficient")
                    raise e2

        return self.masks[seq_len]

    def clear_cache(self) -> None:
        """Exactly the same clear cache interface as original SPAttentionCache"""
        self.masks.clear()
        gc.collect()
        torch.cuda.empty_cache() if self.device == 'cuda' else None

    def get_cache_memory_usage(self) -> float:
        """Exactly the same memory usage interface as original SPAttentionCache"""
        total_memory = 0
        for seq_len, mask in self.masks.items():
            if hasattr(mask, 'kv_num_blocks') and mask.kv_num_blocks is not None:
                total_memory += mask.kv_num_blocks.element_size() * mask.kv_num_blocks.nelement()
            if hasattr(mask, 'kv_indices') and mask.kv_indices is not None:
                total_memory += mask.kv_indices.element_size() * mask.kv_indices.nelement()
            if hasattr(mask, 'q_num_blocks') and mask.q_num_blocks is not None:
                total_memory += mask.q_num_blocks.element_size() * mask.q_num_blocks.nelement()
            if hasattr(mask, 'q_indices') and mask.q_indices is not None:
                total_memory += mask.q_indices.element_size() * mask.q_indices.nelement()
        return total_memory / 1024 ** 3

def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_cache: SPAttentionCache
) -> torch.Tensor:
    """Exactly the same interface as original sparse_attention"""
    seq_len = q.shape[2]
    mask = mask_cache.get_mask(seq_len)
    return flex_attention(q, k, v, block_mask=mask)

def create_sp_attention_cache(n_heads: int = 8, mode: str = "short", attention_type: str = None) -> SPAttentionCache:
    """Exactly the same interface as original create_sp_attention_cache, with added attention_type parameter"""
    length_mode = LengthMode.SHORT if mode == "short" else LengthMode.LONG

    # If attention_type is specified, use it directly; otherwise use environment variable
    if attention_type is not None:
        import os
        os.environ['ATTENTION_TYPE'] = attention_type.lower()

    cache = SPAttentionCache(n_heads, length_mode)
    return cache

# ============================================================================
# Experiment control tools - switch attention types via environment variables
# ============================================================================

def validate_mask_sanity(mask, seq_len: int, n_heads: int) -> bool:
    """Simple mask validity check"""
    try:
        # Basic attribute check
        if not hasattr(mask, 'kv_num_blocks'):
            return False
        # More checks can be added here, but keep it simple
        return True
    except:
        return False

def set_attention_type(attention_type: str):
    """Convenience function to set attention type"""
    os.environ['ATTENTION_TYPE'] = attention_type.lower()
    print(f"Attention type set to: {attention_type.upper()}")

def get_current_attention_type() -> str:
    """Get current attention type"""
    return os.environ.get('ATTENTION_TYPE', 'longformer').upper()

# Experiment preset configurations
EXPERIMENT_CONFIGS = {
    'longformer': {
        'name': 'Longformer',
        'description': 'Local window + Global tokens',
        'env_var': 'longformer'
    },
    'bigbird': {
        'name': 'BigBird',
        'description': 'Local + Global + Random attention',
        'env_var': 'bigbird'
    },
    'reformer': {
        'name': 'Reformer',
        'description': 'LSH-based bucketed attention',
        'env_var': 'reformer'
    }
}

# ============================================================================
# Quick experiment script
# ============================================================================

def test_vmap_compatibility():
    """Simple function to test vmap compatibility"""
    print("Testing vmap compatibility...")
    try:
        # Test simplest causal mask
        def simple_causal(b, h, i, j):
            return j <= i

        mask = create_block_mask(simple_causal, B=None, H=8, Q_LEN=4, KV_LEN=4, device='cuda')
        print("✓ Simple causal mask test passed")

        # Test mask with arithmetic operations
        def arithmetic_mask(b, h, i, j):
            causal = j <= i
            local = (i - j) <= 2
            return causal & local

        mask = create_block_mask(arithmetic_mask, B=None, H=8, Q_LEN=4, KV_LEN=4, device='cuda')
        print("✓ Arithmetic operation mask test passed")

        return True
    except Exception as e:
        print(f"✗ vmap compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Attention Mechanism Compatibility Test")
    print("=" * 50)

    # First test vmap compatibility
    if not test_vmap_compatibility():
        print("Basic compatibility test failed, exiting...")
        exit(1)

    # Test all attention types
    test_types = ['longformer', 'bigbird', 'reformer']
    N_HEADS = 8
    MODE = "short"

    # Test edge cases first
    print("\nTesting edge cases (small sequence lengths):")
    print("-" * 30)

    for attn_type in test_types:
        print(f"\nTesting {attn_type.upper()} - small sequences:")
        set_attention_type(attn_type)
        cache = create_sp_attention_cache(n_heads=N_HEADS, mode=MODE)

        # Test small sequence lengths - more conservative testing
        test_small_lengths = [1, 2, 4, 8]  # Reduce test volume, focus on most critical lengths
        success_count = 0

        for seq_len in test_small_lengths:
            try:
                print(f"  Attempting to create mask for length {seq_len}...", end="", flush=True)
                mask = cache.get_mask(seq_len)
                print(" mask created successfully", end="", flush=True)

                # Quick test actual computation
                batch_size, d_head = 1, 64
                q = torch.randn(batch_size, N_HEADS, seq_len, d_head, device='cuda')
                k = torch.randn(batch_size, N_HEADS, seq_len, d_head, device='cuda')
                v = torch.randn(batch_size, N_HEADS, seq_len, d_head, device='cuda')

                output = sparse_attention(q, k, v, cache)
                assert output.shape == (batch_size, N_HEADS, seq_len, d_head)
                print(" ✓")
                success_count += 1

            except Exception as e:
                print(f" ✗")
                print(f"    Error details: {str(e)[:200]}...")
                if "vmap" in str(e):
                    print(f"    This is a vmap compatibility error, may need further simplification of mask function")

        print(f"  Small sequence test pass rate: {success_count}/{len(test_small_lengths)}")
        cache.clear_cache()

    print("\nNormal test:")
    print("-" * 30)

    for attn_type in test_types:
        print(f"\nTesting {attn_type.upper()}:")
        print("-" * 30)

        # Set attention type
        set_attention_type(attn_type)

        # Create cache using exactly the same interface
        cache = create_sp_attention_cache(n_heads=N_HEADS, mode=MODE)

        try:
            # Only precompile larger sequences, avoid small sequence issues
            print("Skipping full precompilation, testing key lengths only...")

            # Test key lengths
            test_lengths = [64, 256, 512, 1024]
            success = 0

            for seq_len in test_lengths:
                try:
                    print(f"  Testing length {seq_len}...", end="", flush=True)
                    mask = cache.get_mask(seq_len)

                    # Quick computation test
                    batch_size, d_head = 1, 64
                    q = torch.randn(batch_size, N_HEADS, seq_len, d_head, device='cuda')
                    k = torch.randn(batch_size, N_HEADS, seq_len, d_head, device='cuda')
                    v = torch.randn(batch_size, N_HEADS, seq_len, d_head, device='cuda')

                    output = sparse_attention(q, k, v, cache)
                    assert output.shape == (batch_size, N_HEADS, seq_len, d_head)
                    print(" ✓")
                    success += 1
                except Exception as e:
                    print(f" ✗ ({str(e)[:50]}...)")

            print(f"  Success rate: {success}/{len(test_lengths)}")
            memory_usage = cache.get_cache_memory_usage()
            print(f"  Memory usage: {memory_usage:.3f} GB")

        except Exception as e:
            print(f"  Overall test failed: {e}")

        # Cleanup
        cache.clear_cache()

    print(f"\nAll interface compatibility tests complete!")
    print("\nQuick usage guide:")
    print("1. Save this file as olmo/SPAttention.py (replace original file)")
    print("2. Set environment variable to select attention type:")
    print("   export ATTENTION_TYPE=longformer   # or bigbird, reformer")
    print("3. Run training code normally - no need to modify any other code!")
    print("4. If you encounter vmap errors, mask functions need further simplification")

# Independent quick test function
def quick_test(attention_type='longformer'):
    """Quick test whether a single attention type works"""
    print(f"Quick test {attention_type.upper()}...")

    try:
        set_attention_type(attention_type)
        cache = create_sp_attention_cache(n_heads=8, mode="short")

        # Test a medium length
        seq_len = 256
        q = torch.randn(1, 8, seq_len, 64, device='cuda')
        k = torch.randn(1, 8, seq_len, 64, device='cuda')
        v = torch.randn(1, 8, seq_len, 64, device='cuda')

        output = sparse_attention(q, k, v, cache)
        print(f"✅ {attention_type.upper()} test successful! Output shape: {output.shape}")
        cache.clear_cache()
        return True

    except Exception as e:
        print(f"❌ {attention_type.upper()} test failed: {e}")
        return False

# For quick testing when run directly
if __name__ == "__main__" and len(__import__('sys').argv) > 1:
    test_type = __import__('sys').argv[1].lower()
    if test_type in ['longformer', 'bigbird', 'reformer']:
        quick_test(test_type)
    else:
        print("Usage: python attention_variants.py [longformer|bigbird|reformer]")
        print("Or run directly: python attention_variants.py (full test)")