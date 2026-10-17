from transformers import DynamicCache


def reuse_cache(cache: DynamicCache, keep_len: int, new_tokens: list[int]) -> DynamicCache:
    """Return cache cut to its first keep_len tokens (fresh if keep_len is 0)."""
    if not new_tokens:
        raise ValueError("new_tokens must be non-empty")
    if keep_len == 0:
        return DynamicCache()
    cached = cache.get_seq_length()
    if keep_len > cached:
        raise ValueError(
            f"keep_len {keep_len} exceeds cached length {cached}; worker cache was lost"
        )
    cache.crop(keep_len - cached)  # transformers 5 takes -tokens_to_remove, not a length
    return cache
