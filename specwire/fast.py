import torch
from transformers import StaticCache


class StaticRunner:
    """Runs a model on a fixed-size KV cache, replaying a CUDA graph per input length.

    The cache is preallocated to max_len positions and `length` marks how many are
    valid, so rollback just moves that mark. Fixed shapes are what make a step
    capturable: the first call with a given input length (up to graph_max_len) records
    the model's GPU work once, and later calls replay it in a single launch. Longer
    inputs (prompts) and non-CUDA devices run eagerly on the same cache.
    """

    def __init__(self, model, max_len: int = 1024, graph_max_len: int = 16):
        self.model = model
        self.device = model.device
        self.max_len = max_len
        self.graph_max_len = graph_max_len if self.device.type == "cuda" else 0
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)
        self.length = 0
        self.graphs = {}  # input length -> (graph, input ids, offsets, positions, logits)
        self.pool = None  # one memory pool shared by every graph

    def rollback(self, keep_len: int) -> None:
        """Keep the first keep_len cached positions; later ones are overwritten on write."""
        if keep_len > self.length:
            raise ValueError(
                f"keep_len {keep_len} exceeds cached length {self.length}; worker cache was lost"
            )
        self.length = keep_len
        self._set_write_position(keep_len)

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        """Process tokens (1-D, on device) after the cached ones; return their logits.

        A graph's logits live in a fixed buffer that the next call overwrites, so use
        or copy them before calling again.
        """
        n = tokens.shape[0]
        if n == 0:
            raise ValueError("tokens must be non-empty")
        if self.length + n > self.max_len:
            raise ValueError(f"sequence of {self.length + n} tokens exceeds max_len {self.max_len}")
        if n > self.graph_max_len:
            logits = self._run(tokens[None], self._positions(n))
        else:
            if n not in self.graphs:
                self.graphs[n] = self._capture(n)
            graph, ids, offsets, positions, logits = self.graphs[n]
            ids.copy_(tokens[None])
            torch.add(offsets, self.length, out=positions)
            graph.replay()
        self.length += n
        return logits

    def _run(self, ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(
            ids, past_key_values=self.cache, cache_position=positions, use_cache=True
        ).logits[0]

    def _positions(self, n: int) -> torch.Tensor:
        return torch.arange(self.length, self.length + n, device=self.device)

    def _set_write_position(self, position: int) -> None:
        # transformers' StaticLayer writes at its own on-device counter and ignores the
        # cache_position we pass (that only drives the attention mask and RoPE), so the
        # counter must track `length`. Calls and replays advance it by themselves.
        # The cache is allocated under inference_mode, so it can only be written there.
        with torch.inference_mode():
            for layer in self.cache.layers:
                layer.cumulative_length.fill_(position)

    def _capture(self, n: int):
        ids = torch.zeros((1, n), dtype=torch.long, device=self.device)
        offsets = torch.arange(n, device=self.device)
        positions = offsets + self.length
        # Warm up on a side stream, as capture requires. These runs write past `length`,
        # which holds nothing valid, and the counter is rewound after each.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._run(ids, positions)
                self._set_write_position(self.length)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            logits = self._run(ids, positions)
        self.pool = graph.pool()
        self._set_write_position(self.length)
        return graph, ids, offsets, positions, logits
