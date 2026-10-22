import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch

from specwire.draft import Drafter
from specwire.proto import specwire_pb2, specwire_pb2_grpc
from specwire.target import Verifier

MAX_K = 32
MAX_TOP_K = 1000


def check_cache_request(request, worker, ctx):
    """Abort unless request.keep_len and request.new_tokens fit the worker's cache."""
    if request.keep_len < 0:
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "keep_len must be >= 0")
    if not request.new_tokens:
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "empty new_tokens")
    cached = worker.cached_len
    if request.keep_len > cached:
        ctx.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"keep_len {request.keep_len} exceeds cached length {cached}; worker cache was lost",
        )
    if request.temperature < 0:
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "temperature must be >= 0")


def check_draft_dists(request, vocab_size, ctx):
    """Abort unless every draft token is in the vocab and, when sampling, was drawn
    from a valid sparse distribution that gives it nonzero probability."""
    if any(not 0 <= token < vocab_size for token in request.draft):
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "draft token outside the vocab")
    if request.temperature == 0:
        return
    if len(request.draft_dists) != len(request.draft):
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "need one draft dist per draft token")
    for token, dist in zip(request.draft, request.draft_dists):
        if not dist.ids or len(dist.ids) != len(dist.probs):
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "draft dist ids and probs must match")
        if any(not 0 <= i < vocab_size for i in dist.ids) or any(q < 0 for q in dist.probs):
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "bad draft dist")
        if dict(zip(dist.ids, dist.probs)).get(token, 0) <= 0:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "draft token has zero draft probability")


def to_pb_dists(dists):
    return [specwire_pb2.DraftDist(ids=ids, probs=probs) for ids, probs in dists]


def from_pb_dists(dists):
    return [(list(d.ids), list(d.probs)) for d in dists]


class DraftServicer(specwire_pb2_grpc.DraftServicer):
    """Serves a Drafter over gRPC."""

    def __init__(self, drafter):
        self.drafter = drafter

    def Propose(self, request, ctx):
        check_cache_request(request, self.drafter, ctx)
        if not 1 <= request.k <= MAX_K:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"k must be in [1, {MAX_K}]")
        if request.temperature > 0 and not 1 <= request.top_k <= MAX_TOP_K:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"top_k must be in [1, {MAX_TOP_K}]")
        max_len = self.drafter.max_len
        if request.keep_len + len(request.new_tokens) + request.k > max_len:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "sequence + k exceeds model max length")
        start = time.perf_counter()
        tokens, dists = self.drafter.propose(
            request.keep_len,
            list(request.new_tokens),
            request.k,
            request.temperature,
            request.top_k,
            request.seed,
        )
        return specwire_pb2.ProposeReply(
            tokens=tokens, dists=to_pb_dists(dists), compute_s=time.perf_counter() - start
        )


class TargetServicer(specwire_pb2_grpc.TargetServicer):
    """Serves a Verifier over gRPC."""

    def __init__(self, verifier):
        self.verifier = verifier

    def Verify(self, request, ctx):
        check_cache_request(request, self.verifier, ctx)
        if len(request.draft) > MAX_K:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"draft longer than {MAX_K}")
        max_len = self.verifier.max_len
        if request.keep_len + len(request.new_tokens) + len(request.draft) > max_len:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "sequence + draft exceeds model max length")
        check_draft_dists(request, self.verifier.model.config.vocab_size, ctx)
        start = time.perf_counter()
        n, nxt, choices = self.verifier.verify(
            request.keep_len,
            list(request.new_tokens),
            list(request.draft),
            from_pb_dists(request.draft_dists),
            request.temperature,
            request.seed,
        )
        return specwire_pb2.VerifyReply(
            num_accepted=n,
            next_token=nxt,
            target_choices=choices,
            compute_s=time.perf_counter() - start,
        )


# Not handled yet: no auth or TLS, so before listening on a public interface (RunPod)
# restrict the firewall to our own machines or add a shared-token check.
# Clients also set no per-call deadline.
def start_draft_server(drafter, address: str = "localhost:50051"):
    """Start a Draft server; return (server, bound_port)."""
    # One worker: the model handles one request at a time.
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    specwire_pb2_grpc.add_DraftServicer_to_server(DraftServicer(drafter), server)
    port = server.add_insecure_port(address)
    server.start()
    return server, port


def start_target_server(verifier, address: str = "localhost:50052"):
    """Start a Target server; return (server, bound_port)."""
    # One worker: the model handles one request at a time.
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    specwire_pb2_grpc.add_TargetServicer_to_server(TargetServicer(verifier), server)
    port = server.add_insecure_port(address)
    server.start()
    return server, port


class DraftClient:
    """Remote Drafter with the same propose() signature.

    last_compute_s is the server's time in the last propose, without the network.
    """

    def __init__(self, address: str):
        self.channel = grpc.insecure_channel(address)
        self.stub = specwire_pb2_grpc.DraftStub(self.channel)
        self.last_compute_s = 0.0

    def propose(
        self,
        keep_len: int,
        new_tokens: list[int],
        k: int,
        temperature: float = 0.0,
        top_k: int = 20,
        seed: int = 0,
    ) -> tuple[list[int], list[tuple[list[int], list[float]]]]:
        request = specwire_pb2.ProposeRequest(
            keep_len=keep_len,
            new_tokens=new_tokens,
            k=k,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        reply = self.stub.Propose(request)
        self.last_compute_s = reply.compute_s
        return list(reply.tokens), from_pb_dists(reply.dists)


class TargetClient:
    """Remote Verifier with the same verify() signature.

    last_compute_s is the server's time in the last verify, without the network.
    """

    def __init__(self, address: str):
        self.channel = grpc.insecure_channel(address)
        self.stub = specwire_pb2_grpc.TargetStub(self.channel)
        self.last_compute_s = 0.0

    def verify(
        self,
        keep_len: int,
        new_tokens: list[int],
        draft: list[int],
        draft_dists: list[tuple[list[int], list[float]]] = (),
        temperature: float = 0.0,
        seed: int = 0,
    ) -> tuple[int, int, list[int]]:
        request = specwire_pb2.VerifyRequest(
            keep_len=keep_len,
            new_tokens=new_tokens,
            draft=draft,
            draft_dists=to_pb_dists(draft_dists),
            temperature=temperature,
            seed=seed,
        )
        reply = self.stub.Verify(request)
        self.last_compute_s = reply.compute_s
        return reply.num_accepted, reply.next_token, list(reply.target_choices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["draft", "target"])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--fast", action="store_true", help="static cache + CUDA graphs")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    if args.role == "draft":
        port = args.port or 50051
        drafter = Drafter(args.model or "Qwen/Qwen2.5-0.5B", args.device, dtype, args.fast)
        server, _ = start_draft_server(drafter, f"{args.host}:{port}")
    else:
        port = args.port or 50052
        verifier = Verifier(args.model or "Qwen/Qwen2.5-1.5B", args.device, dtype, args.fast)
        server, _ = start_target_server(verifier, f"{args.host}:{port}")
    print(f"{args.role} server listening on {args.host}:{port}", flush=True)
    server.wait_for_termination()
