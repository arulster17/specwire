import argparse
from concurrent.futures import ThreadPoolExecutor

import grpc

from specwire.draft import Drafter
from specwire.proto import specwire_pb2, specwire_pb2_grpc
from specwire.target import Verifier

MAX_K = 32


def check_cache_request(request, cache, ctx):
    """Abort unless request.keep_len and request.new_tokens fit the worker's cache."""
    if request.keep_len < 0:
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "keep_len must be >= 0")
    if not request.new_tokens:
        ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "empty new_tokens")
    cached = cache.get_seq_length()
    if request.keep_len > cached:
        ctx.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"keep_len {request.keep_len} exceeds cached length {cached}; worker cache was lost",
        )


class DraftServicer(specwire_pb2_grpc.DraftServicer):
    """Serves a Drafter over gRPC."""

    def __init__(self, drafter):
        self.drafter = drafter

    def Propose(self, request, ctx):
        check_cache_request(request, self.drafter.cache, ctx)
        if not 1 <= request.k <= MAX_K:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"k must be in [1, {MAX_K}]")
        max_len = self.drafter.model.config.max_position_embeddings
        if request.keep_len + len(request.new_tokens) + request.k > max_len:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "sequence + k exceeds model max length")
        tokens = self.drafter.propose(request.keep_len, list(request.new_tokens), request.k)
        return specwire_pb2.ProposeReply(tokens=tokens)


class TargetServicer(specwire_pb2_grpc.TargetServicer):
    """Serves a Verifier over gRPC."""

    def __init__(self, verifier):
        self.verifier = verifier

    def Verify(self, request, ctx):
        check_cache_request(request, self.verifier.cache, ctx)
        if len(request.draft) > MAX_K:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, f"draft longer than {MAX_K}")
        max_len = self.verifier.model.config.max_position_embeddings
        if request.keep_len + len(request.new_tokens) + len(request.draft) > max_len:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "sequence + draft exceeds model max length")
        n, nxt, choices = self.verifier.verify(
            request.keep_len, list(request.new_tokens), list(request.draft)
        )
        return specwire_pb2.VerifyReply(num_accepted=n, next_token=nxt, target_choices=choices)


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
    """Remote Drafter with the same propose() signature."""

    def __init__(self, address: str):
        self.channel = grpc.insecure_channel(address)
        self.stub = specwire_pb2_grpc.DraftStub(self.channel)

    def propose(self, keep_len: int, new_tokens: list[int], k: int) -> list[int]:
        request = specwire_pb2.ProposeRequest(keep_len=keep_len, new_tokens=new_tokens, k=k)
        return list(self.stub.Propose(request).tokens)


class TargetClient:
    """Remote Verifier with the same verify() signature."""

    def __init__(self, address: str):
        self.channel = grpc.insecure_channel(address)
        self.stub = specwire_pb2_grpc.TargetStub(self.channel)

    def verify(
        self, keep_len: int, new_tokens: list[int], draft: list[int]
    ) -> tuple[int, int, list[int]]:
        request = specwire_pb2.VerifyRequest(keep_len=keep_len, new_tokens=new_tokens, draft=draft)
        reply = self.stub.Verify(request)
        return reply.num_accepted, reply.next_token, list(reply.target_choices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["draft", "target"])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model")
    args = parser.parse_args()

    if args.role == "draft":
        port = args.port or 50051
        drafter = Drafter(args.model or "Qwen/Qwen2.5-0.5B")
        server, _ = start_draft_server(drafter, f"{args.host}:{port}")
    else:
        port = args.port or 50052
        verifier = Verifier(args.model or "Qwen/Qwen2.5-1.5B")
        server, _ = start_target_server(verifier, f"{args.host}:{port}")
    print(f"{args.role} server listening on {args.host}:{port}", flush=True)
    server.wait_for_termination()
