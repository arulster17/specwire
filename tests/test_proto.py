from specwire.proto import specwire_pb2, specwire_pb2_grpc


def test_messages_round_trip():
    assert specwire_pb2_grpc.DraftStub and specwire_pb2_grpc.TargetStub

    req = specwire_pb2.ProposeRequest(keep_len=7, new_tokens=[1, 2, 3], k=4)
    req2 = specwire_pb2.ProposeRequest.FromString(req.SerializeToString())
    assert req2.keep_len == 7
    assert list(req2.new_tokens) == [1, 2, 3]
    assert req2.k == 4

    rep = specwire_pb2.VerifyReply(num_accepted=2, next_token=42)
    rep2 = specwire_pb2.VerifyReply.FromString(rep.SerializeToString())
    assert rep2.num_accepted == 2
    assert rep2.next_token == 42


def test_draft_dists_round_trip():
    dist = specwire_pb2.DraftDist(ids=[3, 9], probs=[0.75, 0.25])
    rep = specwire_pb2.ProposeReply(tokens=[3], dists=[dist])
    rep2 = specwire_pb2.ProposeReply.FromString(rep.SerializeToString())
    assert list(rep2.dists[0].ids) == [3, 9]
    assert list(rep2.dists[0].probs) == [0.75, 0.25]


def test_memory_reply_round_trip():
    rep = specwire_pb2.MemoryReply(peak_allocated=5 * 2**30, peak_reserved=6 * 2**30)
    rep2 = specwire_pb2.MemoryReply.FromString(rep.SerializeToString())
    assert (rep2.peak_allocated, rep2.peak_reserved) == (5 * 2**30, 6 * 2**30)
    assert specwire_pb2.MemoryRequest(reset=True).reset
