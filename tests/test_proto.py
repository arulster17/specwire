from specwire.proto import specwire_pb2, specwire_pb2_grpc


def test_messages_round_trip():
    assert specwire_pb2_grpc.DraftStub and specwire_pb2_grpc.TargetStub

    req = specwire_pb2.ProposeRequest(context=[1, 2, 3], k=4)
    req2 = specwire_pb2.ProposeRequest.FromString(req.SerializeToString())
    assert list(req2.context) == [1, 2, 3]
    assert req2.k == 4

    rep = specwire_pb2.VerifyReply(num_accepted=2, next_token=42)
    rep2 = specwire_pb2.VerifyReply.FromString(rep.SerializeToString())
    assert rep2.num_accepted == 2
    assert rep2.next_token == 42
