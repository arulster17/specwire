from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DraftDist(_message.Message):
    __slots__ = ("ids", "probs")
    IDS_FIELD_NUMBER: _ClassVar[int]
    PROBS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedScalarFieldContainer[int]
    probs: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, ids: _Optional[_Iterable[int]] = ..., probs: _Optional[_Iterable[float]] = ...) -> None: ...

class MemoryRequest(_message.Message):
    __slots__ = ("reset",)
    RESET_FIELD_NUMBER: _ClassVar[int]
    reset: bool
    def __init__(self, reset: _Optional[bool] = ...) -> None: ...

class MemoryReply(_message.Message):
    __slots__ = ("peak_allocated", "peak_reserved")
    PEAK_ALLOCATED_FIELD_NUMBER: _ClassVar[int]
    PEAK_RESERVED_FIELD_NUMBER: _ClassVar[int]
    peak_allocated: int
    peak_reserved: int
    def __init__(self, peak_allocated: _Optional[int] = ..., peak_reserved: _Optional[int] = ...) -> None: ...

class ProposeRequest(_message.Message):
    __slots__ = ("keep_len", "new_tokens", "k", "temperature", "top_k", "seed")
    KEEP_LEN_FIELD_NUMBER: _ClassVar[int]
    NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    keep_len: int
    new_tokens: _containers.RepeatedScalarFieldContainer[int]
    k: int
    temperature: float
    top_k: int
    seed: int
    def __init__(self, keep_len: _Optional[int] = ..., new_tokens: _Optional[_Iterable[int]] = ..., k: _Optional[int] = ..., temperature: _Optional[float] = ..., top_k: _Optional[int] = ..., seed: _Optional[int] = ...) -> None: ...

class ProposeReply(_message.Message):
    __slots__ = ("tokens", "dists", "compute_s")
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    DISTS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_S_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedScalarFieldContainer[int]
    dists: _containers.RepeatedCompositeFieldContainer[DraftDist]
    compute_s: float
    def __init__(self, tokens: _Optional[_Iterable[int]] = ..., dists: _Optional[_Iterable[_Union[DraftDist, _Mapping]]] = ..., compute_s: _Optional[float] = ...) -> None: ...

class VerifyRequest(_message.Message):
    __slots__ = ("keep_len", "new_tokens", "draft", "draft_dists", "temperature", "seed")
    KEEP_LEN_FIELD_NUMBER: _ClassVar[int]
    NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    DRAFT_FIELD_NUMBER: _ClassVar[int]
    DRAFT_DISTS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    keep_len: int
    new_tokens: _containers.RepeatedScalarFieldContainer[int]
    draft: _containers.RepeatedScalarFieldContainer[int]
    draft_dists: _containers.RepeatedCompositeFieldContainer[DraftDist]
    temperature: float
    seed: int
    def __init__(self, keep_len: _Optional[int] = ..., new_tokens: _Optional[_Iterable[int]] = ..., draft: _Optional[_Iterable[int]] = ..., draft_dists: _Optional[_Iterable[_Union[DraftDist, _Mapping]]] = ..., temperature: _Optional[float] = ..., seed: _Optional[int] = ...) -> None: ...

class VerifyReply(_message.Message):
    __slots__ = ("num_accepted", "next_token", "target_choices", "compute_s")
    NUM_ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    NEXT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TARGET_CHOICES_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_S_FIELD_NUMBER: _ClassVar[int]
    num_accepted: int
    next_token: int
    target_choices: _containers.RepeatedScalarFieldContainer[int]
    compute_s: float
    def __init__(self, num_accepted: _Optional[int] = ..., next_token: _Optional[int] = ..., target_choices: _Optional[_Iterable[int]] = ..., compute_s: _Optional[float] = ...) -> None: ...
