from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ProposeRequest(_message.Message):
    __slots__ = ("keep_len", "new_tokens", "k")
    KEEP_LEN_FIELD_NUMBER: _ClassVar[int]
    NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    keep_len: int
    new_tokens: _containers.RepeatedScalarFieldContainer[int]
    k: int
    def __init__(self, keep_len: _Optional[int] = ..., new_tokens: _Optional[_Iterable[int]] = ..., k: _Optional[int] = ...) -> None: ...

class ProposeReply(_message.Message):
    __slots__ = ("tokens",)
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, tokens: _Optional[_Iterable[int]] = ...) -> None: ...

class VerifyRequest(_message.Message):
    __slots__ = ("keep_len", "new_tokens", "draft")
    KEEP_LEN_FIELD_NUMBER: _ClassVar[int]
    NEW_TOKENS_FIELD_NUMBER: _ClassVar[int]
    DRAFT_FIELD_NUMBER: _ClassVar[int]
    keep_len: int
    new_tokens: _containers.RepeatedScalarFieldContainer[int]
    draft: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, keep_len: _Optional[int] = ..., new_tokens: _Optional[_Iterable[int]] = ..., draft: _Optional[_Iterable[int]] = ...) -> None: ...

class VerifyReply(_message.Message):
    __slots__ = ("num_accepted", "next_token", "target_choices")
    NUM_ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    NEXT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TARGET_CHOICES_FIELD_NUMBER: _ClassVar[int]
    num_accepted: int
    next_token: int
    target_choices: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, num_accepted: _Optional[int] = ..., next_token: _Optional[int] = ..., target_choices: _Optional[_Iterable[int]] = ...) -> None: ...
