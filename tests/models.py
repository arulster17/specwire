"""Models the tests run. CI sets smaller ones, since exactness doesn't depend on size;
draft and target must share a tokenizer."""

import os

DRAFT = os.environ.get("SPECWIRE_TEST_DRAFT", "Qwen/Qwen2.5-0.5B")
TARGET = os.environ.get("SPECWIRE_TEST_TARGET", "Qwen/Qwen2.5-1.5B")
