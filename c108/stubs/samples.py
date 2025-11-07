from dataclasses import dataclass
from typing_extensions import Self

from c108.dataclasses import mergeable


# Sample Mergeable -----------------------------------------------------------------------------------------------------
@mergeable(include=["name", "value"])
@dataclass
class Mergeable:
    name: str
    value: int | None = None
    timeout: int = 30
    retries: int = 3

    def merge(self, name: str = None, value: int | None = None) -> Self:
        """Create a new SampleMerge instance with selectively updated fields."""
        # This is a stub for docs and type hinting
        raise NotImplementedError("The implementation is handled by @mergeable decorator")


@mergeable
@dataclass
class MergeableShort:
    name: str
    value: int | None = None
    timeout: int = 52
    retries: int = 540

    def merge(self, **kwargs) -> Self:
        """Create a new SampleMergeShort instance with selectively updated fields."""
        # This is a stub for docs and type hinting
        raise NotImplementedError("Implementation handled by @mergeable")


@dataclass
class Merge:
    name: str
    value: int | None = None
    timeout: int = 52
    retries: int = 540

    def merge(self, **kwargs) -> Self:
        """Create a new SampleMergeShort instance with selectively updated fields."""
        # This is a stub for Docs and type hinting
        raise NotImplementedError("Implementation handled by @mergeable")
