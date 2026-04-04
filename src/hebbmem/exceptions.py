"""Custom exceptions for hebbmem."""


class HebbMemError(Exception):
    """Base exception for all hebbmem errors.

    Catch this to handle any hebbmem-specific error.
    """


class EncoderError(HebbMemError):
    """Raised when encoder fails to encode text or is misconfigured.

    Common triggers:
        - ``HebbMem(encoder="sentence-transformer")`` without the package installed.
        - ``HebbMem(encoder="nonexistent")`` with an unknown encoder name.
    """


class PersistenceError(HebbMemError):
    """Raised on save/load failures.

    Common triggers:
        - Loading a file that is not a valid hebbmem SQLite database.
        - SQLite write errors (e.g., disk full, permission denied).
    """


class MemoryNotFoundError(HebbMemError):
    """Raised when referencing a memory_id that doesn't exist.

    Common triggers:
        - ``mem.forget(unknown_uuid)`` with an ID not in the graph.
    """


class ConfigError(HebbMemError):
    """Raised when config values are invalid.

    Common triggers:
        - Decay rates outside valid ranges.
        - Missing scoring weight keys.
    """
