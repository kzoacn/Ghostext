class GhostextError(Exception):
    """Base class for all project exceptions."""


class PacketError(GhostextError):
    """Raised when packet framing is invalid."""


class ConfigMismatchError(GhostextError):
    """Raised when runtime configuration does not match the packet."""


class IntegrityError(GhostextError):
    """Raised when authenticated decryption fails."""


class SynchronizationError(GhostextError):
    """Raised when the observed text diverges from the expected protocol path."""


class EncodingExhaustedError(GhostextError):
    """Raised when encoding cannot finish within the configured token budget."""


class StallDetectedError(GhostextError):
    """Raised when encoding stops making forward bit progress for too long."""


class LowEntropyRetryLimitError(GhostextError):
    """Raised when repeated encoding attempts all fall into a low-entropy regime."""


class UnsafeTokenizationError(GhostextError):
    """Raised when every candidate would retokenize into a different token path."""


class ModelBackendError(GhostextError):
    """Raised when the backend cannot provide or parse tokens."""
