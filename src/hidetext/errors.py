class HideTextError(Exception):
    """Base class for all project exceptions."""


class PacketError(HideTextError):
    """Raised when packet framing is invalid."""


class MagicMismatchError(PacketError):
    """Raised when the packet magic is not recognized."""


class ConfigMismatchError(HideTextError):
    """Raised when runtime configuration does not match the packet."""


class IntegrityError(HideTextError):
    """Raised when authenticated decryption fails."""


class SynchronizationError(HideTextError):
    """Raised when the observed text diverges from the expected protocol path."""


class EncodingExhaustedError(HideTextError):
    """Raised when encoding cannot finish within the configured token budget."""


class StallDetectedError(HideTextError):
    """Raised when encoding stops making forward bit progress for too long."""


class ModelBackendError(HideTextError):
    """Raised when the backend cannot provide or parse tokens."""
