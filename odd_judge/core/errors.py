"""Error types for ODD judgment."""


class OddJudgeError(Exception):
    """Base error."""


class ConfigError(OddJudgeError):
    """Invalid configuration."""


class ObservationError(OddJudgeError):
    """Invalid observation input."""


class RuleTypeError(OddJudgeError):
    """Unknown or invalid rule type."""
