"""Suite-level exceptions."""


class SuiteError(Exception):
    """Base error for odd_suite."""


class ConfigError(SuiteError):
    """Invalid or missing configuration."""


class MissingDependencyError(SuiteError):
    """An optional heavy dependency is not installed."""

    def __init__(self, package: str, extra: str):
        super().__init__(
            f"{package} is required for this operation. "
            f"Install with: pip install 'odd-suite[{extra}]'"
        )
        self.package = package
        self.extra = extra
