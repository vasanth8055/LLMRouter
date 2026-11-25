class ProviderError(RuntimeError):
    """Base class for safe, provider-facing inference errors."""

    def __init__(self, message: str, *, retryable: bool = False, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class ProviderConfigurationError(ProviderError):
    """Required provider configuration is missing or invalid."""


class ProviderAuthenticationError(ProviderError):
    """Provider rejected credentials."""


class ProviderRateLimitError(ProviderError):
    """Provider rate limit was reached."""

    def __init__(self, message: str, *, retryable: bool = True, status_code: int | None = 429) -> None:
        super().__init__(message, retryable=retryable, status_code=status_code)


class ProviderTimeoutError(ProviderError):
    """Provider request exceeded its configured timeout."""

    def __init__(self, message: str, *, retryable: bool = True, status_code: int | None = None) -> None:
        super().__init__(message, retryable=retryable, status_code=status_code)


class ProviderResponseError(ProviderError):
    """Provider returned an unusable response."""
