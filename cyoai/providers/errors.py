class ProviderError(RuntimeError):
    pass


class MissingCredentialsError(ProviderError):
    pass
