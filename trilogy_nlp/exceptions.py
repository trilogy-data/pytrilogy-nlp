class SemanticExecutionException(BaseException):
    pass


class ValidationPassedException(BaseException):
    def __init__(self, ir, *args, **kwargs):
        self.ir = ir
        super().__init__(*args, **kwargs)
