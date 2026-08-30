import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Builds a detailed error string: which file, which line, and the
    original exception message. Far more useful in logs than the bare
    exception when something breaks three layers deep in a pipeline.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return (
        f"Error occurred in python script [{file_name}] "
        f"line number [{line_number}] error message [{str(error)}]"
    )


class CustomException(Exception):
    """
    Wrap any caught exception in this before re-raising:
        try:
            ...
        except Exception as e:
            raise CustomException(e, sys)
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
