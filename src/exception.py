import sys
from src.logger import logging

def get_error_message(error, error_detail: sys):
    _, _ , exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        line_number,
        str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail: sys) -> None:
        super().__init__(error)
        self.error_message = get_error_message(error=error, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
    
if __name__ == '__main__':
    try:
        temp = 1/0
    except Exception as e:
        logging.info("Divide by zero error")
        raise CustomException(e, sys)