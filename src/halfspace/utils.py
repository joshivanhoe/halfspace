import logging
from typing import Union, Iterable


def log_table_header(columns: Iterable[str]) -> None:
    columns = ["{:15}".format(col) for col in columns]
    logging.info('-{}-'.format("-".join("-" * len(col) for col in columns)))
    logging.info('|{}|'.format("|".join(columns)))
    logging.info('-{}-'.format("-".join("-" * len(col) for col in columns)))


def log_table_row(values: Iterable[Union[float, int]]) -> None:
    values = [("{:15}" if isinstance(value, int) else "{:15.3e}").format(value) for value in values]
    logging.info('|{}|'.format("|".join(values)))

