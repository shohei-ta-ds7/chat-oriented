# coding: utf-8

import sys

def loginfo_and_print(logger, message):
    logger.info(message)
    print(message)


def logerror_and_print(logger, message):
    logger.error(message)
    sys.stderr.write(message+"\n")
