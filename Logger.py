"""
This file provides functions which will be used in the fixationanalyzer
backend to print out messages
"""


def printmsg(*messages):
    """prints a fixationanalyzer message without any priority markers"""
    print('(fixationanalyzer) ', *messages)


def debug(*messages):
    """prints a fixationanalyzer debug message"""
    print('(fixationanalyzer)[  DEBUG   ] ', *messages)


def info(*messages):
    """prints a fixationanalyzer info message"""
    print('(fixationanalyzer)[   INFO   ] ', *messages)


def warning(*messages):
    """prints a fixationanalyzer warning message"""
    print('(fixationanalyzer)[ WARNING  ] ', *messages)


def error(*messages):
    """prints a fixationanalyzer error message"""
    print('(fixationanalyzer)[  ERROR   ] ', *messages)


def critical(*messages):
    """prints a fixationanalyzer warning message"""
    print('(fixationanalyzer)[ CRITICAL ] ', *messages)
