def error(message, should_exit=True):
    print("\033[91mERROR:\033[0m " + message)
    if should_exit:
        exit()


def warning(message):
    print("\033[93mWarning:\033[0m " + message)
