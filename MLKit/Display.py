import MLKit


def error(message, should_exit=True):
    print("\033[91mERROR:\033[0m " + message)
    if should_exit:
        exit()


def warning(message):
    print("\033[93mWarning:\033[0m " + message)


def sized_str(string, size):
    final_str = string

    if len(string) > size:
        first_part = final_str[:int((size - 5) / 2)]
        second_part = final_str[-int((size - 5) / 2):]
        final_str = "  " + first_part + "..." + second_part

    for _ in range(0, size - len(final_str)):
        final_str = " " + final_str
    
    return final_str
