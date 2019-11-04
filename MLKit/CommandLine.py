import sys
import MLKit


def get_file_name():
    if len(sys.argv) < 2:
        MLKit.Display.error("Should take one argument.")
    
    return sys.argv[1]
