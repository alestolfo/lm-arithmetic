import pandas as pd
import numpy as np
from copy import deepcopy


def check_same_length(a, b, tokenizer):
    tok_e1_tok = tokenizer.tokenize(a)
    tok_e2_tok = tokenizer.tokenize(b)
    if len(tok_e1_tok) == len(tok_e2_tok):
        return True
    else:
        return False


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def batch(iterable, bsize=1):
    total_len = len(iterable)
    for ndx in range(0, total_len, bsize):
        yield list(iterable[ndx:min(ndx + bsize, total_len)])

def extract_last_number(text):
    text = ''.join(char if char.isdigit() else ' ' for char in text)
    text = text.split()
    return int(text[-1]) if text else -1


def convert_to_words(num):
    # Python program to print a given number in
    # words. The program handles numbers
    # from 0 to 9999

    # Credits: Mithun Kumar

    num = str(num)

    l = len(num)

    # Base cases
    if l == 0:
        print("empty string")
        return

    if l > 4:
        print("Length more than 4 is not supported")
        return

    # The first string is not used,
    # it is to make array indexing simple
    single_digits = ["zero", "one", "two", "three",
                     "four", "five", "six", "seven",
                     "eight", "nine"]

    # The first string is not used,
    # it is to make array indexing simple
    two_digits = ["", "ten", "eleven", "twelve",
                  "thirteen", "fourteen", "fifteen",
                  "sixteen", "seventeen", "eighteen",
                  "nineteen"]

    # The first two string are not used,
    # they are to make array indexing simple
    tens_multiple = ["", "", "twenty", "thirty", "forty",
                     "fifty", "sixty", "seventy", "eighty",
                     "ninety"]

    tens_power = ["hundred", "thousand"]

    # Used for debugging purpose only
    # print(num, ":", end=" ")
    res = ''

    # For single digit number
    if l == 1:
        res += (single_digits[ord(num[0]) - 48])
        return res.strip()

    # Iterate while num is not '\0'
    x = 0
    while x < len(num):

        # Code path for first 2 digits
        if l >= 3:
            if ord(num[x]) - 48 != 0:
                res += single_digits[ord(num[x]) - 48] + ' '
                res += tens_power[l - 3] + ' '
                # here len can be 3 or 4

            l -= 1

        # Code path for last 2 digits
        else:

            # Need to explicitly handle
            # 10-19. Sum of the two digits
            # is used as index of "two_digits"
            # array of strings
            if ord(num[x]) - 48 == 1:
                sum = (ord(num[x]) - 48 +
                       ord(num[x + 1]) - 48)
                res += two_digits[sum] + ' '
                return res.strip()

            # Need to explicitly handle 20
            elif (ord(num[x]) - 48 == 2 and
                  ord(num[x + 1]) - 48 == 0):
                return "twenty"


            # Rest of the two digit
            # numbers i.e., 21 to 99
            else:
                i = ord(num[x]) - 48
                if i > 0:
                    res += tens_multiple[i] + ' '
                else:
                    print("", end="")
                x += 1
                if ord(num[x]) - 48 != 0:
                    res += (single_digits[ord(num[x]) - 48])
        x += 1

    return res.strip()
