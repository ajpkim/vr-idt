import argparse
import os

from parser import get_parser


def main():

    print("hello!")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
