import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--angle_dispersion_threshold",
        type=float,
        required=True,
        metavar="",
        help="Max dispersion for fixation",
    )
    parser.add_argument(
        "-t",
        "--base_window_time",
        type=float,
        required=True,
        metavar="",
        help="Starting window time length",
    )
    parser.add_argument(
        "-f",
        "--min_frequency",
        type=float,
        required=True,
        metavar="",
        help="Minimum sampling frequency",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        metavar="",
        help="Path of directory of data csv files to classify",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        metavar="",
        help="Path of location to write to",
    )
    return parser
