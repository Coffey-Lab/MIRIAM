#!/usr/bin/env python3
import time
import argparse
from CellSeg import CellSegQuant


def main():
    t0 = time.time()
    choice = [0, 1]
    parser = argparse.ArgumentParser(
        description="""Cell Segmentation: CellSeg(Directory, Quantification, Shape, Stroma, Tumor, Start)
                                     Run cell segmentaiton on multiplexed imaging data. Runs all options and starts at first image by default."""
    )
    parser.add_argument(
        "directory",
        metavar="directory",
        type=str,
        help="directory to AFRemoved and registered images",
    )
    parser.add_argument(
        "quantify",
        metavar="quantify",
        type=int,
        choices=choice,
        default=1,
        help="whether or not to quantify, 1=yes, 0=no",
    )
    parser.add_argument(
        "shape",
        metavar="shape",
        type=int,
        choices=choice,
        default=1,
        help="whether or not to quantify, 1=yes, 0=no",
    )
    parser.add_argument(
        "stroma",
        metavar="stroma",
        type=int,
        choices=choice,
        default=1,
        help="whether or not to quantify, 1=yes, 0=no",
    )
    parser.add_argument(
        "tumor",
        metavar="tumor",
        type=int,
        choices=choice,
        default=1,
        help="whether or not to quantify, 1=yes, 0=no",
    )
    parser.add_argument(
        "--start",
        metavar="start",
        type=int,
        choices=choice,
        default=0,
        help="what image position to start processing on, default=0",
    )

    args = parser.parse_args()
    CellSegQuant.CellSeg(
        args.directory, args.quantify, args.shape, args.stroma, args.tumor, args.start
    )
    t1 = time.time()
    print("time:", t1 - t0)


if __name__ == "__main__":
    main()
