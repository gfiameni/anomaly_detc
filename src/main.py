# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = anomaly_detc.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import argparse
import logging
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb



from src.anomaly_detc import __version__, iso_forest

__author__ = ""
__copyright__ = ""
__license__ = ""

_logger = logging.getLogger(__name__)


def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for i in range(n - 1):
        a, b = b, a + b
    return a


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version="anomaly_detc {ver}".format(ver=__version__))
    parser.add_argument(
        dest="n",
        help="n-th Fibonacci number",
        type=int,
        metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args = []):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    """ 
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    _logger.info("Script ends here")
    """

    setup_logging(1)
    _logger.debug("Starting crazy calculations...")
    _logger.info("Script ends here")

    sb.set_style(style="whitegrid")
    sb.set_color_codes()

    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # diagonal covariance
    Nobjs = 3000
    x, y = np.random.multivariate_normal(mean, cov, Nobjs).T
    # Add manual outlier
    x[0] = 3.3
    y[0] = 3.3
    X = np.array([x, y]).T
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=15, facecolor='k', edgecolor='k')

    start = time.time()

    F = iso_forest.iForest(X, ntrees=500, sample_size=256)
    S = F.compute_paths(X_in=X)

    end = time.time()
    _logger.info("Elapsed (with compilation) = %s" % (end - start))

    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(S, kde=True, color="b", ax=axes, axlabel='anomaly score')

    ss = np.argsort(S)
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=15, c='b', edgecolor='b')
    plt.scatter(x[ss[-10:]], y[ss[-10:]], s=55, c='k')
    plt.scatter(x[ss[:10]], y[ss[:10]], s=55, c='r')

    N = 4000
    x2 = np.random.rand(N)
    y2 = np.sin(x2 * 10.) + np.random.randn(N) / 2.

    x2[0] = 0.4;
    y2[0] = 0.9
    x2[1] = 0.6;
    y2[1] = 1.5
    x2[2] = 0.5;
    y2[2] = -3.
    X2 = np.array([x2, y2]).T
    plt.figure(figsize=(9, 6))
    plt.scatter(x2, y2, c='b', edgecolor='b')
    plt.scatter(x2[:3], y2[:3], c='k')
    plt.ylim(-3.2, 3.2)
    plt.xlim(0, 1)

    F2 = iso_forest.iForest(X2, ntrees=500, sample_size=512)
    S2 = F2.compute_paths(X_in=X2)
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    sb.distplot(S2, kde=True, color="b", ax=axes, axlabel='anomaly score')

    ss = np.argsort(S2)
    plt.figure(figsize=(9, 6))
    plt.scatter(x2, y2, c='b', edgecolors='b')
    plt.scatter(x2[ss[-10:]], y2[ss[-10:]], s=55, c='k')
    plt.scatter(x2[ss[:100]], y2[ss[:100]], s=55, c='r')

    # plt.show()

def run():
    """Entry point for console_scripts
    """
    # main(sys.argv[1:])
    main()


if __name__ == "__main__":
    run()
