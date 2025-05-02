import argparse

from .run_sim import run_img_sim
from .utils import benchmark


@benchmark
def cli():
    """Run the simulation on the command line."""
    parser = argparse.ArgumentParser(
        prog="imgsim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
            Recombination image simulation.

            This module takes parameters from a JSON file and uses them to 
            generate images simulating recombination events of different 
            types, as well as images where no recombination has occurred.
            """,
    )

    parser.add_argument(
        "config", help="JSON file with parameters for the simulation."
    )
    args = parser.parse_args()
    run_img_sim(args.config)


if __name__ == "__main__":
    cli()
