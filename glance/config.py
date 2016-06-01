import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--arrangements", help="enable axis arrangements", action="store_true")
args, _ = parser.parse_known_args()


enable_axis_arrangements = args.arrangements