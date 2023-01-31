'''ddmd run'''
import os
import argparse
import evb
import evb.scripts.run_md
import evb.scripts.run_evb

# from ddmd.utils import parse_args

def main(): 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='evb '+evb.__version__)

    modules = [
        evb.scripts.run_md, 
        evb.scripts.run_evb,
    ]

    subparsers = parser.add_subparsers(title='Choose a command')
    subparsers.required = 'True'

    def get_str_name(module):
        return os.path.splitext(os.path.basename(module.__file__))[0]

    for module in modules:
        this_parser = subparsers.add_parser(get_str_name(module), description=module.__doc__)
        this_parser.set_defaults(func=module.main)
        this_parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
        )

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()

