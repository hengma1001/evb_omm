#!env python

from evb.sim import Simulate
from evb.utils import parse_args, dict_from_yaml

def main(args): 
    cfg_file = args.config
    sim_setup = dict_from_yaml(cfg_file)
    sim_imp = Simulate(**sim_setup)
    sim_imp.md_run()


if __name__ == '__main__': 
    args = parse_args()
    main(args)