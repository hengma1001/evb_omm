import evb
from evb.evb import evb_run
from evb.utils import parse_args

logger = evb.utils.build_logger(debug=1)

def main(args):
    cfg_file = args.config
    evb_runs = evb_run(cfg_file)
    if args.dry_run: 
        evb_runs.build_tasks()
    else:
        evb_runs.run()


if __name__ == '__main__': 
    args = parse_args()
    main(args)