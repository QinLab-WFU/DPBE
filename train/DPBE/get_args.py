import os

from argsbase import get_baseargs


def get_args():

    parser = get_baseargs()

    parser.add_argument('--use-lam', default=True, help='Use lam')
    parser.add_argument('--loss', default="acm")
    parser.add_argument("--miner", default="dwm", help="choice form [dwm, semi, random]")
    parser.add_argument('--train-n-samples', default=1)
    parser.add_argument('--valid-n-samples', default=1)
    parser.add_argument('--max-pairs', default=5000)
    parser.add_argument('--hessian-memory-factor', default=0.999, type=float, help='Dropout rate')

    parser.add_argument('--calculate-uncertainty', default=False)

    args = parser.parse_args()
    args.method = 'DPBE'
    args.save_dir = os.path.join(args.save_dir, args.method, str(args.output_dim))

    return args
