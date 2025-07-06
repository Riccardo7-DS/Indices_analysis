from analysis import summarize_metric_across_horizons, compute_ensamble_rmse, compute_ssr, tensor_ssim
import argparse, os
parser = argparse.ArgumentParser(conflict_handler="resolve")

###### add SSIM index

parser.add_argument('-f')
parser.add_argument("--num_ensambles", type=int, default=os.getenv("num_ensambles", 1), help="if making ensamble predictions")
parser.add_argument("--days", type=int, nargs='+', required=True, help="List of days")
parser.add_argument("--eta", type=float, default=os.getenv("eta", 0.), help="eta value for ddim sampling")
args = parser.parse_args()

metric, funct = "RMSE", compute_ensamble_rmse  # or "mbe", "rmse", etc.
summarize_metric_across_horizons(funct, args.eta, args.num_ensambles, metric, day_list=args.days)
metric, funct = "SSR", compute_ssr  # or "mbe", "rmse", etc.
summarize_metric_across_horizons(funct, args.eta, args.num_ensambles, metric, day_list=args.days)
metric, funct = "SSIM", tensor_ssim  # or "mbe", "rmse", etc.
summarize_metric_across_horizons(funct, args.eta, args.num_ensambles, metric, day_list=args.days)