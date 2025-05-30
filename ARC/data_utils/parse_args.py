import argparse

parser = argparse.ArgumentParser()

# -----------------------File------------------------
parser.add_argument('--city',                 default="Chi",       help='City name, can be NY or Chi or SF')
parser.add_argument('--task',                 default="checkIn",  help='Downstrea task name, can be crime or checkIn or serviceCall')
parser.add_argument('--mobility_dist',        default='/mob_dist.npy')
parser.add_argument('--POI_dist',             default='/poi_dist.npy')
parser.add_argument('--landUse_dist',         default='/landUse_dist.npy')
parser.add_argument('--mobility_adj',         default='/mob-adj.npy')
parser.add_argument('--POI_simi',             default='/poi_simi.npy')
parser.add_argument('--landUse_simi',         default='/landUse_simi.npy')

# -----------------------Model-----------------------
parser.add_argument('--embedding_size',          type=int,    default=144)
parser.add_argument('--learning_rate',           type=float,  default=0.01)
parser.add_argument('--weight_decay',            type=float,  default=1e-6)
parser.add_argument('--epochs',                  type=int,    default=2000)
parser.add_argument('--hidden_dim',              type=int,    default=128)
parser.add_argument('--z_dim',                   type=int,    default=24)
parser.add_argument('--beta_start_value',        type=float,  default=1e-3)
parser.add_argument('--beta_end_value',          type=int,    default=1)
parser.add_argument('--beta_n_iterations',       type=int,    default=100000)
parser.add_argument('--beta_start_iteration',    type=int,    default=50000)
parser.add_argument('--temperature',             type=float,  default=1.0)
parser.add_argument('--random_seed',             type=int,    default=42)
parser.add_argument('--mask_ratio',              type=float,  default=0.2)                                                                                                                                               #NY 0.1
parser.add_argument('--random_mask_ratio',       type=float,  default=0.15)                                                                        #NY0.15
parser.add_argument('--alpha',                   type=float,  default=1.0)
parser.add_argument('--beta',                    type=float,  default=1.0)
parser.add_argument('--gamma',                   type=float,  default=1.0)



args = parser.parse_args()

# -----------------------City--------------------------- #

if args.city == 'NY':
    parser.add_argument('--data_path',                    default='./data/data_NY')
    parser.add_argument('--POI_dim',         type=int,    default=26)
    parser.add_argument('--landUse_dim',     type=int,    default=11)
    parser.add_argument('--region_num',      type=int,    default=180)

elif args.city == "Chi":
    parser.add_argument('--data_path',                    default='./data/data_Chi')
    parser.add_argument('--POI_dim',         type=int,    default=26)
    parser.add_argument('--landUse_dim',     type=int,    default=12)
    parser.add_argument('--region_num',      type=int,    default=77)


elif args.city == "SF":
    parser.add_argument('--data_path',                    default='./data/data_SF')
    parser.add_argument('--POI_dim',         type=int,    default=26)
    parser.add_argument('--landUse_dim',     type=int,    default=23)
    parser.add_argument('--region_num',      type=int,    default=175)


args = parser.parse_args()