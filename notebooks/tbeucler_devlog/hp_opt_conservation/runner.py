import sherpa
import sherpa.schedulers
import argparse

from utils import build_directory

parser = argparse.ArgumentParser()

# sherpa SGE params
parser.add_argument('--max_concurrent',help='Number of concurrent processes',type=int, default=3)
parser.add_argument('-P',help="Specifies the project to which this  job  is  assigned.",default='arcus_gpu.p')
parser.add_argument('-q',help='Defines a list of cluster queues or queue instances which may be used to execute this job.',default='arcus.q')
parser.add_argument('-l', help='the given resource list.',default="hostname=\'(arcus-1|arcus-2|arcus-3|arcus-4|arcus-5|arcus-6|arcus-7|arcus-8|arcus-9|arcus-10)\'")
parser.add_argument('--env', help='Your environment path.',default='/home/jott1/Projects/SHERPA_EX/.profile',type=str)
parser.add_argument('--alg',default='local',type=str, choices=['local', 'random', 'bayes'])
parser.add_argument('--sch',default='local',type=str, choices=['local', 'sge'])
parser.add_argument('--gpus',default='0,1,2,3',type=str)

# ---------------- Important parameters -------------------------
parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'weak_loss'], help='What to run?')
parser.add_argument('--net_type', type=str, default='normal', choices=['normal', 'conservation'], help='What to run?')
parser.add_argument('--data', type=str, choices=['fluxbypass_aqua', 'land_data', '8col', '32col'])

# params okay left as defaults
parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
parser.add_argument('--data_dir', type=str, default='/baldig/chemistry/earth_system_science/')
parser.add_argument('--max_dense_layers', type=int, default=8, help='Max dense layers allowed')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs used for training')
parser.add_argument('--patience', type=int, default=10, help='How long to wait for an improvement')

FLAGS = parser.parse_args()


# Define Hyperparameter ranges
parameters = [
    sherpa.Continuous('alpha', [0., 1]),
    sherpa.Continuous('dropout', [0., 0.5]),
    sherpa.Continuous('lr', [0.00001, 0.01]),
    sherpa.Continuous('leaky_relu', [0., 0.5]),
    sherpa.Ordinal('batch_norm', [1, 0]),
    sherpa.Discrete('num_layers', [2, FLAGS.max_dense_layers]),
]

parameters.extend([
    sherpa.Discrete('layer_{}'.format(i), [32, 512]) for i in range(FLAGS.max_dense_layers)
])

dict_flags = vars(FLAGS)

for arg in dict_flags:
    parameters.append(sherpa.Choice(name=arg, range=[dict_flags[arg] ]))

if FLAGS.alg == 'local':
    from default import default_params
    default_params.update(dict_flags)
    algorithm = sherpa.algorithms.LocalSearch(default_params)
elif FLAGS.alg == 'bayes':
    algorithm = sherpa.algorithms.GPyOpt(max_num_trials=100)
else:
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=10*FLAGS.max_concurrent)

# The scheduler
if FLAGS.sch == 'sge':
    opt = '-N MNISTPBT -P {} -q {} -l {} -l gpu=1'.format(FLAGS.P, FLAGS.q, FLAGS.l)
    scheduler = sherpa.schedulers.SGEScheduler(environment=FLAGS.env, submit_options=opt)
else:
    scheduler = sherpa.schedulers.LocalScheduler(resources=[FLAGS.gpus])

output_path = 'SherpaResults/{data}/{net_type}_{loss_type}/output/'.format(data=FLAGS.data, net_type=FLAGS.net_type, loss_type=FLAGS.loss_type)
models_path = 'SherpaResults/{data}/{net_type}_{loss_type}/Models/'.format(data=FLAGS.data,net_type=FLAGS.net_type, loss_type=FLAGS.loss_type)

build_directory(output_path)
build_directory(models_path)

# Running it all
sherpa.optimize(algorithm=algorithm,
                scheduler=scheduler,
                parameters=parameters,
                lower_is_better=True,
                filename="main.py",
                max_concurrent=FLAGS.max_concurrent,
                output_dir=output_path)
