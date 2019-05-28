# imports from standard python
from __future__ import print_function
import random
import sys
import argparse
import ConfigParser as configparser
import subprocess

# imports from local packages

# imports from pip packages
from keras.backend.tensorflow_backend import set_session

# imports from MiST
from MiST import utilis
from MiST import globaldef as gl


def init_tf(ncores, verbose):

    # ugly, but importing at the beginning will screw up the argparser
    print('--- Starting Tensorflow...')
    import tensorflow as tf

    # some custom settings, this needs to be done before anything else
    print('--- Set custom settings...')

    # create tf setting instance
    config = tf.ConfigProto()

    # limit memory ussage
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # allow memory groth which will prevent cancelling processes
    config.gpu_options.allow_growth=True

    # automatically select device if selected is not found
    # config.gpu.options.allow_soft_placement(True)

    # limit the number of GPU cores occupied
    ncores_total = 4
    tmp_listofcores = range(0,ncores_total)
    tmp_listofcores_selected = []

    # don't use busy devices or devices without free memory (experimental)
    autoselect = True
    if autoselect:
        # check for status of devices
        print('--- Looking for free devices...')
        cmd_monitoring = 'nvidia-smi --format=csv,noheader,nounits'
        popen_cores = subprocess.Popen(cmd_monitoring + ' --query-gpu=utilization.gpu',
                                       shell=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        popen_mem = subprocess.Popen(cmd_monitoring + ' --query-gpu=memory.free',
                                     shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)

        res_cores_stdout, res_cores_stderr = popen_cores.communicate()
        res_mem_stdout, res_mem_stderr = popen_mem.communicate()

        if (res_cores_stderr != '') or (res_mem_stderr != ''):
            print('### something went wrong with nvidia-smi, better check or disable autoselect feature!:')
            print(res_cores_stderr)
            print(res_mem_stderr)
            sys.exit(1)

        status_cores = res_cores_stdout.replace('\n',',')[:-1].split(',')
        status_mem = res_mem_stdout.replace('\n',',')[:-1].split(',')

        status_cores = [int(i) for i in status_cores]
        status_mem = [int(i) for i in status_mem]

        threshold_max_core_usage = 50
        threshold_min_mem = 2000

        for i in tmp_listofcores:
            if threshold_max_core_usage < status_cores[i]:
                print('\t--- Skipping device %i due to usage of %i%% (max %i%%)' %(i, status_cores[i], threshold_max_core_usage))
                continue
            if threshold_min_mem > status_mem[i]:
                print('\t--- Skipping device %i due to only %i MB free memory (min %i MB required)' %(i, status_mem[i], threshold_min_mem))
                continue
            tmp_listofcores_selected.append(i)

    else:
        while ncores > len(tmp_listofcores_selected):
            tmp_selected = random.choice(tmp_listofcores)
            if tmp_selected not in tmp_listofcores_selected:
                tmp_listofcores_selected.append(tmp_selected)
        tmp_listofcores_selected.sort()

    cores_string = ",".join(str(i) for i in tmp_listofcores_selected)
    config.gpu_options.visible_device_list = cores_string

    print('--- Using core(s) ' + cores_string)

    # additional log information
    if verbose:
        config.log_device_placement=True

    # feed to tf
    set_session(tf.Session(config=config))

    print('--- Settings done')


def options(argv=None):
    # Do argv default this way, as doing it in the functional declaration sets it at compile time.
    if argv is None:
        argv = sys.argv

    # Parse any conf_file specification
    # We make this parser with add_help=False so that
    # it doesn't parse -h and print help.
    conf_parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False)

    conf_parser.add_argument('-c','-C','--config',
                        metavar='FILE',
                        help='(relative) path to an additional config file')

    args, remaining_argv = conf_parser.parse_known_args()

    defaults = {}

    if args.config:
        args_from_config = configparser.SafeConfigParser()
        args_from_config.read([args.config])
        for i in args_from_config.sections():
            defaults.update(dict(args_from_config.items(i)))

    # Parse rest of arguments
    # Don't suppress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser]
        )
    parser.set_defaults(**defaults)

    parser.add_argument('-t','-T','--type',
                        help='type to use')

    parser.add_argument('-m','-M','--method',
                        help='mva method to use')

    parser.add_argument('-s','-S','--signal',
                        help='signal file for training in training mode')

    parser.add_argument('-b','-B','--background',
                        help='background file for training in training mode')

    parser.add_argument('-s_t','-S_t','--signal_test',
                        help='signal file for testing in training mode')

    parser.add_argument('-b_t','-B_t','--background_test',
                        help='background file for testing in training mode')

    parser.add_argument('--multi_train',
                        help='multiclassification training files')

    parser.add_argument('--multi_test',
                        help='multiclassification testing files')

    parser.add_argument('--reco_samples',
                        help='samples for reco')

    parser.add_argument('-g','-G', '--gpucores', '--GPU','--GPUS',
                        type=int,
                        help='how many GPU cores to be used')

    parser.add_argument('-i','-I','--input',
                        help='input file for eval mode')

    # -t and -T are now used for the type
    parser.add_argument('--tree',
                        help=' tree name inside the root file including path')

    parser.add_argument('--mva_path',
                        help='input path of training')

    parser.add_argument('--batchsize',
                        type=int,
                        help='batchsize')

    parser.add_argument('-e','-E','--epochs',
                        type=int,
                        help='training epochs')

    parser.add_argument('--validationsize',
                        type=float,
                        help='fraction of data used for validation')

    parser.add_argument('--plotvars',
                        action='store_true',
                        help='make plots for input variables')

    parser.add_argument('--print_options',
                        action='store_true',
                        help='print all options')

    parser.add_argument('-v','-V','--verbose',
                        action='store_true',
                        help='increase level of warnings and provide additional debug output')

    parser.add_argument('-d','-D','--debug',
                        action='store_true',
                        help='when something unexpected happened...')

    args = parser.parse_args(remaining_argv)

    # turn arguments into dictionary
    arg = vars(args)

    # convert arguments with multiple entries from strings into proper lists
    for key in arg:
        if isinstance(arg[key], basestring):
            if key == 'variables':
                arg[key] = utilis.resolve_config_string(arg[key],False)
            else:
                arg[key] = utilis.resolve_config_string(arg[key],True)
            if ' ' in arg[key]:
                arg[key] = utilis.conv_str_to_list(arg[key], ' ')

    # quick check for lists and fix if not
    for i in ['variables', 'weights', 'signal', 'signal_test', 'background', 'background_test', 'input', 'reco_samples']:
        if isinstance(arg[i], basestring):
            arg[i] = [arg[i]]

    if args.print_options:
        print()
        print('-'*gl.screenwidth)
        print('--- Options')
        print('-'*gl.screenwidth)
        for i in arg:
            print('--- {:20s} {:s}'.format(i, str(arg[i])))
        print('-'*gl.screenwidth)
        print()

    return arg
