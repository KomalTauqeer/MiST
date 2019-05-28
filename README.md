# MiST - Machine learning in Single Top

Providing an easy way to utilize state-of-the-art machine learning tools in HEP analyses

## Requirements

To avoid dependency on local software installations, `virtualenv` is used to create a virtual python environment in which any additional python packages are installed using `pip`. The GPU is interfaced with the CUDA API, but in principle the code also runs on CPU with some minor adjustments (`tensorflow-gpu` -> `tensorflow` below). Other than that, only `ROOT` (v6) needs to be installed locally for the instructions below.

## Instructions

1. Connect to the `ekpdeepthought` machine to utilize GPU power:

        ssh ekpdeepthought

2. To make use of the GPU, add the following to your `~/.bashrc` or `~/.profile`, you can also place it outside the function to source it automatically:

        cuda_init() {

            MYCUDAVERSION="cuda-9.0"

            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/$MYCUDAVERSION/lib64:/usr/local/$MYCUDAVERSION/extras/CUPTI/lib64"
            export CUDA_HOME="/usr/local/$MYCUDAVERSION"

        }

    You may also add something like this to automatically use the same `ROOT` version

        if [ "$HOSTNAME" = "ekpdeepthought" ]; then
            . /usr/local/bin/thisroot.sh
        fi

3. Source your bashrc/profile again and call `cuda_init` (if necessary).

4. Clone the repo either using your ssh key:

        git clone git@gitlab.ekp.kit.edu:tH/MiST.git

    OR via https:

        git clone https://gitlab.ekp.kit.edu/tH/MiST.git

5. Create a new `virtualenv` instance (only tested with python2) and `cd` to cloned repo

        virtualenv venv
        source venv/bin/activate
        cd MiST

6. Install all required python packages:

        pip install --upgrade --force-reinstall --no-cache argparse pandas numpy tensorflow-gpu==1.12.0 keras six tqdm scipy scikit-learn matplotlib h5py pydot pympler pandas root_pandas

    Make sure that your `ROOT` installation is working and that the corresponding environment variables are set correctly!

7. The program is started with:

        python run.py [-...]

    Description of additional arguments is provided with:

        python run.py --help

    You can also use a config file which provides all arguments:

        python run.py -c 'config/settings.config'

    Options set with such a config file can be overwritten:

        python run.py -c 'config/settings.config' --epochs 500
        
If the programm crashes during plots, execute the following:

   	echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc