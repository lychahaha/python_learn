conda create -n jh python=3.5
conda create -f env.yaml


source activate jh
deactivate

conda env list
conda env export > env.yaml

conda env remove -n jh

conda install numpy
conda install numpy=1.13
conda uninstall numpy

conda list

