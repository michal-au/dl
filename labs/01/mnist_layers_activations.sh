# qsub -cwd -pe smp 1 -j y -l mem_free=8G,act_mem_free=8G,h_vmem=12 mnist_layers_activations.py

cd /net/work/people/auersperger/npfl114/labs/01/
source ../../.venv/bin/activate

./mnist_layers_activations.py

