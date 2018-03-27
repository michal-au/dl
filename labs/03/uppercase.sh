# qsub -cwd -pe smp 1 -j y -l mem_free=4G,act_mem_free=4G,h_vmem=8 uppercase.sh

cd /net/work/people/auersperger/npfl114/labs/03/
source ../../.venv/bin/activate

python uppercase.py --epochs 10 --batch_size 100 --window 7 --hidden_layers 2 --hidden_layer_sizes 100 40 --hidden_layer_activations relu relu 
#python uppercase.py --epochs 1 --batch_size 100 --window 7 --hidden_layers 1 --hidden_layer_sizes 50 --hidden_layer_activations relu 
