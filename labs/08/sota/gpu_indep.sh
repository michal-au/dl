# qsub -cwd -pe smp 1 -j y -l mem_free=4G,act_mem_free=4G,h_vmem=8 fashion_masks.sh

export LD_LIBRARY_PATH=/opt/cuda-9.0/lib64/:/opt/cuda/cudnn/7.0/lib64/

cd /net/work/people/auersperger/npfl114/labs/08/sota/
source ../../../.gpuvenv/bin/activate
#source ../../../.gpuvenvold/bin/activate

python tagger_sota_indep.py --rnn_output_process_architecture R-1600 --cle_dim 32 --cnne_filters 32 --epochs 3 --cnne_max 5
