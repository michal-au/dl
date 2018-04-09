# qsub -cwd -pe smp 1 -j y -l mem_free=4G,act_mem_free=4G,h_vmem=8 fashion_masks.sh

export LD_LIBRARY_PATH=/opt/cuda-9.0/lib64/:/opt/cuda/cudnn/7.0/lib64/

cd /net/work/people/auersperger/npfl114/labs/05/
source ../../.gpuvenv/bin/activate

python fashion_masks.py --epochs 60 --batch_size 100 --encoder C-64-3-1-same --classification-decoder C-64-3-1-same,M-2-2,D-0.1,C-64-3-1-same,M-2-2,D-0.3,F,R-256,D-0.5,R-64 --mask-decoder C-64-3-1-same,C-32-5-1-same,D-0.1,C-16-7-1-same,D-0.15,C-10-9-1-same,C-10-1-1-same

