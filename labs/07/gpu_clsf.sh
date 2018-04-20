# qsub -cwd -pe smp 1 -j y -l mem_free=4G,act_mem_free=4G,h_vmem=8 fashion_masks.sh

export LD_LIBRARY_PATH=/opt/cuda-9.0/lib64/:/opt/cuda/cudnn/7.0/lib64/

cd /net/work/people/auersperger/npfl114/labs/07/
source ../../.gpuvenv/bin/activate

python nsketch_classifier.py --batch_size 128 --arch R-512,R-512,R-512 --epochs 20 
