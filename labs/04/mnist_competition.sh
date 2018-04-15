# qsub -cwd -pe smp 1 -j y -l mem_free=4G,act_mem_free=4G,h_vmem=8 uppercase.sh

cd /net/work/people/auersperger/npfl114/labs/04/
source ../../.venv/bin/activate

python mnist_competition.py --epochs 100 --batch_size 100 --cnn CB-10-3-2-same,M-3-2,F,D-0.5,R-100

