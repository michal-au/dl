# qsub -cwd -pe smp 1 -j y -l mem_free=4G,act_mem_free=4G,h_vmem=8 uppercase.sh

cd /net/work/people/auersperger/npfl114/labs/03/
source ../../.venv/bin/activate

python mnist_competition.py --epochs 10 --batch_size 100 --cnn

