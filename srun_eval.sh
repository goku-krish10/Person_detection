srun -K --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh --container-workdir=`pwd` --container-mounts=/netscratch:/netscratch -p batch --gpus=1 --cpus-per-gpu=2 --job-name="YOLACT Evaluation" ./run_eval.sh
