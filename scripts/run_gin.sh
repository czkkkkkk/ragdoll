

mkdir -p logs
rm logs/* -rf

n=$1
comm=$2
dataset=$3
dataset_dir=$4
gccl_config_dir=$5


display_usage() {
    echo "Usage: $0 [n] [comm] [dataset] [dataset_dir] [gccl_config_dir]"
    echo "    n: the number of GPUs to run"
    echo "    comm: the communication pattern. {dgcl, alltoall, replication, swap, nocomm}"
    echo "    dataset: the dataset to run. {webgoogle, reddit, wiki-talk, com-orkut}"
    echo "    dataset_dir: the directory of dataset"
    echo "    gccl_config_dir: the directory storing gccl config"
    echo "\$dataset_dir/\$dataset/graph.txt must exist"
    echo "For example, $0 8 dgcl webgoogle \$HOME/datasets \$HOME/gccl/configs"
}
if [ "$#" -ne 5 ]; then
    display_usage
    exit 1
fi

# check the communication pattern
# comm \in {dgcl, alltoall, replication}
if [ "$comm" != "dgcl" ] && [ "$comm" != "alltoall" ] && [ "$comm" != "replication" ] && [ "$comm" != "swap" ] && [ "$comm" != "nocomm" ]; then
    echo "comm must in {dgcl, alltoall, replication, swap, nocomm}"
    display_usage
    exit 1
fi
if [ "$comm" == "dgcl" ]; then
   export GCCL_COMM_PATTERN=GREEDY 
fi
if [ "$comm" == "alltoall" ]; then
   export GCCL_COMM_PATTERN=ALLTOALL
fi

# check dataset
# dataset \in {webgoogle, reddit, wiki-talk, com-orkut}

if [ "$dataset" != "webgoogle" ] && [ "$dataset" != "reddit" ] && [ "$dataset" != "wiki-talk" ] && [ "$dataset" != "com-orkut" ]; then
    echo " Dataset must in {webgoogle, reddit, wiki-talk, com-orkut} "
    display_usage
    exit 1
fi

base_dir=$dataset_dir/$dataset
input_graph=$base_dir/graph.txt
cached_dir=$base_dir

if [ "$comm" == "replication" ]; then
    input_graph=$dataset_dir/partition-$n-2hop/graph.txt
    cached_dir=$cached_dir/partition-$n-2hop
    n=1
fi

if [ "$dataset" == "reddit" ]; then
    feat_size=602
    n_hidden=256
elif [ "$dataset" == "com-orkut" ]; then
    feat_size=128
    n_hidden=128
else
    feat_size=256
    n_hidden=256
fi

export GCCL_CONFIG=${gccl_config_dir}/gpu${n}.json
export GLOG_log_dir=./logs


layers=2
python examples/gin/train.py --world_size $n  \
  --input_graph=${graph} \
  --num_layers=${layers} \
  --num_mlp_layers=2\
  --hidden_dim=${n_hidden} \
  --feat_size=${feat_size} \
  --cached_dir=${cached_dir}
