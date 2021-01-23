
## Ragdoll
Ragdoll is a distributed GNN system on GPUs. Currently, it uses DGL for single GPU computation and GCCL for graph communication.


## Prerequisites

### git clone --recursively

### GCCL installation on ${GCCL_HOME}

### python==3.6

### DGL==0.4.3.post2
pip install dgl-cu102==0.4.3.post2

### Pytorch==1.7.1
pip install torch torchvision

## BUILD
./build.sh

## Example

### GCN

```bash
./run_gcn.sh
```

### GIN

```bash
./run_gin.sh
```



