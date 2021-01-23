mkdir -p logs

# mpirun --oversubscribe -np 3 ./scripts/mpi_wrapper.sh python test/test_torch.py TorchTests.test_dispatch_data
mpirun --oversubscribe -np 3 ./scripts/mpi_wrapper.sh python test/test_torch.py
