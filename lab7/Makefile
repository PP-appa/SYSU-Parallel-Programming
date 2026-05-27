CXX := mpic++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra
CXX_PTHREAD := g++
PTHREAD_FLAGS := -O3 -std=c++17 -Wall -Wextra -pthread

SRC_MPI := src/mpi_fft.cpp
BIN_MPI := bin/mpi_fft
SRC_HP := src/heated_plate_parallel_for.cpp
BIN_HP := bin/heated_plate_parallel_for

all: $(BIN_MPI) $(BIN_HP)

$(BIN_MPI): $(SRC_MPI)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_HP): $(SRC_HP)
	$(CXX_PTHREAD) $(PTHREAD_FLAGS) $< -o $@

run: $(BIN_MPI)
	mpirun -np 4 $(BIN_MPI) 4096 20

clean:
	rm -f $(BIN_MPI) $(BIN_HP)
