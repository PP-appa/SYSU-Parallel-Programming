CXX = g++
CXXFLAGS = -std=c++11 -pthread -O2 -Wall
LDFLAGS = -pthread

SRC = src/heated_plate_pthread.cpp
TARGET = bin/heated_plate_pthread

.PHONY: all clean run test run_1 run_2 run_4 run_8

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build successful: $@"

clean:
	rm -f $(TARGET)
	@echo "Clean complete"

run: $(TARGET)
	./$(TARGET) 4 0.001

run_1: $(TARGET)
	./$(TARGET) 1 0.001

run_2: $(TARGET)
	./$(TARGET) 2 0.001

run_4: $(TARGET)
	./$(TARGET) 4 0.001

run_8: $(TARGET)
	./$(TARGET) 8 0.001

test: $(TARGET)
	@echo "=== Testing with 1 thread ==="
	./$(TARGET) 1 0.001
	@echo ""
	@echo "=== Testing with 2 threads ==="
	./$(TARGET) 2 0.001
	@echo ""
	@echo "=== Testing with 4 threads ==="
	./$(TARGET) 4 0.001
