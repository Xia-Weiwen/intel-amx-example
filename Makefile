CFLAGS = -g -march=native -fno-strict-aliasing
CC = g++

objects = int8-gemm-small int8-gemm-large bf16-gemm-small bf16-gemm-large
all: $(objects)

$(objects): %: %.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	-rm -rf $(objects)

.PHONY: clean