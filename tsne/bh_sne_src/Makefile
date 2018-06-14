#CXX = clang++
#CFLAGS = -march=haswell -ffast-math -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize  -Rpass-analysis=loop-vectorize
#CFLAGS = -march=haswell -ffast-math -O3

CXX ?= g++
CFLAGS += -std=c++11 -ffast-math -O3 -flto

CFLAGS += -ffunction-sections
LDFLAGS += -Wl,--gc-sections

all: bh_tsne

bh_tsne: main.o tsne.o sptree.o
	$(CXX) $(CFLAGS) $(LDFLAGS) $^ -o $@

sptree.o: sptree.cpp sptree.h
	$(CXX) $(CFLAGS) -c $< -o $@

main.o tsne.o: tsne.cpp tsne.h sptree.h vptree.h
	$(CXX) $(CFLAGS) -c $< -o $@

main.o: main.cpp tsne.h sptree.h vptree.h
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -Rf *.o bh_tsne
