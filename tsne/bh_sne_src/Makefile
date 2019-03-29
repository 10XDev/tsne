#CXX = clang++
#CFLAGS = -march=haswell -ffast-math -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize  -Rpass-analysis=loop-vectorize
#CFLAGS = -march=haswell -ffast-math -O3

CXX?=g++
CFLAGS?=-ffast-math -O3 -Wall -std=c++14

all: bh_tsne

bh_tsne: main.o tsne.o sptree.o
	$(CXX) $(CFLAGS) $(LDFLAGS) $^ -o $@

sptree.o: sptree.cpp sptree.h
	$(CXX) $(CFLAGS) -c $<

tsne.o: tsne.cpp tsne.h sptree.h vptree.h
	$(CXX) $(CFLAGS) -c $<

main.o: main.cpp tsne.h
	$(CXX) $(CFLAGS) -c $<

clean:
	rm -Rf *.o bh_tsne
