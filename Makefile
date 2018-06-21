build: tsne/bh_sne.pyx \
	   tsne/bh_sne_src/tsne.cpp \
	   tsne/bh_sne_src/sptree.cpp \
	   $(wildcard tsne/bh_sne_src/*.h)
	python setup.py build_ext --inplace

install: build
	python setup.py install

sdist:
	python setup.py sdist

clean:
	rm -rf *.pyc *.so build/ bh_sne.cpp
	rm -rf tsne/*.pyc tsne/*.so tsne/build/ tsne/bh_sne.cpp
