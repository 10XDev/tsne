.PHONY: build

build: setup.py tsne/bh_sne.pyx \
       $(wildcard tsne/bh_sne_src/*.h) \
       $(wildcard tsne/bh_sne_src/*.cpp)
	python $< build_ext --inplace

install: setup.py build
	python $< install

sdist: setup.py tsne/bh_sne.pyx \
       $(wildcard tsne/bh_sne_src/*.h) \
       $(wildcard tsne/bh_sne_src/*.cpp)
	python $< sdist

test: build
	cd tsne/tests && py.test -s -vv

clean:
	make -C tsne/bh_sne_src clean
	rm -rf *.pyc *.so build/ bh_sne.cpp
	rm -rf tsne/*.pyc tsne/*.so tsne/*.o tsne/build/ tsne/bh_sne.cpp
	rm -rf .pytest_cache/
