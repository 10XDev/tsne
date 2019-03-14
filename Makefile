.PHONY: build

build: setup.py tsne/bh_sne.pyx \
       $(wildcard tsne/bh_sne_src/*.h) \
       $(wildcard tsne/bh_sne_src/*.cpp)
	python $< build_ext --inplace

PROFILE_LOCATION=$(shell pwd)/profile

$(PROFILE_LOCATION)/bh_sne.profdata: PROFILE=generate
$(PROFILE_LOCATION)/bh_sne.profdata: setup.py Makefile tsne/bh_sne.pyx \
                               $(wildcard tsne/bh_sne_src/*.h) \
                               $(wildcard tsne/bh_sne_src/*.cpp)
	rm -rf build tsne/*.so *.so $(PROFILE_LOCATION)/*.profraw
	PROFILE=generate python $< build_ext --inplace
	mkdir -p $(PROFILE_LOCATION)
	cd tsne/tests && LLVM_PROFILE_FILE="$(PROFILE_LOCATION)/%p-%m.profraw" py.test
	llvm-profdata merge -output=$@ $(PROFILE_LOCATION)/*.profraw
	rm -rf tsne/*.so *.so build tsne/*.o

build_pgo: setup.py $(PROFILE_LOCATION)/bh_sne.profdata
	PROFILE="$(PROFILE_LOCATION)/bh_sne.profdata" python $< build_ext --inplace

install: setup.py build
	python $< install

sdist: setup.py tsne/bh_sne.pyx \
       $(wildcard tsne/bh_sne_src/*.h) \
       $(wildcard tsne/bh_sne_src/*.cpp)
	python $< sdist

test: build
	cd tsne/tests && time py.test -s -vv

clean:
	make -C tsne/bh_sne_src clean
	rm -rf *.pyc *.so build/ profile/ bh_sne.cpp
	rm -rf tsne/*.pyc tsne/*.so tsne/*.o tsne/build/ tsne/bh_sne.cpp
	rm -rf .pytest_cache/
