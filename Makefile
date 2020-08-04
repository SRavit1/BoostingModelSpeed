all: clean
	mkdir build && cd build && cmake .. && make
clean:
	rm -rf build
