# Makefile for general visualization

CXX = g++
#CXXFLAGS = -std=c++14 -g -Wall -march=native -fopenmp -fno-omit-frame-pointer -O2 -flto -Wno-sign-compare -fsanitize=address
CXXFLAGS = -std=c++14 -g -O2 -lfftw3 #-Wall -march=native -fopenmp -fno-omit-frame-pointer -flto -Wno-sign-compare -lfftw

LIBFLAGS = `pkg-config --cflags --libs cairo libavformat libavcodec libswresample libswscale libavutil`
BINS = vis_test
OBJ = vis_test.o layer.o scene.o camera.o shapes.o animators.o video_encoder.o png_encoder.o
DEPS = include/vec.h include/scene.h include/layer.h include/camera.h include/shapes.h include/animators.h include/video_encoder.h include/png_encoder.h include/encoder.h

$(BINS): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(BINS) $^ $(LIBFLAGS)
	#./vis_test

%.o: src/%.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(LIBFLAGS)

clean:
	rm -Rf $(BINS) $(OBJ) file*


