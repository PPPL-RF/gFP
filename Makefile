MFEM_PREFIX ?= /usr/local
MFEM_BUILD_DIR ?= $(MFEM_PREFIX)/share/mfem
MFEM_INC_DIR ?= $(MFEM_PREFIX)/include
MFEM_LIB_DIR ?= $(MFEM_PREFIX)/lib
CONFIG_MK = $(MFEM_BUILD_DIR)/config.mk
-include $(CONFIG_MK)

CXXFILE = $(wildcard *.cpp)
OFILE = $(CXXFILE:.cpp=.o)

.PHONY: all clean build clean-build

default: build buildp


%.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

gFP:
build: gfp.o
	echo $(CONFIG_MK)
	$(MFEM_CXX) -ccbin mpicxx -o gFP $< $(MFEMLIB) $(MFEM_LIBS)

buildp: pgfp.o
	$(MFEM_CXX) -ccbin mpicxx -o pgFP $< $(MFEMLIB) $(MFEM_LIBS)

clean:
	rm -f *.o
	rm -f gFP
	rm -f pgFP
