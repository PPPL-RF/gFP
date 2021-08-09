MFEM_PREFIX ?= /home/shiraiwa/sandbox
MFEM_BUILD_DIR ?= $(MFEM_PREFIX)/share/mfem
MFEM_INC_DIR ?= $(MFEM_PREFIX)/include
MFEM_LIB_DIR ?= $(MFEM_PREFIX)/lib
CONFIG_MK = $(MFEM_BUILD_DIR)/config.mk
-include $(CONFIG_MK)

CXXFILE = $(wildcard *.cpp)
OFILE = $(CXXFILE:.cpp=.o)

.PHONY: all clean build clean-build

default: build buildp buildo buildTD


%.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

gFP:
build: gfp.o
	echo $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o gFP $< $(MFEMLIB) $(MFEM_LIBS)

buildp: pgfp.o
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o pgFP $< $(MFEMLIB) $(MFEM_LIBS)

buildo: gFPOut.o
	echo $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o gFPOut $< $(MFEMLIB) $(MFEM_LIBS)

buildTD: gFPTD.o
	echo $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) -o gFPTD $< $(MFEMLIB) $(MFEM_LIBS)

clean:
	rm -f *.o
	rm -f gFP
	rm -f pgFP
	rm -f gFPOut
	rm -f gFPTD




#7/15/21 Changed -ccbin mpicxx to $(MFEM_LINK_FLAGS) for build, buildp, buildo, buildTD
