MFEM_BUILD_DIR ?= ..
MFEM_INC_DIR ?= /usr/local/includ
MFEM_LIB_DIR ?= /usr/local/lib
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
-include $(CONFIG_MK)

CXXFILE = $(wildcard *.cpp)
OFILE = $(CXXFILE:.cpp=.o)

.PHONY: all clean build clean-build

default: build


%.o: %.cpp
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< 

gFP:
build: gfp.o
	$(MFEM_CXX) -o gFP $(OFILE) $(MFEMLIB) $(MFEM_LIBS)

clean:
	rm -f *.o
	rm -f gFP
