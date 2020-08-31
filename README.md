# gFP


Build
  make MFEM_BUILD_DIR=../mfem MFEM_INC_DIR=../mfem MFEM_LIB_DIR=../mfem

RUN
  gFP
  gFP -pa -d cuda 




File Descriptions
     gfp.cpp: serial Fokker-Planck solver
     pgfp.cpp: parallel Fokker-Planck solver
     gFPOut.cpp: reads out solution from solver as space seperated values
     gFPOut.py: read output of gFPOut.cpp and calculates current, power, efficiency