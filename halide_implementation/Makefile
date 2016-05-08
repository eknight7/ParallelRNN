# This Makefile's skeleton comes from 15-618, Assignment 1, Spring '16.
# define the C compiler to use
CXX = g++

# define any compile-time flags
CXXFLAGS  = --std=c++11 -g -Wall

APP_NAME=parallelRNN
INCLUDE_DIR=./include
SRC_DIR=src
OBJ_DIR=objs

# define any directories containing header files
INCLUDES = -I/Users/esphinx/halide/include \
  -I/Users/esphinx/halide/tools \
  -I$(INCLUDE_DIR) \
  `pkg-config --cflags-only-I protobuf`

# define library paths
LFLAGS = -L/Users/esphinx/halide/bin

# define any libraries to link into executable:
LIBS = -lHalide -ldl

EXTRA_SCRIPTS = `libpng-config --cflags --ldflags`

# define the C source files
#SRCS = src/main.cpp

#OBJS = $(SRCS:.cc=.o)

default: $(APP_NAME)
	@echo Project compiled!

.PHONY: dirs clean

dirs:
	/bin/mkdir -p $(OBJ_DIR)/

clean:
	/bin/rm -rf $(OBJ_DIR) $(APP_NAME)

SRCS=$(SRC_DIR)/main.cpp
OBJS=$(SRCS:$(SRC_DIR)/.cc=$(OBJ_DIR)/.o)
#OBJS=$(OBJ_DIR)/main.o

#$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
#	$(CXX) $< $(CXXFLAGS) -c -o $@

#$(OBJDIR)/%.o: $(INCLUDE_DIR)/%.cpp
#	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJ_DIR)/main.o: $(INCLUDE_DIR)/CycleTimer.h

$(APP_NAME): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LFLAGS) $(LIBS) $(EXTRA_SCRIPTS)
