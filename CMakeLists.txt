cmake_minimum_required(VERSION 3.3)
project(linearprogramming)

set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14 -fopenmp -Wall")

add_subdirectory(src)
add_subdirectory(include)

set(SS_CHOLMOD_INC "" CACHE PATH "Cholesky factorization") # No default value
set(SS_CHOLMOD_LIB "" CACHE PATH "Cholesky factorization") # No default value
set(BLAZE_INC "" CACHE PATH "Blaze headers") # No default value
set(BOOST_LIB "" CACHE PATH "Boost library") #No default value

set(SPDLOG_INC "" CACHE PATH "Spdlog library") # No default value

include_directories(SYSTEM ${BLAZE_INC})
include_directories(SYSTEM ${SS_CHOLMOD_INC})
include_directories(SYSTEM ${SPDLOG_INC})

#SuiteSparse
find_library(SS_AMD amd PATHS ${SS_CHOLMOD_LIB} NO_DEFAULT_PATH)
find_library(SS_CAMD camd PATHS ${SS_CHOLMOD_LIB} NO_DEFAULT_PATH)
find_library(SS_CCOLAMD ccolamd PATHS ${SS_CHOLMOD_LIB} NO_DEFAULT_PATH)
find_library(SS_CHOLMOD cholmod PATHS ${SS_CHOLMOD_LIB} NO_DEFAULT_PATH)
find_library(SS_COLAMD colamd PATHS ${SS_CHOLMOD_LIB} NO_DEFAULT_PATH)
find_library(SS_CONFIG suitesparseconfig PATHS ${SS_CHOLMOD_LIB} NO_DEFAULT_PATH)
#SuiteSparse

# Dependecies for BLAZE
find_library(BOOST_SYSTEM boost_system PATHS ${BOOST_LIB} NO_DEFAULT_PATH)
find_library(BOOST_THREAD boost_thread PATHS ${BOOST_LIB} NO_DEFAULT_PATH)
# To read LP files from filesystem
find_library(BOOST_FS boost_filesystem PATHS ${BOOST_LIB})

# Dependecies for parser
# TODO Remove after testing
# TODO Notice that paths are hardcoded remove these
include_directories(SYSTEM /home/satya/libs/glpk/include)
find_library(GLPK_PARSER glpk PATHS "/home/satya/libs/glpk/lib")

include_directories(SYSTEM /home/satya/myLibs/lpParser/include)
find_library(LP_PARSER parser PATHS "/home/satya/myLibs/lpParser/lib")


# TODO Currently testing/interface is done in main.cpp it should be moved to src directory where everything defined for lp
# Idea is to use this as linked library rather than header does it make sense!! lets see
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(linearprogramming main.cpp include/Point.hpp include/Residuals.hpp include/Solver.hpp include/CholeskyLDLTSolver.hpp include/NTScalings.hpp include/KKTUtil.hpp include/BlazeUtil.hpp)

target_link_libraries(${PROJECT_NAME}
		      ${BOOST_SYSTEM} ${BOOST_THREAD} ${BOOST_FS}
		      ${SS_CHOLMOD} ${SS_CONFIG}		      
		      ${SS_AMD} ${SS_CAMD} ${SS_COLAMD} ${SS_CCOLAMD}
			  ${LP_PARSER} ${GLPK_PARSER}
		      m rt
		      pthread hwloc)
		      
install(TARGETS linearprogramming RUNTIME DESTINATION bin)



