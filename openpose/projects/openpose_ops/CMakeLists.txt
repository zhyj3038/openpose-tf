###
# Copyright (c) 2017, 申瑞珉 (Ruimin Shen)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#	list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#	this list of conditions and the following disclaimer in the documentation
#	and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})
file(GLOB_RECURSE SOURCE_LIST ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
add_library(${PROJECT_NAME} SHARED ${SOURCE_LIST})
add_definitions(-DCMAKE_BINARY_DIR=\"${CMAKE_BINARY_DIR}\")

FIND_PACKAGE(OpenCV REQUIRED)
if(OpenCV_FOUND)
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif()

set(CNPY_DIR ${CMAKE_SOURCE_DIR}/third_party/cnpy)
if(EXISTS ${CNPY_DIR})
	include_directories(${CNPY_DIR})
	target_link_libraries(${PROJECT_NAME} cnpy)
	add_definitions(-DENABLE_NPY)
endif()

include_directories(${CMAKE_SOURCE_DIR}/libs/openpose)
target_link_libraries(${PROJECT_NAME} openpose)

get_filename_component(OP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${OP_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set(TF_OPS_DIR "~/lib" CACHE PATH "")
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${TF_OPS_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TF_OPS_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${TF_OPS_DIR})
foreach(CMAKE_CONFIGURATION_TYPE ${CMAKE_CONFIGURATION_TYPES})
	string(TOUPPER ${CMAKE_CONFIGURATION_TYPE} CMAKE_CONFIGURATION_TYPE)
	set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${CMAKE_CONFIGURATION_TYPE} ${TF_OPS_DIR})
	set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${CMAKE_CONFIGURATION_TYPE} ${TF_OPS_DIR})
	set_target_properties(${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${CMAKE_CONFIGURATION_TYPE} ${TF_OPS_DIR})
endforeach()
