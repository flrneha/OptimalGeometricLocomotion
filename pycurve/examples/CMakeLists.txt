project(examples)

add_executable(evaluateEnergy evaluateEnergy.cpp)
target_link_libraries(evaluateEnergy PUBLIC curve-energy_static)
