# pycurve

C++ implementation of a discrete elastic deformation energy of curves with python bindings using nanobind.
Implemented for curves in a 2d plane in 3d space. Membrane energy works for general curves in 3d space.

## Compiling
with python interface 
```bash
cd <PYCURVE_DIR>
mkdir build
cd build
cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=ON ..
cmake --build . --config Release
```

If you are using another version of Python than is provided by your system (e.g. if you are using Anaconda), then you need to provide this information to CMake via the option `-DPython_EXECUTABLE=/path/to/python-exectuable`. 
For Anaconda, this path will typically look like `<ANACONDA_DIRECTORY>/bin/python`.

On Linux and macOS, the Python interface is provided as `pycurve.<SOMETHING>.so` in the `python`-subdirectory of your `build`-directory.
If you get an error during build about a failed link you might have to use a dynamic build. For this change [this line](python/CMakeLists.txt#L31) to
```
target_link_libraries(pycurve PUBLIC curve-energy)
```
## Using in Python
To use the Python interface, you first need to either add the path were the resulting the library is stored to your `PYTHONPATH` environment variable, or copy the library to the path of your proeject/where you are executing python.
Then the [Python script](pyexample/evaluate_energy.py) demonstrates the 'usage' of the included functions

## Using in C++
See the [C++ example script](examples/evaluateEnergy.cpp)
