# torch_lfilter

Bring low pass filtering to PyTorch!

This pytorch extension offers a PyTorch alternative for Scipy's
`lfilter` - with gradient tracking.

## CPU tensors only (efficiently...)

Although it's certainly the goal to implement an efficient CUDA
lfilter in C++, for now only the CPU version is implemented in C++.
That said, the implementation is reasonably fast and doing the
filtering on the CPU might be a viable option. Moreover, the
pure-python implementation works on all devices.

## Installation

The library can be installed with pip:

```
pip install torch_lfilter
```

Please note that no pre-built wheels exist. This means that `pip` will
attempt to install the library from source. Make sure you have the
necessary dependencies installed for your OS.

## Dependencies

### Linux

On Linux, having PyTorch installed is often enough to be able install
the library (along with the typical developer tools for your
distribution). Run the following inside a conda environment:

```
conda install pytorch -c pytorch
pip install torch_lfilter
```

### Windows

On Windows, the installation process is a bit more involved as
typically the build dependencies are not installed. To install those,
download **Visual Studio Community 2017** from
[here](https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads).
During installation, go to **Workloads** and select the following
workloads:

- Desktop development with C++
- Python development

Then go to **Individual Components** and select the following
additional items:

- C++/CLI support
- VC++ 2015.3 v14.00 (v140) toolset for desktop

After installation, run the following commands _inside_ a **x64 Native
Tools Command Prompt for VS 2017**, after activating your conda
environment:

```
conda install pytorch -c pytorch
pip install torch_lfilter
```

## License

© Floris Laporte 2020, [GPLv3](license)

