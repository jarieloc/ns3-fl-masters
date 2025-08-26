# ns3-fl-masters

This framework was adapted from the following creators:
- [Emily Ekaireb][https://github.com/eekaireb/ns3-fl] (The initial work)
- [halsimov][https://github.com/halsimov/ns-fl] (adaptation)
The framework is mostly a personal implementtion for demonstrative purposes and quickly deployable on DTU's HPC servers, for simultaion.

Todo
- Write and clean project folder
- Ref originl and adapted work
- Add `bsub` files and guide for submitting to DTU's HPC node.


# How to run:
Use whatever virtual environment you prefer (conda, venv or something else). This example uses `python venv`.
```
python -m venv .myVenv
source .myVenv/bin/activate
```

Update and install the necessary requirements:
```
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
```

Proceed to install `requirements.txt` or manual paackage management.
```
python -m pip install -r requirements.txt
```

Now you can `cd` into `ns3-fl-network`, and run `ns3`, or `cd` into `flsim` to run a specific manifest/config file.
