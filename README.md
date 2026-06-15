# AllRec

## AllRec Conda Environment

The environment is defined in `env.yml` and is intended to be created in one step.
It already includes PyTorch CUDA 12.1 wheels, FAISS GPU, vLLM, xFormers, TRL,
PEFT, bitsandbytes, and the other Python packages used by the current AllRec
setup. No local wheel file or extra manual package installation is required.

```bash
conda env create -f env.yml
conda activate AllRec
python -m pip check
```

If an old `AllRec` environment already exists and you want a clean reinstall:

```bash
conda env remove -n AllRec
conda env create -f env.yml
conda activate AllRec
python -m pip check
```
