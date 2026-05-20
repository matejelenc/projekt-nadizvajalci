# projekt-nadizvajalci
This repo contains code for **Geographic analysis of MMC articles** as a part of the UOZP course at FRI, UNI LJ.

# Usage
## Environment
First ensure, **uv** is installed, which can be done by running:
```bash
./scripts/env.sh
```
This creates the `uv` environment and install dependecies used by all parts of the project. The instructions below showcase how to install extra dependencies for each part, to avoid downloading large libraries such as **torch** or **transformers** for simply running the web app. However, if you wish to install all packages, run:
```bash
uv sync --all-extras
```

### Preprocessing
To install the MMC articles and necessary supplementary files for preprocessing to [./data/raw](./data/raw), run:
```bash
./scripts/download.sh
```
The NLP preprocessing pipeline also requires additional installation of packages. The pipeline should ideally be ran on a computer that supports **CUDA** - for this specific project, the [Arnes HPC cluster](https://www.sling.si/en/arnes-hpc-cluster/), which is managed by **slurm**, was used. If using Arnes HPC cluster, the required dependecies can be installed by running:
```bash
uv sync --extra nlp
```
To submit an `sbatch` job for preprocessing, run:
```bash
sbatch ./scripts/preprocess.sh
``` 
Or if using a custom stack, ensure you have correct **torch** and **CUDA** versions installed, and run the script directly:
```bash
uv run python -u -m geonews.nlp.preprocess
```

### Database
To run the database, ensure **docker** is installed, which can be done by following the official [instructions](https://docs.docker.com/engine/install/). Running the database and associated functions requires additional dependecies which can be added using:
```bash
uv sync --extra app
```
After obtaining preprocessed data, which should be saved as separate `jsonl` files under [./data/clean](./data/clean), copy [.env_example](.env_example) to [.env](.env) and change settings appropriately. The Neo4j database can be created, and data exported to it, using:
```bash
./scripts/database.sh
```
After initialization, the database can be started/stopped using `docker compose`.

### Development
To add code, refer to the internal **geonews** module and follow **google-style** docstrings. Examples, short scripts and other tools, such as a simple [annotation app](./tools/annotate.py), can be found under [./tools](./tools).