# Experimental Pipeline

## Author

Eldar Abraham

## Prerequisites

* Create project home folder `~/.../causal_features/`
* Navigate to project home folder `cd causal_features`
* Create folder `bin/` and save setup files:
    * `setup_project.sh`
    * `setup_anaconda.sh`
    * `setup_drive.sh`
* Add execution permissions `chmod +x bin`
* Create folder `cfg/` and save environment file:
    * `CausalFeatures_env_2.yml`
* Create a Google Drive folders (or ask Boaz to share them):
    * `causal_features/datasets`
    * `causal_features/experiments`
* Get the IDs of the folders you created from URL bar.
    * For example, `1cuoVI7qHkFJZn7yJZQGs45v2hPMFE2qB`
* Get GitHub username of the repository owner (for cloning)
    * `bendboaz`
* Run setup
    ```
    bash bin/setup_project.sh [project_name] [datasets_id] [exeriments_id] [yml_path] [github_username]
    ```
  For example,
    ```
    bash bin/setup_project.sh CausalFeatures 1cuoVI7qHkFJZn7yJZQGs45v2hPMFE2qB 11LQNzvB8NtfyRoCZzrO6TtMxdspVQoxT ~/conda_env_configs/CausalFeatures_env.yml bendboaz
    ```
* Get `user_specific.ini`, change `DATA_DIR` to point to your data folder and save it under under `dev/CausalFeatures/`.

## Project Directory Structure

```
causal_features
──────────────
├── bin
│   ├── setup_project.sh
│   ├── setup_anaconda.sh
│   └── setup_drive.sh
│
├── cfg
│   └── CausalFeatures_env_2.yml
│
└── dev
│   └── CausalFeatures
│   │   ├── run_experiments.py
│   │   ├── ...
│   │   └── user_specific.ini
│
└── data
│   ├── amazon_reviews
│   ├── glove
│   ├── ...
│   └── imdb_proc
│
└── experiments
│   ├── confings
│   ├── finetuned
│   ├── logs
│   ├── more-domain-pairs-again
│   ├── past exeriments
│   ├── per-sample
│   └── pretraining

```

## Running Experiments

```bash
conda activate CausalFeatures
cd ~/.../causal_features
python dev/CausalFeatures/run_experiment.py path/to/config.yml
```
