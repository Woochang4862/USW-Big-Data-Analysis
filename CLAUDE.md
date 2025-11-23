# Tech Stack
- Language: Python 3.11+
- ML: scikit-learn, xgboost, tensorflow
- Data: pandas, numpy

# Project Structure
- `final-assignment/src/`: Reusable modules (data, models, ensemble, utils)
- `final-assignment/experiments/`: Executable scripts (experiments, comparison, utils)
- `final-assignment/submissions/`: Submission CSV files
- `final-assignment/legacy/`: Old scripts (reference only, do not edit)
- `final-assignment/config/`: Configuration files
- `final-assignment/results/`: Experiment results

# Commands
- `python experiments/overfitting_experiment.py`: Run overfitting experiment
- `python experiments/ensemble_experiment.py --case stacking`: Run ensemble experiment
- `python experiments/comparison/compare_all_ensemble_methods.py`: Compare ensemble methods
- `python experiments/utils/test.py [submission_file]`: Evaluate submission file

# Code Style
- Use type hints for all functions
- All models inherit from `BaseModel` in `src/models/base.py`
- Register new models in `src/models/factory.py`
- Import from `src/` modules, not from experiment scripts
- Use factory pattern: `create_model(model_name, params)` from `src.models.factory`

# File Organization Rules
- `src/`: Reusable modules imported by other scripts
- `experiments/`: Executable scripts that import from `src/`
- Do not edit files in `legacy/`
- New models: Add class in `src/models/`, register in `factory.py`
- New preprocessing: Add function in `src/data/preprocessing.py`

# Temporary Files
- Files not included in the architecture are considered temporary files
- Temporary files can be created
- Temporary files must be deleted after use
- If a file is expected to be used in the future, it should be added to the architecture and related documentation should be updated

# Module Usage
- Data: `from src.data import load_feature_label_pairs, impute_by_rules`
- Models: `from src.models.factory import create_model`
- Metrics: `from src.utils.metrics import evaluate_model`
- Ensemble: `from src.ensemble.stacking import StackingEnsemble`

# Do Not
- Do not edit files in `legacy/` directory
- Do not duplicate code between `src/` and `experiments/`
- Do not import from experiment scripts in `src/` modules

# Documentation
- Update `ARCHITECTURE.md` when adding new sources or modifying project structure
- Keep `ARCHITECTURE.md` synchronized with code changes

