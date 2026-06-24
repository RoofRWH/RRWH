# Public RRWH Workflow

This folder contains the reusable, public pieces of the rainfall and rooftop water harvesting workflow.

## Included files

- `config/paths.yaml`: example input and output paths using generic relative locations
- `config/parameters.yaml`: default model and distribution settings
- `notebooks/02_Data_Preprocessing_public.ipynb`: public preprocessing notebook
- `notebooks/RWH_distribution_fitting_public.ipynb`: public distribution-fitting notebook
- `notebooks/RWH_distributions_analysis_public.ipynb`: simplified public rainfall analysis notebook
- `notebooks/07_Final_Policy_Outputs_public.ipynb`: public policy summary notebook
- `src/dist_fitting.py`: rainfall distribution fitting helpers, including the full candidate list and custom `gb2`/`dagum` support
- `src/imd_io.py`: dataset loading helpers
- `src/preprocessing.py`: configuration, loading, and regridding helpers
- `src/rwh_model.py`: core RWH calculations
- `src/spatial_utils.py`: spatial helper functions
- `src/validation.py`: validation metrics and quantile mapping helpers

## Not included

- Private hardcoded study-area paths were removed from the copied config.

## Typical workflow

1. Open `notebooks/02_Data_Preprocessing_public.ipynb` first to produce rainfall statistics.
2. Fit distributions with `notebooks/RWH_distribution_fitting_public.ipynb` or `notebooks/RWH_distributions_analysis_public.ipynb`.
3. Run `notebooks/07_Final_Policy_Outputs_public.ipynb` to build the district summary table.
4. Set your AOI, CHIRPS, population, and roof raster paths in `config/paths.yaml`.
5. Tune model settings in `config/parameters.yaml`.
6. Use `src/imd_io.py` and `src/preprocessing.py` to load and align data.
7. Use `src/dist_fitting.py` to fit rainfall distributions for each pixel.
8. Use `src/rwh_model.py` and `src/validation.py` to generate RWH outputs and assess results.

## Notes

- This is a code-first public workflow and now includes the public notebooks in `notebooks/`.
- The `src/` package already includes `__init__.py` and a small runner script at the workflow root.
