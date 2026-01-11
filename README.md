# Deep Learning-Based Detection of Active, Fallow and Abandoned Land in Khajura Rural Municipality, Nepal

End-to-end pipeline for mapping active, fallow, abandoned, and other land cover
classes using Sentinel-2 imagery. It repository includes data
extraction, EDA, rule-based algorithm for fallow land detection, BiLSTM modeldeep learning training/inference,
evaluation, and visualization code. 



## Google Earth Engine (GEE)

Data preparation and inference in this project require access to Google Earth
Engine (GEE). Please keep your GEE Project ID configured either in the
project configuration (for example under the `configs/` files) or set it as an
environment variable named `GEE_PROJECT_ID` so scripts and notebooks can access
GEE without manual edits.

For Colab or other remote environments, export the environment variable before
running data preparation or inference, e.g.:

```
export GEE_PROJECT_ID=your-gee-project-id
```

## Citation

If you use this work, please cite:
```
bibtex
@misc{khajura_fallow_land_detection_2025,
  title   = {Deep Learning-Based Detection of Active, Fallow and Abandoned Land: Khajura Municipality, Nepal},
  author  = {Tandukar, Nishon and Acharya, Bishnu and Rawat, Bikram and Acharya, Rishi and Sharma, Sandesh and Thapa, Saruna and Basi, Nishim},
  year    = {2025},
  url     = {https://github.com/KU-GEOI-525-2023/fallow-land-detection-khajura-bilstm},
  note    = {GitHub repository, accessed 2026-01-11}
}
```

## TODO

- [x] Jan 11, 2025 - open-sourced the full codebase and configuration files
- [ ] Draft and publish the technical paper describing methodology and results

## Notebook

- Overall Pipeline notebook: notebooks/basic_pipeline.ipynb
- Open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KU-GEOI-525-2023/fallow-land-detection-khajura-bilstm/blob/master/notebooks/basic_pipeline.ipynb)
