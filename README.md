# Automatic-Satelitte-Island-Discovery
Project outside of course scope at (BSc) Machine Learning and Data Science education programme. Colab between IGN and DIKU at University of Copenhagen.

## Folder structure
```
.
├── data
│   ├── *.tif
│   ├── ice.dbf
│   ├── ice.prj
│   ├── ice.sbn
│   ├── ice.sbx
│   ├── ice.shp
│   ├── ice.shx
│   ├── ice.xml
│   ├── land.dbf
│   ├── land.prj
│   ├── land.sbn
│   ├── land.sbx
│   ├── land.shp
│   ├── land.shx
│   ├── land.xml
│   ├── island.dbf
│   ├── island.prj
│   ├── island.sbn
│   ├── island.sbx
│   ├── island.shp
│   └── island.shx
│   ├── island.xml
├── src
│   ├── UNet.py
│   ├── preds_analysis.ipynb
|   ├── mosaic_creation_testing
|   |   ├── mosaic_creation_testing.ipynb
|   |   ├── mosaic_creation_testing.py
├── output
│   ├── figs
│   │   ├── *.png
│   ├── predictions
│   │   ├── imgs.npy
│   │   ├── masks.npy
│   │   ├── pred_masks.npy
│   ├── models
│   │   ├── *.pth
│   ├── logs
│   │   ├── *.json
│   ├── tensors
│   │   ├── 0.pt
│   │   ├── 1.pt
│   │   ├── 2.pt
│   │   ├── 3.pt
|  LICENSE
|  README.md
|  requirements.txt
```

## Replication not tested
The code has only been tested on machines with below specifications. The replication-process of the project has not been tested.
### Machine specifications (local)
- OS: Ubuntu 23.04
- CPU: Intel i9-8950HK
- GPU: NVIDIA Quadro P3200M
- RAM: 32 GB
- Storage: 1 TB SSD
- CUDA: 11.8

### Machine specifications (GPU cluster: Slurm)
- GPU: NVIDIA® RTX™ A6000
- CPU: 8x Unspecified
- RAM: 64 GB

## License



## Data
The data is located in the `data` folder. The data consists of 2 different types of data: Shapefiles (`.dbf`, `.prj`, `.sbn`, `.sbx`, `.shp`, `.shx`, `.xml`) and Satellite images (`.tif`). The shapefiles are used to create the masks for the satellite images. Windows of the satellite images are extracted and the corresponding masks are created.
- Satellite images (`.tif`)
- Masks (Shapefiles: `.dbf`, `.prj`, `.sbn`, `.sbx`, `.shp`, `.shx`, `.xml`)
  - Island data
  - Land data
  - Ice data
  - Island data

## Source code
The source code is located in the `src` folder. The source code consists of 2 files and 1 folder:
- `UNet.py`: The UNet model used for the project.
- `preds_analysis.ipynb`: The notebook used for analysing the predictions.
- `mosaic_creation_testing`: The folder containing the code used for making mosaics from non-orth-rectified tiles (`.py` and `.ipynb`). These mosaics where not used in the project, as the orth-rectified satellite images where used instead.

## Output
The output is located in the `output` folder. The output consists of 4 different types of data:
- Figures (`.png`)
- Predictions (`.npy`)
  - Images (`.npy`)
  - Masks (`.npy`)
  - Predicted masks (`.npy`)
- Models (`.pth`)
- Logs (`.json`)
- Tensors (`.pt`: These are stored to avoid having to recompute them for each run)

## How to run
1. Clone the repository.
2. Get hands on the data and place it in the `data` folder.
3. Run the code.
#### CUDA
If you have a CUDA enabled GPU, I recommend using it. I used CUDA 11.8. To install CUDA 11.8 visit the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). To install CUDA 11.8 on Ubuntu 20.04 follow the instructions in the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

#### Virtual environment
I recommend using a virtual environment to run the code. I used `conda` to create a virtual environment -- more specifically `mamba` which is a faster version of `conda`. To create a virtual environment using `mamba` with the required packages (specified in environment.yml) run the following command:
```bash
mamba env -n <env_name> -f environment.yml
```
To activate the virtual environment run the following command:
```bash
conda activate <env_name>
```

#### Requirements (without conda)
To install the requirements (without conda, NOT recommended) run the following command:
```bash
pip install -r requirements.txt
```
### Running the code
1. Open the `UNet.py` file and change the `DATA_PATH` variable to the path of the `data` folder.
2. Run the `UNet.py` file to train the model.
3. Open the `preds_analysis.ipynb` file and change the `DATA_PATH` variable to the path of the `data` folder.
4. Run the `preds_analysis.ipynb` file to analyse the predictions.

## Used libraries
- `numpy`
- `matplotlib`
- `torch`
- `torchvision`
- `tqdm`
- `scipy`
- `shapely`
- `geopandas`
- `rasterio`
- `segmentation_models_pytorch`
- `json`
- `tifffile`
- `pickle`
- `torchmetrics`
- `pathlib`

## Author
- [Johannes Brøns Christensen](mailto:johannes@broens.com), KU-ID: [swd122](mailto:swd122@ku.dk)

## Supervisors
### Main supervisor (from DIKU)
- [François Bernard Lauze](mailto:francois@di.ku.dk), ([GitHub](https://loutchoa.github.io/), [KU-page](https://di.ku.dk/english/staff/?pure=en/persons/200294))

### Co-supervisor (from IGN)
- [Anders Anker Bjørk](mailto:aab@ign.ku.dk) ([KU-page](https://ign.ku.dk/ansatte/geografi/?pure=da/persons/288976)) (provided the target data for the project)

### Other people to thank for their help
- [Ankit Kariryaa](mailto:ak@di.ku.dk)
- [Jonas Kvist Andersen](mailto:joka@ign.ku.dk) ([KU-page](https://ign.ku.dk/ansatte/geografi/?pure=da/persons/779957))
- [Stefan Oehmcke](mailto:stefan.oehmcke@di.ku.dk)
- [Christian Igel](mailto:igel@di.ku.dk)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Useful links
- [Planet](https://www.planet.com/) (satellite image provider for the project)