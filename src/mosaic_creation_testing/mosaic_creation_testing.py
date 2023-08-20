######################################################################
#                                                                    #
#  Note: THIS SCRIPT DID NOT TURN OUT TO BE USEFUL FOR THE PROJECT!  #
#                                                                    #
######################################################################

## Purpose: Preprocess the data for the deep learning model by creating a mosaic of the images and a mask of the islands.
# Author: Johannes Brøns Christensen
# License: MIT License

# %% Imports
import rasterio, os, torch, numpy as np, matplotlib.pyplot as plt, torchgeo, cv2, matplotlib.patches as patches, geopandas as gpd
from torchgeo.datasets import RasterDataset
from torch.utils.data import Dataset, DataLoader
from rasterio.merge import merge
from pathlib import Path
from shapely.geometry import shape, Polygon, MultiPolygon, mapping, Point, LineString, box
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnums
from rasterio.crs import CRS
# Nearest neighbor interpolation
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
from tqdm import tqdm

# %% Constants


# %% Paths and directories
Path('output').mkdir(parents=True, exist_ok=True)
MOSAIC_OUT_PATH = os.path.join('output', 'mosaic.tif')
MASK_OUT_PATH = os.path.join('output', 'mask.tif')
mosaic_out_path = os.path.join('output', 'mosaic.tif')
mask_out_path = os.path.join('output', 'mask.tif')
HOME = os.path.expanduser('~')
GTIFFs_PATH = f"{HOME}/Documents/summer_project/tifs"
tiles = os.listdir(GTIFFs_PATH)

MULTI_POLYGON_PATH = f"{HOME}/Documents/summer_project/data/Masks/islands_allGRL_SDFI_50K/islands_allGRL_SDFI_50K.shp"

PATH_TO_FIGS = os.path.join('output', 'figs')
os.makedirs(PATH_TO_FIGS, exist_ok=True)

# %% Device
DEVICE = torch.device(device=('cuda' if torch.cuda.is_available() else 'cpu'))
print(f'Using device: {DEVICE}')
if torch.cuda.is_available():
    GPU_COUNT = torch.cuda.device_count()
    gpu_count_str = "Number of available GPU's : "+ str(GPU_COUNT)
    print(gpu_count_str)
    print(f'GPU properties: {torch.cuda.get_device_properties(DEVICE)}')

# %% RasterDataset class
class custom_raster_set(RasterDataset):
    is_image = True
    separate_files = True
    all_bands: list[str] = ['Blue', 'Green', 'Red', 'NIR']
    rgb_bands: list[str] = ['Red', 'Green', 'Blue']
    # crs = EPSG:32621
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root=root)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.mosaic_out_path = MOSAIC_OUT_PATH
        self.mask_out_path = MASK_OUT_PATH
        if self.crs is None:
            # Calculate the CRS of highest frequency
            self.crs = self.calculate_crs()
        self.bands: list[str] = self.rgb_bands
        self.filepaths: list[str] = []
        for f in os.listdir(root):
            self.filepaths.append(os.path.join(self.root, f))
        # Sort self.filepaths
        self.filepaths.sort()
        self.worst_res = self.get_worst_res()

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        with rasterio.open(filepath) as ds:
            img = torch.from_numpy(ndarray=ds.read().astype(np.float64)).to(device=DEVICE)
        return img

    def __len__(self):
        return len(self.filepaths)

    def calculate_crs(self, verbose:bool=False):
        # Calculate the CRS of highest frequency
        crs_list = [rasterio.open(fp=filepath).crs for filepath in self.filepaths]
        unique_crs = list(set(crs_list))
        crs_count = {crs: crs_list.count(crs) for crs in crs_list}
        maximal_crs = max(crs_count, key=crs_count.get)
        if crs_count[maximal_crs] < len(self.filepaths)/2:
            print(f'Warning: The CRS {maximal_crs} is not the CRS of the majority of the files. ')
            print(f'The CRS of the majority of the files is {max(crs_count, key=crs_count.get)}. ')
        if verbose:
            print(f'Number of unique CRS: {len(unique_crs)}')
            print(f'CRS of highest frequency: {maximal_crs}')
        return maximal_crs

    def get_worst_res(self):
        # Get the worst resolution
        worst_res = max([rasterio.open(fp=filepath).res for filepath in self.filepaths])
        return worst_res

    def make_mosaic(self):
        raster_files:list[str] = self.filepaths
        raster_to_mosaic = []
        for filepath in raster_files:
            with rasterio.open(filepath, mode="r") as src:
                transform, width, height = calculate_default_transform(src.crs, self.crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': self.crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                with rasterio.open(fp=filepath, mode="r+", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=self.crs,
                            resampling=Resampling.nearest
                        )
            raster_to_mosaic.append(filepath)
        # Open the reprojected files and get the mosaic
        mosaic, output_transform = merge(datasets=raster_files, resampling=Resampling.lanczos, nodata=0, dtype=rasterio.uint16, target_aligned_pixels=True, indexes=[1, 2, 3, 4], dst_kwds={'crs': self.crs}, method='last')

        # Get the metadata of the mosaic
        output_meta = src.meta.copy()
        output_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output_transform,
            "crs": self.crs,
            "count": mosaic.shape[0],
            "nodata": 0,
            "dtype": rasterio.uint16,
        })
        with rasterio.open(fp=self.mosaic_out_path, mode="w", **output_meta) as mosaic_file:
            mosaic_file.write(mosaic, indexes=[1, 2, 3, 4])
        return self.mosaic_out_path


    def make_mask(self, multipolygon_path=MULTI_POLYGON_PATH, shiftpix:tuple=(0,0)):
        # Open the mosaic
        with rasterio.open(fp=self.mosaic_out_path, mode="r", crs=self.crs, driver='GTiff') as mosaic:
            # Move the multipolygon to the right coordinate reference system
            multi_polygon = gpd.read_file(filename=MULTI_POLYGON_PATH).to_crs(crs=self.crs)
            # Get the bounds of the mosaic
            bounds = mosaic.bounds
            # Get the bounds of the multipolygon
            multi_polygon_bounds = multi_polygon.bounds
            # Rasterize the multipolygon
            shapes = ((geom, 1) for geom in multi_polygon.geometry)
            mask = rasterio.features.rasterize(shapes=shapes, out_shape=mosaic.shape, transform=mosaic.transform, fill=0, all_touched=False, dtype=rasterio.uint8)
            # Get the mask's metadata
            mask_meta = mosaic.meta.copy()
            # Update the metadata
            mask_meta.update({
                "driver": "GTiff",
                "height": mask.shape[0],
                "width": mask.shape[1],
                "transform": mosaic.transform,
                "count": 1,
                "dtype": rasterio.uint8,
                "nodata": 0,
                "crs": self.crs
            })
            # Shift the mask
            mask = np.roll(mask, shift=shiftpix, axis=(0,1))
            # Write the mask
            with rasterio.open(fp=self.mask_out_path, mode="w", **mask_meta) as dest:
                dest.write(mask, 1)
            return self.mask_out_path




    def update_mosaic(self, force=False, shiftpix:tuple=(0,0)):
        if os.path.exists(path=self.mosaic_out_path) and os.path.exists(path=self.mask_out_path) and not force:
            print('Mosaic and mask already exist')
        else:
            mosaic_out_path = self.make_mosaic()
            mask_out_path = self.make_mask(shiftpix=shiftpix)
        return self.mosaic_out_path, self.mask_out_path


    def get_bounds(self):
        # Get the bounds of the mosaic
        with rasterio.open(fp=self.mosaic_out_path) as mosaic:
            bounds = mosaic.bounds
        return bounds

    def coordinates_to_pixel(self, coordinates):
        # Convert coordinates to pixel
        with rasterio.open(fp=self.mosaic_out_path) as mosaic:
            pixel = rasterio.transform.rowcol(transform=mosaic.transform, xs=coordinates[0], ys=coordinates[1])
        return pixel

    def pixel_to_coordinates(self, pixel):
        # Convert pixel to coordinates
        with rasterio.open(fp=self.mosaic_out_path) as mosaic:
            coordinates = rasterio.transform.xy(transform=mosaic.transform, rows=pixel[0], cols=pixel[1])
        return coordinates

    def get_mosaic_array(self):
        """
        The mosaic array is a numpy array with shape (bands, height, width)
        The number of bands of the dataset is 4 (Red, Green, Blue, NIR)
        The height is the number of pixels in the y-direction, and the width is the number of pixels in the x-direction.
        Each pixel has a value for each band and the value is of type int16. Each pixel covers an area of 3m x 3m.
        """
        # Get the mosaic array
        with rasterio.open(fp=self.mosaic_out_path) as mosaic:
            mosaic_array = mosaic.read()
        return mosaic_array

    def get_mask_array(self):
        # Get the mask array
        with rasterio.open(fp=self.mask_out_path) as mask:
            mask_array = mask.read()
        return mask_array

    def get_shape(self):
        # Get the shape of the mosaic
        with rasterio.open(fp=self.mosaic_out_path) as mosaic:
            shape = mosaic.shape
        return shape

    def train_val_test_split(self, train:float=0.6, val:float=0.2, test:float=0.2, force_update:bool=False, plot = False):
        """
        Iterates though possible seperating lines along the second axis (y-axis) to split the mosaic and mask into train and (1-train) parts. The split is determined by the distribution of positives of the mask.
        Now two parts exist: train (north/top) and test+val (south/bottom).
        The test+val part is split into test and val parts by iterating through possible seperating lines along the first axis (x-axis). The split is determined by the distribution of positives of the mask.
        The train, val and test mosaic and mask are saved to the output directory and returned.
        """
        os.makedirs(os.path.join('output', 'splitted'), exist_ok=True)
        train_mosaic_path = os.path.join('output', 'splitted', 'train_mosaic.npy')
        train_mask_path = os.path.join('output', 'splitted', 'train_mask.npy')
        val_mosaic_path = os.path.join('output', 'splitted', 'val_mosaic.npy')
        val_mask_path = os.path.join('output', 'splitted', 'val_mask.npy')
        test_mosaic_path = os.path.join('output', 'splitted', 'test_mosaic.npy')
        test_mask_path = os.path.join('output', 'splitted', 'test_mask.npy')
        if ((not force_update) and (os.path.exists(train_mosaic_path) and os.path.exists(train_mask_path) and os.path.exists(val_mosaic_path) and os.path.exists(val_mask_path) and os.path.exists(test_mosaic_path) and os.path.exists(test_mask_path))):
            train_mosaic = np.load(file=train_mosaic_path, allow_pickle=True)
            train_mask = np.load(file=train_mask_path, allow_pickle=True)
            val_mosaic = np.load(file=val_mosaic_path, allow_pickle=True)
            val_mask = np.load(file=val_mask_path, allow_pickle=True)
            test_mosaic = np.load(file=test_mosaic_path, allow_pickle=True)
            test_mask = np.load(file=test_mask_path, allow_pickle=True)
            print(f'Data already splitted and loaded. ')
            return train_mosaic, train_mask, val_mosaic, val_mask, test_mosaic, test_mask
        mosaic_array = self.get_mosaic_array()
        mask_array = self.get_mask_array()
        # Number of positives in the mask
        num_positives_all = np.sum(mask_array, axis=(0,1,2))
        num_positives_y = np.sum(mask_array, axis=(0,1)) / num_positives_all
        # Make a columative sum of the number of positives in the mask along the second axis (y-axis)
        num_positives_y_cumsum = np.cumsum(num_positives_y)
        # Find the index of the first element in num_positives_y_cumsum that is closest to train
        idx_y = np.argmin(np.abs(num_positives_y_cumsum - train))
        # Split the mosaic and mask into train and (val+test)
        mosaic_array_train = mosaic_array[:, :, :idx_y]
        mask_array_train = mask_array[:, :, :idx_y]
        mosaic_array_val_test = mosaic_array[:, :, idx_y:]
        mask_array_val_test = mask_array[:, :, idx_y:]

        num_positives_all = np.sum(mask_array_val_test, axis=(0,1,2))
        # Number of positives in the mask
        num_positives_x = np.sum(mask_array_val_test, axis=(0,2)) / num_positives_all
        # Make a columative sum of the number of positives in the mask along the first axis (x-axis)
        num_positives_x_cumsum = np.cumsum(num_positives_x)
        # Find the index of the first element in num_positives_x_cumsum that is closest to val
        idx_x = np.argmin(np.abs(num_positives_x_cumsum - val))
        # Split the val+test into val and test
        mosaic_array_val = mosaic_array_val_test[:, :idx_x, :]
        mask_array_val = mask_array_val_test[:, :idx_x, :]
        mosaic_array_test = mosaic_array_val_test[:, idx_x:, :]
        mask_array_test = mask_array_val_test[:, idx_x:, :]
        # Save the train, val and test mosaic and mask
        np.save(file=train_mosaic_path, arr=mosaic_array_train)
        np.save(file=train_mask_path, arr=mask_array_train)
        np.save(file=val_mosaic_path, arr=mosaic_array_val)
        np.save(file=val_mask_path, arr=mask_array_val)
        np.save(file=test_mosaic_path, arr=mosaic_array_test)
        np.save(file=test_mask_path, arr=mask_array_test)
        print(f'Saved splitted mosaic and mask to {train_mosaic_path}, {train_mask_path}, {val_mosaic_path}, {val_mask_path}, {test_mosaic_path}, {test_mask_path}')

        if plot:
            # Indicate where the splits are on the mosaic
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            img = mosaic_array.transpose(1,2,0)[..., [3,2,1]]
            q_min = np.quantile(img, q=0.05)
            q_max = np.quantile(img, q=0.95)
            img = np.clip(img, q_min, q_max)
            # Normalize
            img = (255 * (img - q_min) / (q_max - q_min)).astype(np.uint8)
            ax.imshow(img)
            ax.axhline(y=idx_y, color='r', linestyle='--')
            ax.axvline(x=idx_x, color='r', linestyle='--')
            ax.set_title('Mosaic with train, val and test splits')
            plt.savefig(os.path.join(PATH_TO_FIGS, 'mosaic_with_train_val_test_splits.png'))
            plt.show()

        return mosaic_array_train, mask_array_train, mosaic_array_val, mask_array_val, mosaic_array_test, mask_array_test



    def make_patches(self, patch_size:int=512, stride:int=256):
        """
        Makes patches of the train, val and test mosaic and mask.
        The patches are saved to the output directory and returned.
        """
        train_mosaic, train_mask, val_mosaic, val_mask, test_mosaic, test_mask = self.train_val_test_split()
        # Make patches of the train mosaic and mask
        train_mosaic_patches = np.array([train_mosaic[:,x:x+patch_size,y:y+patch_size] for x in range(0, train_mosaic.shape[1]-patch_size+1, stride) for y in range(0, train_mosaic.shape[2]-patch_size+1, stride)])
        train_mask_patches = np.array([train_mask[:,x:x+patch_size,y:y+patch_size] for x in range(0, train_mask.shape[1]-patch_size+1, stride) for y in range(0, train_mask.shape[2]-patch_size+1, stride)])
        # Make patches of the val mosaic and mask
        val_mosaic_patches = np.array([val_mosaic[:,x:x+patch_size,y:y+patch_size] for x in range(0, val_mosaic.shape[1]-patch_size+1, stride) for y in range(0, val_mosaic.shape[2]-patch_size+1, stride)])
        val_mask_patches = np.array([val_mask[:,x:x+patch_size,y:y+patch_size] for x in range(0, val_mask.shape[1]-patch_size+1, stride) for y in range(0, val_mask.shape[2]-patch_size+1, stride)])
        # Make patches of the test mosaic and mask
        test_mosaic_patches = np.array([test_mosaic[:,x:x+patch_size,y:y+patch_size] for x in range(0, test_mosaic.shape[1]-patch_size+1, stride) for y in range(0, test_mosaic.shape[2]-patch_size+1, stride)])
        test_mask_patches = np.array([test_mask[:,x:x+patch_size,y:y+patch_size] for x in range(0, test_mask.shape[1]-patch_size+1, stride) for y in range(0, test_mask.shape[2]-patch_size+1, stride)])

        # Save the train, val and test mosaic and mask
        os.makedirs(os.path.join('output', 'patches'), exist_ok=True)
        train_mosaic_patches_path = os.path.join('output', 'patches', 'train_mosaic_patches.npy')
        train_mask_patches_path = os.path.join('output', 'patches', 'train_mask_patches.npy')
        val_mosaic_patches_path = os.path.join('output', 'patches', 'val_mosaic_patches.npy')
        val_mask_patches_path = os.path.join('output', 'patches', 'val_mask_patches.npy')
        test_mosaic_patches_path = os.path.join('output', 'patches', 'test_mosaic_patches.npy')
        test_mask_patches_path = os.path.join('output', 'patches', 'test_mask_patches.npy')
        # Save the train, val and test mosaic and mask as numpy arrays
        np.save(file=train_mosaic_patches_path, arr=train_mosaic_patches)
        np.save(file=train_mask_patches_path, arr=train_mask_patches)
        np.save(file=val_mosaic_patches_path, arr=val_mosaic_patches)
        np.save(file=val_mask_patches_path, arr=val_mask_patches)
        np.save(file=test_mosaic_patches_path, arr=test_mosaic_patches)
        np.save(file=test_mask_patches_path, arr=test_mask_patches)
        print(f'Saved train mosaic patches to {train_mosaic_patches_path}')
        print(f'Saved train mask patches to {train_mask_patches_path}')
        print(f'Saved val mosaic patches to {val_mosaic_patches_path}')
        print(f'Saved val mask patches to {val_mask_patches_path}')
        print(f'Saved test mosaic patches to {test_mosaic_patches_path}')
        print(f'Saved test mask patches to {test_mask_patches_path}')
        return train_mosaic_patches, train_mask_patches, val_mosaic_patches, val_mask_patches, test_mosaic_patches, test_mask_patches

    def get_random_point(self):
        # Get a random point
        bounds = self.get_bounds()
        x = np.random.uniform(low=bounds.left, high=bounds.right)
        y = np.random.uniform(low=bounds.bottom, high=bounds.top)
        return (x, y)

    def get_random_pixel(self, delta:int=0):
        # Get a random pixel
        shape = self.get_shape()
        x = np.random.randint(low=0+delta, high=shape[0]-delta)
        y = np.random.randint(low=0+delta, high=shape[1]-delta)
        return (x, y)

    def get_random_pixels(self, num_pixels:int=NUM_BOXES, pixels:int=256, p:float=P_CONTAINS_ISLAND):
        assert pixels > 0, 'pixels must be greater than 0'
        assert 0 <= p <= 1, 'p must be in range [0, 1]'
        assert num_pixels > 0, 'num_pixels must be greater than 0'
        pixels_half = pixels // 2
        mask_array = self.get_mask_array()[:,pixels_half:-pixels_half,pixels_half:-pixels_half]
        masked_idx = np.argwhere(mask_array >= 1/2).T
        not_masked_idx = np.argwhere(mask_array < 1/2).T
        X_is = masked_idx[1]
        Y_is = masked_idx[2]
        X_isnt = not_masked_idx[1]
        Y_isnt = not_masked_idx[2]
        random_pixels = []
        is_isnt = np.random.choice([True, False], size=num_pixels, p=[p, 1-p])
        for i in range(num_pixels):
            to_be = is_isnt[i]
            if to_be:
                # Get a random pixel where the mask is one
                x = np.random.choice(X_is)
                y = np.random.choice(Y_is)
            else:
                # Get a random pixel where the mask is zero
                x = np.random.choice(X_isnt)
                y = np.random.choice(Y_isnt)
            # Add pixel_tuple to random_pixels
            x,y = x+pixels_half, y+pixels_half
            random_pixels.append((x, y))
        return random_pixels

    def get_boxes_around_random_pixels(self, num_boxes:int=NUM_BOXES, pixels:int=256):
        assert pixels > 0, "Pixels must be greater than 0. "
        pixels_half = pixels // 2
        # Get the random pixels
        random_pixels = self.get_random_pixels(num_pixels=num_boxes, pixels = pixels)
        # Get random boxes around the random pixels
        boxes = []
        for pixel in random_pixels:
            x, y = pixel
            # Get the box bounds
            box_bounds = [x-pixels_half, y-pixels_half, x+pixels_half, y+pixels_half]
            boxes.append(box_bounds)
        return boxes

    def close_square(self, a):
        a = int(a)
        small:int = 2 ** 1
        big:int = 2 ** 2
        while a > big:
            small = small * 2
            big   = big   * 2
        return small

    def get_random_submosaics_and_masks(self, N:int=NUM_BOXES, km:float=KILOMETERS):
        mosaic = self.get_mosaic_array()
        mask = self.get_mask_array()
        pixels = self.close_square(int((km * 1000)/self.worst_res[0]))
        print(f'Getting {N} random submosaics and submasks with ≃{km} km width and height which is "converted and rounded down" to {pixels} pixels in width and height')
        # Get the random boxes
        boxes = self.get_boxes_around_random_pixels(num_boxes = N, pixels = pixels)
        # Get the submosaics
        X = np.zeros(shape=(N, 4, pixels, pixels))
        Y = np.zeros(shape=(N, 1, pixels, pixels))
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            # Get the submosaic
            X[i] = mosaic[:, x_min:x_max, y_min:y_max]
            Y[i] = mask[:, x_min:x_max, y_min:y_max]
        return X,Y


    def plot_arrays(self, colors:list[str]=['NIR', 'R', 'G'] ,q:float = 0.05, crop_black_corners: bool=False, num_boxes=NUM_BOXES, km:float=KILOMETERS, show_grid:bool=True):
        mosaic_array = self.get_mosaic_array().astype(np.float32)
        mask_array = self.get_mask_array().astype(np.uint8).squeeze()
        pixels = self.close_square(int((km * 1000)/self.worst_res[0]))
        boxes = self.get_boxes_around_random_pixels(num_boxes=num_boxes, pixels=pixels)
        box_array = np.zeros_like(mask_array, dtype=np.uint8).squeeze()
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            box_array[x_min:x_max, y_min:y_max] = 1
        assert 0<=q<=1, "q is supposed to be float in range [0, 1]"
        q_low: float = np.min([q, 1-q])
        q_high: float = np.max([q, 1-q])
        col_dict: dict[str, int] = {'NIR':3, 'nir':3,
                                    'R':2,'RED':2,'Red':2,'red':2,'r':2,
                                    'G':1,'g':1,'Green':1,'green':1,'GREEN':1,
                                    'B':0,'Blue':0,'BLUE':0,'blue':0,'b':0}
        if type(colors[0]) == str:
            colors = [col_dict[c] for c in colors]
        if mosaic_array.shape[0] <= 4:
            mosaic_array = np.moveaxis(mosaic_array, 0, -1)
        if not mosaic_array.shape[-1] <= 4:
            raise ValueError('mosaic_array should have 4 or less bands')
        if mosaic_array.shape[-1] == 4:
            mosaic_array = mosaic_array[..., colors]
        # Remove outliers
        q_min = np.quantile(mosaic_array, q=q_low)
        q_max = np.quantile(mosaic_array, q=q_high)
        mosaic_array = np.clip(mosaic_array, q_min, q_max)
        # Normalize
        mosaic_array = (255 * (mosaic_array - q_min) / (q_max - q_min)).astype(np.uint8)
        # Zoom in by cropping until the image does not contain any black in the corners
        if crop_black_corners:
            while np.all(mosaic_array[0,0,:] == 0):
                mosaic_array = mosaic_array[1:,1:,:]
                mask_array = mask_array[1:,1:]
                box_array = box_array[1:,1:]
            while np.all(mosaic_array[-1,-1,:] == 0):
                mosaic_array = mosaic_array[:-1,:-1,:]
                mask_array = mask_array[:-1,:-1]
                box_array = box_array[:-1,:-1]
            while np.all(mosaic_array[0,-1,:] == 0):
                mosaic_array = mosaic_array[1:,:-1,:]
                mask_array = mask_array[1:,:-1]
                box_array = box_array[1:,:-1]
            while np.all(mosaic_array[-1,0,:] == 0):
                mosaic_array = mosaic_array[:-1,1:,:]
                mask_array = mask_array[:-1,1:]
                box_array = box_array[:-1,1:]
        fig = plt.figure(figsize=(8, 8), dpi= 200, tight_layout=True)
        ax1 = fig.add_subplot(2, 2, 1)
        # Show the masked mosaic
        ax1.imshow(X=mosaic_array)
        ax1.imshow(X=mask_array, cmap='gray', alpha=0.5)
        ax1.set_title(label='Mosaic with mask')
        ax1.axis('off')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(X=mosaic_array)
        ax2.set_title(label='Mosaic')
        ax2.axis('off')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(X=mask_array)
        ax3.set_title(label='Mask')
        ax3.axis('off')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(X=box_array)
        ax4.set_title(label='Box')
        ax4.axis('off')
        # Set the grid
        if show_grid:
            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax4.grid()
        # Save the figure
        path_to_this_fig = os.path.join(PATH_TO_FIGS, 'mosaic_mask_boxes.png')

        plt.savefig(path_to_this_fig, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Figure of mosaic, mask and boxes saved to {path_to_this_fig}. ')
        return mosaic_array, mask_array, box_array

# %% Create the dataset
raw_dataset = custom_raster_set(root=GTIFFs_PATH)

# %% Calculate the CRS of the dataset
max_crs = raw_dataset.calculate_crs(verbose=True)

# %% Make the mosaic and mask from the dataset and save them
SHIFT_PIXELS = (y_shift, x_shift)
print(f'Making mosaic and mask from dataset. Shifting mask by {SHIFT_PIXELS} pixels.')
mosaic_path, mask_path = raw_dataset.update_mosaic(force=False, shiftpix=SHIFT_PIXELS)
print(f'Mosaic and mask saved to {mosaic_path} and {mask_path}')

# %%
# Print the shape of the mosaic and mask
print(f"Mosaic shape: {raw_dataset.get_mosaic_array().shape}")
print(f"Mask shape: {raw_dataset.get_mask_array().shape}")

# %% Make the train, val and test mosaic and mask
train_mosaic, train_mask, val_mosaic, val_mask, test_mosaic, test_mask = raw_dataset.train_val_test_split(train=0.6, val=0.2, test=0.2, force_update=True, plot=True)
# Print the shape of the train, val and test mosaic and mask
print(f"Train mosaic shape: {train_mosaic.shape}")
print(f"Train mask shape: {train_mask.shape}")
print(f"Val mosaic shape: {val_mosaic.shape}")
print(f"Val mask shape: {val_mask.shape}")
print(f"Test mosaic shape: {test_mosaic.shape}")
print(f"Test mask shape: {test_mask.shape}")
# %% Make patches of the train, val and test mosaic and mask
train_mosaic_patches, train_mask_patches, val_mosaic_patches, val_mask_patches, test_mosaic_patches, test_mask_patches = raw_dataset.make_patches(patch_size=512, stride=512)
# Print the shape of the train, val and test mosaic and mask patches
print(f"Train mosaic patches shape: {train_mosaic_patches.shape}")
print(f"Train mask patches shape: {train_mask_patches.shape}")
print(f"Val mosaic patches shape: {val_mosaic_patches.shape}")
print(f"Val mask patches shape: {val_mask_patches.shape}")
print(f"Test mosaic patches shape: {test_mosaic_patches.shape}")
print(f"Test mask patches shape: {test_mask_patches.shape}")

# Stop the program here
import sys
print('Stopped the program. ')
sys.exit()

# %% Define the dataset of submosaics and submasks
class custom_submosaic_set(Dataset):
    def __init__(self, submosaics, submasks, storage_path = os.path.join('output', 'dataset'), transform=None, target_transform=None):
        self.submosaics = np.array(submosaics)
        self.submasks = np.array(submasks)
        self.storage_path = storage_path
        self.transform = transform
        self.target_transform = target_transform
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        self.storage_path = os.path.join(storage_path, 'dataset.npz') if not storage_path.endswith('.npz') else storage_path

    def __getitem__(self, index):
        return self.submosaics[index], self.submasks[index]

    def __len__(self):
        return self.submosaics.shape[0]

    def save_as_npz(self):
        np.savez_compressed(file=self.storage_path, submosaics=self.submosaics, submasks=self.submasks)
        print(f'Saved dataset to {self.storage_path}')
        return self.storage_path

    def load_from_npz(self):
        dataset = np.load(file=self.storage_path)
        self.submosaics = dataset['submosaics']
        self.submasks = dataset['submasks']
        print(f'Loaded dataset from {self.storage_path}')
        return self.submosaics, self.submasks

# %%
# Make dataset of submosaics and submasks
submosaics, submasks = raw_dataset.get_random_submosaics_and_masks(N=NUM_BOXES, km=KILOMETERS)
# Print shape of submosaics and submasks
print(f'submosaics.shape: {submosaics.shape}')
print(f'submasks.shape: {submasks.shape}')

# %% Save the dataset of submosaics and submasks
dataset = custom_submosaic_set(submosaics=submosaics, submasks=submasks)
storage_path = dataset.save_as_npz()

# %% Make a plot of the mosaic, mask and box
# mosaic_array, mask_array, box_array = raw_dataset.plot_arrays(colors=['NIR', 'R', 'G'], q=0.1, crop_black_corners=True, num_boxes=NUM_BOXES, km=KILOMETERS, show_grid=True)
