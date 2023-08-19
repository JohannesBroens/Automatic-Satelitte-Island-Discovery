# %%


# %% [markdown]
# ## Imports

# %%
import numpy as np, datetime, os, matplotlib.pyplot as plt, torch, sys, pickle, torch.nn as nn, torch.nn.functional as F, torchmetrics, geopandas as gpd, rasterio, json
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import from_numpy
from torchvision import transforms

#from torchgeometry.losses import tversky, TverskyLoss
from scipy.ndimage import binary_dilation
from pathlib import Path
from torch import nn
from rasterio.features import rasterize
import rasterio.io as rio_io
from rasterio import open as rio_open
from geopandas import read_file as gpd_reader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.utils.losses import DiceLoss, JaccardLoss
from segmentation_models_pytorch.utils.metrics import IoU, Precision, Recall, Fscore, Accuracy
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

#from torch.nn.functional import grid_sample, affine_grid

# %% [markdown]
# ## Constants
# %%

THIS_FILE_IS_NOTEBOOK = False

if THIS_FILE_IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
# %%
TEST_RUN = False
SAVE_RESULTS = not TEST_RUN
MAKE_NEW_DATA = False
FORCE_CPU = False
DATA_DIR = Path('data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() and not FORCE_CPU else 'cpu')
WINDOW_SIZE = 512
BATCH_SIZE = 1
EPOCHS = 50
if TEST_RUN:
    EPOCHS = 1
LEARNING_RATE = 0.00001
ENCODER_NAME = "efficientnet-b7"
ENCODER_DEPTH = 5
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 4
ACTIVATION = "sigmoid"
FORCE_CPU = False

IS_CUDA = torch.cuda.is_available() and not FORCE_CPU
if IS_CUDA:
    GPU_COUNT = torch.cuda.device_count()
    tqdm.write(s="CUDA is available, GPU count: " + str(GPU_COUNT) + ", GPUs:", end="\n", file=sys.stderr)
    for i in range(GPU_COUNT):
        tqdm.write(s="\t- GPU " + str(i) + " name: " + torch.cuda.get_device_name(device=i), end="\n", file=sys.stderr)
        tqdm.write(s="\t  GPU " + str(i) + " memory: " + str(np.round(a=(torch.cuda.get_device_properties(device=i).total_memory / 1024 ** 3), decimals=2)) + "GB", end="\n", file=sys.stderr)
    if GPU_COUNT > 1:
        try:
            gpu_id = list(range(GPU_COUNT))
            DEVICE = torch.device("cuda")
        except:
            maximum_memory_gpu_id = np.argmax(a=[torch.cuda.get_device_properties(device=i).total_memory for i in gpu_id])
            gpu_id = gpu_id[maximum_memory_gpu_id]
            DEVICE = torch.device("cuda:" + str(gpu_id))
    else:
        gpu_id = 0
        DEVICE = torch.device("cuda:" + str(gpu_id))
    tqdm.write(s="Using GPU[s]: " + str(gpu_id), end="\n", file=sys.stderr)
    summary_of_device_mem = torch.cuda.memory_summary(device=DEVICE, abbreviated=True)
    tqdm.write(s=str(summary_of_device_mem), end="\n", file=sys.stderr)
else:
    tqdm.write(s="CUDA is not available, using CPU", end="\n", file=sys.stderr)
    DEVICE = torch.device(device="cpu")

# Print remaining parameters
tqdm.write(f"Parameters:\nDEVICE: {DEVICE}\nWINDOW_SIZE: {WINDOW_SIZE}\nBATCH_SIZE: {BATCH_SIZE}\nEPOCHS: {EPOCHS}\nLEARNING_RATE: {LEARNING_RATE}\nENCODER_NAME: {ENCODER_NAME}\nENCODER_DEPTH: {ENCODER_DEPTH}\nENCODER_WEIGHTS: {ENCODER_WEIGHTS}\nIN_CHANNELS: {IN_CHANNELS}\nACTIVATION: {ACTIVATION}\nFORCE_CPU: {FORCE_CPU}\nIS_CUDA: {IS_CUDA}\nGPU_COUNT: {GPU_COUNT}\nDATA_DIR: {DATA_DIR}\nTEST_RUN: {TEST_RUN}\nSAVE_RESULTS: {SAVE_RESULTS}\nMAKE_NEW_DATA: {MAKE_NEW_DATA}")



#print(f"Parameters:\nDEVICE: {DEVICE}\nWINDOW_SIZE: {WINDOW_SIZE}\nBATCH_SIZE: {BATCH_SIZE}\nEPOCHS: {EPOCHS}\nLEARNING_RATE: {LEARNING_RATE}")

# %%
def group_data(data_dir: Path = DATA_DIR):
    X = [] # Tiff files
    y = [] # Shape files
    unknowns = []
    for file in data_dir.iterdir():
        if file.name.endswith(".tif"):
            X.append(file)
        elif file.name.endswith(".shp"):
            y.append(file)
        else:
            unknowns.append(file)
    X.sort()
    # First "island.shp" then "land.shp", finally "ice.shp".
    y.sort(key=lambda x: len(x.name), reverse=True)
    tqdm.write(f"Found {len(X)} tiff files, {len(y)} shape files together with {len(unknowns)} files containing meta-data for the shapefiles.")
    return X, y

# %%
def rectify_X(X):
    """
    Rectifies X to be between 0 and 1
    """
    X = X.astype(np.float32)
    X_min, X_max = np.min(X), np.max(X)
    X = (X - X_min) / (X_max - X_min)
    # As tensor
    X = torch.from_numpy(X).type(torch.float32)
    return X

def get_data(data_dir=DATA_DIR, dialiations=4, crop=True):
    """
    Reads rasters and shapefiles and returns Xs, Ys, Xs_meta
    """
    rasters, shapefiles = group_data(data_dir=data_dir)
    tensors = []
    for raster in tqdm(rasters, desc=f'Reading rasters', total=len(rasters), position=0, leave=False):
        with rio_open(fp=raster,
                      mode='r',
                      driver='GTiff',
                      sharing=True) as raster:
            X = raster.read()
            # Mask where raster is nodata
            raster_is = raster.read(1)
            raster_is[raster_is == raster.nodata] = 0
            raster_is[raster_is != raster.nodata] = 1
            for i, shapefile in tqdm(enumerate(shapefiles), desc=f'Applying and rasterizing shapefiles', total=len(shapefiles)):
                multi_polygons = gpd_reader(shapefile).to_crs(raster.crs)
                shapes = ((geom, 1) for geom in multi_polygons.geometry)
                if i == 0:
                    # Island
                    island = rasterize(
                        shapes = shapes,
                        out_shape = raster.shape,
                        transform = raster.transform,
                        all_touched = True,
                        dtype = rasterio.uint8,
                        fill = 0,
                        default_value = 1
                        ).astype(rasterio.uint8)

                elif i == 1:
                    # Land
                    land = rasterize(
                        shapes = shapes,
                        out_shape = raster.shape,
                        transform = raster.transform,
                        all_touched = True,
                        dtype = rasterio.uint8,
                        fill = 0,
                        default_value = 1
                        ).astype(rasterio.uint8)

                elif i == 2:
                    # Ice
                    ice = binary_dilation(
                        input = rasterize(
                            shapes=shapes,
                            out_shape=raster.shape,
                            transform=raster.transform,
                            all_touched=True,
                            dtype=rasterio.uint8,
                            default_value=1
                            ).astype(bool),
                        iterations = dialiations
                        ).astype(np.float32)
                else:
                    raise ValueError(f'Unknown shapefile: {shapefile}')
                # Mask is 1 if land and not ice
                # Mask is 1 if island
                # Mask is 0 otherwise
            mask = np.logical_or(np.logical_and(land, np.logical_not(ice)), island).astype(rasterio.uint8)
            mask = np.logical_and(mask, raster_is).astype(np.float32)
            if np.unique(mask).tolist() != [0, 1]:
                mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
                mask[mask >= 0.5] = 1
                mask[mask < 0.5] = 0
            if crop:
                # Crop X and mask corners based on mask
                ## top left
                tl = np.argwhere(mask == 1)
                tl = tl.min(axis=0)
                ## bottom right
                br = np.argwhere(mask == 1)
                br = br.max(axis=0)
                # Add to br such that the crop results in length and width that is divisible by WINDOW_SIZE
                if not isinstance(WINDOW_SIZE, tuple):
                    window_size = (WINDOW_SIZE, WINDOW_SIZE)
                else:
                    window_size = WINDOW_SIZE
                br = br + (window_size - (br - tl) % window_size)
                # Crop
                X = X[:, tl[0]:br[0], tl[1]:br[1]]
                mask = mask[tl[0]:br[0], tl[1]:br[1]]
            # As tensor
            img = rectify_X(X=X)
            mask = from_numpy(mask).type(torch.float32).unsqueeze(0)
            # Concatenate img and mask
            tensor = torch.cat((img, mask), dim=0)
            tensors.append(tensor)
    return tensors

# Only get data if not saved to disk
def load_or_save(tensors_dir, data_dir=DATA_DIR, dialiations=5, crop=True):
    """
    Loads or saves tensors to disk
    """
    if not os.path.exists(tensors_dir):
        os.makedirs(tensors_dir)
    if os.listdir(tensors_dir) == [] or MAKE_NEW_DATA:
        tensors = get_data(data_dir=data_dir, dialiations=dialiations, crop=crop)
        for i, tensor in tqdm(enumerate(tensors), desc=f'Saving tensors', total=len(tensors), position=0, leave=True):
            torch.save(tensor, os.path.join(tensors_dir, f'{i}.pt'))
    else:
        tensors = []
        for i in tqdm(range(len(os.listdir(tensors_dir))), desc=f'Loading tensors', total=len(os.listdir(tensors_dir)), position=0, leave=True):
            tensors.append(torch.load(os.path.join(tensors_dir, f'{i}.pt')))
    return tensors

tensors = load_or_save(tensors_dir=os.path.join("output","tensors"), data_dir=DATA_DIR, dialiations=5, crop=True)

# %%
# Make grid sampler of the tensors for UNet
t0,t1,t2,t3 = tensors
x0,y0 = t0[:4],t0[4:]
x1,y1 = t1[:4],t1[4:]
x2,y2 = t2[:4],t2[4:]
x3,y3 = t3[:4],t3[4:]
xs = [x0,x1,x2,x3]
ys = [y0,y1,y2,y3]
# Header
Names = ['train_0', 'train_1', 'validation', 'test']
# For each set, print the shape of the input and output
for i, (x, y) in enumerate(zip(xs, ys)):
    print(f'{Names[i]}: x.shape: {x.shape}, y.shape: {y.shape}')
#%%
#print(f't0.shape: {t0.shape},  x0.shape: {x0.shape},  y0.shape: {y0.shape}')
#print(f't1.shape: {t1.shape},  x1.shape: {x1.shape},  y1.shape: {y1.shape}')
#print(f't2.shape: {t2.shape},  x2.shape: {x2.shape},  y2.shape: {y2.shape}')
#print(f't3.shape: {t3.shape},  x3.shape: {x3.shape},  y3.shape: {y3.shape}')

class GridSampler(torch.utils.data.Sampler):
    """
    Sampler that returns batches of windows of a tensor.
    """
    def __init__(self, tensor:torch.Tensor, window_size:int=WINDOW_SIZE, batch_size:int=BATCH_SIZE, shuffle:bool=False, device=DEVICE):
        """
        Args:
            tensor (torch.Tensor): Tensor to be windowed of shape (5, W, H) for 5 being the number of bands. the first 4 are the input and the last one is the target.
            window_size (int, optional): Size of each window. Defaults to 512.
            shuffle (bool, optional): Whether to shuffle the windows. Defaults to True.
            device ([type], optional): Device to use for the windows. Defaults to DEVICE.
        """
        super().__init__(tensor)
        self.tensor = tensor
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.windows_X, self.windows_y = self._get_windows()
        self.num_windows = self.windows_X.shape[0]
        self.indices = torch.arange(self.num_windows, dtype=torch.int64)
        if self.shuffle:
            self.indices = self.indices[torch.randperm(self.num_windows)]

    def _get_windows(self):
        """
        Returns a list of windows of the tensor.
        """
        windows_X = []
        windows_y = []
        pbar = tqdm(total=self.tensor.shape[-2]*self.tensor.shape[-1] / (self.window_size**2), desc=f'Getting windows', position=0, leave=True)
        for i in range(0, self.tensor.shape[-2], self.window_size):
            for j in range(0, self.tensor.shape[-1], self.window_size):
                pbar.update(1)
                x_win = self.tensor[:-1, i:i+self.window_size, j:j+self.window_size]
                y_win = self.tensor[-1:, i:i+self.window_size, j:j+self.window_size]
                if x_win.shape[-2:] == (self.window_size, self.window_size) and y_win.shape[-2:] == (self.window_size, self.window_size):
                    windows_X.append(x_win)
                    windows_y.append(y_win)
                else:
                    # Pad
                    x_win = F.pad(x_win, (0, self.window_size - x_win.shape[-1], 0, self.window_size - x_win.shape[-2]))
                    y_win = F.pad(y_win, (0, self.window_size - y_win.shape[-1], 0, self.window_size - y_win.shape[-2]))
                    windows_X.append(x_win)
                    windows_y.append(y_win)
        return torch.stack(windows_X), torch.stack(windows_y)

    def __iter__(self):
        """
        Returns an iterator over the windows.
        """
        for i in tqdm(range(0, self.num_windows, self.batch_size), desc=f'Getting batches', total=self.num_windows//self.batch_size, position=0, leave=True):
            yield self.windows_X[self.indices[i:i+self.batch_size]], self.windows_y[self.indices[i:i+self.batch_size]]

    def __len__(self):
        """
        Returns the number of windows.
        """
        return self.num_windows

    def __getitem__(self, index):
        """
        Returns the window at the given index.
        """
        return self.windows_X[index], self.windows_y[index]

# Make grid sampler
grid_sampler_0 = GridSampler(tensor=t0, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE)
grid_sampler_1 = GridSampler(tensor=t1, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE)
grid_sampler_2 = GridSampler(tensor=t2, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle=False, device=DEVICE)
grid_sampler_3 = GridSampler(tensor=t3, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle=False, device=DEVICE)

# Make dataloaders from grid samplers
dataloader_0 = torch.utils.data.DataLoader(grid_sampler_0, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
dataloader_1 = torch.utils.data.DataLoader(grid_sampler_1, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())
dataloader_2 = torch.utils.data.DataLoader(grid_sampler_2, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())
dataloader_3 = torch.utils.data.DataLoader(grid_sampler_3, batch_size=BATCH_SIZE, shuffle=False, pin_memory=torch.cuda.is_available())

def get_decoder_channels(window_size=WINDOW_SIZE, encoder_depth=ENCODER_DEPTH):
    decoder_channels = []
    for i in range(encoder_depth, 0, -1):
        decoder_channels.append(window_size//(2**i))
    return tuple(decoder_channels[::-1])
DECODER_CHANNELS = get_decoder_channels(window_size=WINDOW_SIZE, encoder_depth=ENCODER_DEPTH)
print(f'Decoder channels: {DECODER_CHANNELS}')
unet_model = Unet(
    encoder_name=ENCODER_NAME,
    encoder_depth=ENCODER_DEPTH,
    encoder_weights=ENCODER_WEIGHTS,
    decoder_channels=DECODER_CHANNELS,
    in_channels=IN_CHANNELS,
    classes=1,
    activation=ACTIVATION,
)

OPTIMIZER = Adam([
    dict(params=unet_model.parameters(), lr=LEARNING_RATE),
])
SCHEDULER = ReduceLROnPlateau(optimizer=OPTIMIZER, mode='min', factor=0.1, patience=8, verbose=True)
print(f'Scheduler: {SCHEDULER}')
print(f'Optimizer: {OPTIMIZER}')


# %%
class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super(DiceBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.__name__ = 'DiceBCELoss'

    def forward(self, inputs, targets, smooth=1):
        #inputs = F.sigmoid(inputs)

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * inputs).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (inputs.sum()+inputs.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(input=inputs, target=targets, pos_weight = self.pos_weight)
        Dice_BCE = BCE + dice_loss
        return Dice_BCE
def get_weights(masks_list:list = [y0, y1]):
    """Get the weight for the loss function."""
    total_sum = torch.zeros(size=(1,), dtype=torch.float32, device=DEVICE)
    pos_sum = torch.zeros(size=(1,), dtype=torch.float32, device=DEVICE)
    neg_sum = torch.zeros(size=(1,), dtype=torch.float32, device=DEVICE)
    for mask in masks_list:
        total = torch.tensor(data=torch.numel(mask), dtype=torch.float32, device=DEVICE)
        pos = torch.sum(input=mask, dtype=torch.float32).to(DEVICE)
        neg = total - pos
        total_sum += total
        pos_sum += pos
        neg_sum += neg
    if total_sum == 0:
        weight = torch.tensor(data=100.0, dtype=torch.float32, device=DEVICE)
    else:
        weight = pos_sum / total_sum
    return weight

WGT = get_weights(masks_list=[y0,y1])
print(f'positive class ratio: {WGT.item()}')
# CRITERION = DiceBCELoss(pos_weight=WGT).to(device=DEVICE)
# Try only Dice loss
CRITERION = DiceLoss().to(device=DEVICE)
print(f'Criterion (loss function): {CRITERION.__name__}')

train_epoch = TrainEpoch(
    unet_model,
    loss=CRITERION,
    metrics=[IoU(threshold=0.5)],
    optimizer=OPTIMIZER,
    device=DEVICE,
    verbose=True,
)

valid_epoch = ValidEpoch(
    unet_model,
    loss=CRITERION,
    metrics=[
        IoU(threshold=0.5),
        Precision(threshold=0.5),
        Recall(threshold=0.5),
        Fscore(threshold=0.5),
        Accuracy(threshold=0.5),
    ],
    device=DEVICE,
    verbose=True,
)

def load_model(unet_model=unet_model, model_path=None):
    """Load the model."""
    if model_path is None:
        files_of_interest = os.listdir(os.path.join('output', 'models'))
        # Find the latest model
        model_path = os.path.join('output', 'models', files_of_interest[-1])
    unet_model = torch.load(model_path)
    print(f'Model loaded from {model_path}')
    return unet_model

if TEST_RUN:
    try:
        unet_model = load_model(unet_model = unet_model, model_path='output/models/unet_model.pth_10.pth')
    except:
        unet_model = load_model(model_path='output/models/unet_model_10.pth')
    print('Model loaded for test run.')
else:
    unet_model = load_model(model_path='output/models/unet_model.pth_10.pth')
    print('Model loaded for training.')

def train_model(epochs=EPOCHS):
    """Train the model."""
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    os.makedirs('output/logs', exist_ok=True)
    model_path = os.path.join('output', 'models', 'unet_model')
    max_score = 0.0
    pbar = tqdm(range(1, epochs+1), desc='Epoch: [0]', leave=False)
    for epoch in pbar:
        tqdm.write(f'Epoch: [{epoch}/{epochs}]')
        train_logs = train_epoch.run(dataloader_0)
        train_logs_extra = train_epoch.run(dataloader_1)
        valid_logs = valid_epoch.run(dataloader_2)
        SCHEDULER.step(valid_logs[CRITERION.__name__])
        if max_score < valid_logs['fscore'] and SAVE_RESULTS:
            max_score = valid_logs['fscore']
            path_save = f'{model_path}_{epoch}.pth'
            torch.save(unet_model, path_save)
            tqdm.write(f'Model saved: {path_save}')
        pbar.set_description(f"Train loss: {train_logs[CRITERION.__name__]:.5f} IoU: {train_logs['iou_score']:.5f} | Valid loss: {valid_logs[CRITERION.__name__]:.5f} IoU: {valid_logs['iou_score']:.5f}, IoU_extra: {train_logs_extra['iou_score']:.5f}")
        # Dump logs to file
        if SAVE_RESULTS:
            with open(os.path.join('output', 'logs', 'train_logs.json'), 'a') as f:
                json.dump(train_logs, f)
                f.write('\n')
            with open(os.path.join('output', 'logs', 'train_logs_extra.json'), 'a') as f:
                json.dump(train_logs_extra, f)
                f.write('\n')
            with open(os.path.join('output', 'logs', 'valid_logs.json'), 'a') as f:
                json.dump(valid_logs, f)
                f.write('\n')
    # Save the final model after training and load the best model
    if SAVE_RESULTS:
        path_save_final = f'{model_path}_final.pth'
        torch.save(unet_model, path_save_final)
        tqdm.write(f'Final model saved: {path_save_final}')
    unet_model = load_model(unet_model = unet_model, model_path=path_save)
    return max_score

if not TEST_RUN:
    maxScore = train_model(epochs=EPOCHS)
    print(f'Max score: {maxScore}')

# %%
test_epoch = ValidEpoch(
    unet_model,
    loss=CRITERION,
    metrics=[
        IoU(threshold=0.5),
        Precision(threshold=0.5),
        Recall(threshold=0.5),
        Fscore(threshold=0.5),
        Accuracy(threshold=0.5),
    ],
    device=DEVICE,
    verbose=True,
)

def test_model():
    unet_model.eval()
    with torch.no_grad():
        test_logs = test_epoch.run(dataloader_3)
    print("Testing logs:", test_logs)
    return test_logs

test_logs = test_model()

# Make predictions on the entire test set
def predict_test():
    imgs = []
    masks = []
    pred_masks = []
    unet_model.eval()
    with torch.no_grad():
        test_logs = test_epoch.run(dataloader_3)
        print("Testing logs:", test_logs)
        for img, mask in tqdm(dataloader_3, total=len(dataloader_3), desc='Predicting on test set', position=0, leave=True):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            pred_mask = unet_model(img)
            imgs.append(img.cpu().numpy())
            masks.append(mask.cpu().numpy())
            pred_masks.append(pred_mask.cpu().numpy())
    # Save logs
    imgs, masks, pred_masks =  np.concatenate(imgs), np.concatenate(masks), np.concatenate(pred_masks)
    if SAVE_RESULTS:
        os.makedirs('output/logs', exist_ok=True)
        with open(os.path.join('output', 'logs', 'test_logs.json'), 'a') as f:
            json.dump(test_logs, f)
            f.write('\n')
        os.makedirs('output/predictions', exist_ok=True)
        np.save('output/predictions/imgs.npy', imgs)
        np.save('output/predictions/masks.npy', masks)
        np.save('output/predictions/pred_masks.npy', pred_masks)
    return imgs, masks, pred_masks

imgs, masks, pred_masks = predict_test()

print(f'imgs.shape: {imgs.shape}, masks.shape: {masks.shape}, pred_masks.shape: {pred_masks.shape}')
