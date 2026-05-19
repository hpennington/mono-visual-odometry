import cv2
import numpy as np
import PIL
from IPython import display
from matplotlib.cm import get_cmap

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.proto.ontology_pb2 import Ontology
from dgp.utils.protobuf import open_pbobject
from dgp.utils.visualization import visualize_semantic_segmentation_2d

plasma_color_map = get_cmap('plasma')

# Define high level variables
DDAD_TRAIN_VAL_JSON_PATH = '/data/datasets/ddad_train_val/ddad.json'
DDAD_TEST_JSON_PATH = '/data/datasets/ddad_test/ddad_test.json'
DATUMS = ['lidar'] + ['CAMERA_%02d' % idx for idx in [1, 5, 6, 7, 8, 9]] 

ddad_train = SynchronizedSceneDataset(
    DDAD_TRAIN_VAL_JSON_PATH,
    split='train',
    datum_names=DATUMS,
    generate_depth_from_datum='lidar'
)
print('Loaded DDAD train split containing {} samples'.format(len(ddad_train)))

random_sample_idx = np.random.randint(len(ddad_train))
sample = ddad_train[random_sample_idx] # scene[0] - lidar, scene[1:] - camera datums
sample_datum_names = [datum['datum_name'] for datum in sample]
print('Loaded sample {} with datums {}'.format(random_sample_idx, sample_datum_names))

# Concat images and visualize
images = [cam['rgb'].resize((192,120), PIL.Image.BILINEAR) for cam in sample[1:]]
images = np.concatenate(images, axis=1)
display.display(PIL.Image.fromarray(images))

# Visualize corresponding depths, if the depth has been projected into the camera images
if 'depth' in sample[1].keys():
    # Load and resize depth images
    depths = [cv2.resize(cam['depth'], dsize=(192,120), interpolation=cv2.INTER_NEAREST) \
              for cam in sample[1:]]
    # Convert to RGB for visualization
    depths = [plasma_color_map(d)[:,:,:3] for d in depths]
    depths = np.concatenate(depths, axis=1)
    display.display(PIL.Image.fromarray((depths*255).astype(np.uint8)))