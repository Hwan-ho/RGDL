from __future__ import division
import time
from util import *
import os
import os.path as osp
from darknet import Darknet
from preprocess import prep_image
import scipy.io as sio

images = "YourImagePath"
batch_size = 1
confidence = 0.5
nms_thesh = 0.4
start = 0

CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes('data/coco.names')

# Set up the neural network
print("Loading network.....")
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
print("Network successfully loaded")

model.net_info["height"] = 128
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

read_dir = time.time()
# Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if
              os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[
                  1] == '.jpg']
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

if not os.path.exists("det"):
    os.makedirs("det")

load_batch = time.time()

batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
im_batches = [x[0] for x in batches]
orig_ims = [x[1] for x in batches]
im_dim_list = [x[2] for x in batches]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

if CUDA:
    im_dim_list = im_dim_list.cuda()

leftover = 0

if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                            len(im_batches))])) for i in range(num_batches)]

predictionStore = None
for batch in im_batches:
    x = batch.cuda()

    # Forward
    outputs = {}  # We cache the outputs for the route layer
    write = False
    modules = model.blocks[1:]
    for i in range(len(modules)):

        module_type = (modules[i]["type"])
        if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":

            x = model.module_list[i](x)
            outputs[i] = x


        elif module_type == "route":
            layers = modules[i]["layers"]
            layers = [int(a) for a in layers]

            if (layers[0]) > 0:
                layers[0] = layers[0] - i

            if len(layers) == 1:
                x = outputs[i + (layers[0])]

            else:
                if (layers[1]) > 0:
                    layers[1] = layers[1] - i

                map1 = outputs[i + layers[0]]
                map2 = outputs[i + layers[1]]

                x = torch.cat((map1, map2), 1)
            outputs[i] = x

        elif module_type == "shortcut":
            from_ = int(modules[i]["from"])
            x = outputs[i - 1] + outputs[i + from_]
            outputs[i] = x



        elif module_type == 'yolo':

            anchors = model.module_list[i][0].anchors
            # Get the input dimensions
            inp_dim = int(model.net_info["height"])

            # Get the number of classes
            num_classes = int(modules[i]["classes"])

            # Output the result
            x = x.data
            x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

            if type(x) == int:
                continue

            if not write:
                detections = x
                write = 1

            else:
                detections = torch.cat((detections, x), 1)

            outputs[i] = outputs[i - 1]

    if predictionStore is None:
        predictionStore = outputs[106].cpu().detach()
    else:
        predictionStore = torch.cat([predictionStore, outputs[106].cpu().detach()], dim=0)

latentVariables = predictionStore.cpu().detach().numpy()
latentVariables = latentVariables.reshape((latentVariables.shape[0], -1))
sio.savemat("YourPath",
            {"latentVariables": latentVariables, "imlist": imlist})
