# Validation results for the models inferring using IREE

## Public models

We infer models using the following APIs:

1. IREE, when we load PyTorch models directly from source format.

   ```bash
   python inference_iree.py -t classification -is 1 3 224 224 \
                            -mn densenet121 \
                            -tm torchvision.models \
                            -f pytorch \
                            -i data/ \
                            --norm --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
                            -l labels/image_net_synset.txt \
                            --layout NCHW --channel_swap 2 1 0 \
                            -fn main
   ```

1. IREE, when we load ONNX models directly from source format.

   ```bash
   python inference_iree.py -t classification -is 1 3 224 224 \
                            -mn densenet121 \
                            -m densenet121.onnx \
                            -f onnx \
                            --onnx_opset_version 18 \
                            -i data/ \
                            --norm --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
                            -l labels/image_net_synset.txt \
                            --layout NCHW --channel_swap 2 1 0 \
                            -fn main_graph
   ```

1. PyTorch as source framework for reference.

   ```bash
   python inference_pytorch.py -t classification -is [1,3,224,224] \
                               --input_names data \
                               -mn densenet121 \
                               -mm torchvision.models \
                               -i data/ \
                               --mean [123.675,116.28,103.53] \
                               --input_scale [58.395,57.12,57.375] \
                               -l labels/image_net_synset.txt
   ```

### Notes

1. Models in ONNX format loaded from [onnx/models][onnx-models] repository.
1. The model `squeezenet1.1` is missed in [onnx/models][onnx-models] repository.

### Image classification

#### Test image #1

Data source: [ImageNet][imagenet]

Image resolution: 709 x 510
﻿
<div style='float: center'>
<img width="150" src="images\ILSVRC2012_val_00000023.JPEG"></img>
</div>

Model | Source Framework | Python API (source framework) | Python API (IREE, PyTorch) | Python API (IREE, ONNX) |
-|-|-|-|-|
densenet-121 | PyTorch | 0.9525911 Granny Smith<br>0.0132309 orange <br>0.0123391 lemon <br>0.0028140 banana <br>0.0020238 piggy bank, penny bank | 0.9523347 Granny Smith<br>0.0132272 orange<br>0.0125170 lemon<br>0.0027910 banana<br>0.0020333 piggy bank, penny bank | 0.9523349 Granny Smith<br>0.0132271 orange<br>0.0125169 lemon<br>0.0027909 banana<br>0.0020333 piggy bank, penny bank |
efficientnet-b0 | PyTorch | 0.3421609 Granny Smith<br />0.1089311 piggy bank, penny bank <br />0.0693323 teapot <br />0.0249018 vase <br />0.0205339 saltshaker, salt shaker | 0.3421628 Granny Smith<br>0.1089310 piggy bank, penny bank<br>0.0693315 teapot<br>0.0249016 vase<br>0.0205339 saltshaker, salt shaker | 0.3421622 Granny Smith<br>0.1089308 piggy bank, penny bank<br>0.0693314 teapot<br>0.0249017 vase<br>0.0205338 saltshaker, salt shaker |
googlenet-v1 | PyTorch | 0.5399834 Granny Smith<br>0.1101810 piggy bank, penny bank <br>0.0232574 vase <br>0.0213452 pitcher, ewer <br>0.0198953 bell pepper | 0.5432554 Granny Smith<br>0.1103971 piggy bank, penny bank<br>0.0232568 vase<br>0.0213901 pitcher, ewer<br>0.0196196 bell pepper | 0.5432543 Granny Smith<br>0.1103970 piggy bank, penny bank<br>0.0232569 vase<br>0.0213901 pitcher, ewer<br>0.0196196 bell pepper |
resnet-50 | PyTorch | 0.9280675 Granny Smith<br />0.0129466 orange <br />0.0058861 lemon <br />0.0041993 necklace <br />0.0025445 banana | 0.9278086 Granny Smith<br>0.0129410 orange<br>0.0059573 lemon<br>0.0042141 necklace<br>0.0025712 banana | 0.4216066 Granny Smith<br>0.0661015 dumbbell<br>0.0348192 barbell<br>0.0049673 orange<br>0.0045203 syringe |
squeezenet1.1 | PyTorch | 0.5913458 piggy bank, penny bank<br />0.0682889 Granny Smith <br />0.0610993 lemon <br />0.0596012 necklace <br />0.0492096 bucket, pail | 0.5895361 piggy bank, penny bank<br>0.0677933 Granny Smith<br>0.0610654 necklace<br>0.0610450 lemon<br>0.0490914 bucket, pail | - |

#### Test image #2

Data source: [ImageNet][imagenet]

Image resolution: 500 x 500
﻿
<div style='float: center'>
<img width="150" src="images\ILSVRC2012_val_00000247.JPEG">
</div>

Model | Source Framework | Python API (source framework) | Python API (IREE, PyTorch) | Python API (IREE, ONNX) |
-|-|-|-|-|
densenet-121 | PyTorch | 0.9847536 junco, snowbird<br />0.0068679 chickadee <br />0.0034511 brambling, Fringilla montifringilla <br />0.0015685 water ouzel, dipper <br />0.0012343 indigo bunting, indigo finch, indigo bird, Passerina cyanea | 0.9841590 junco, snowbird<br>0.0072199 chickadee<br>0.0034962 brambling, Fringilla montifringilla<br>0.0016226 water ouzel, dipper<br>0.0012858 indigo bunting, indigo finch, indigo bird, Passerina cyanea | 0.9841590 junco, snowbird<br>0.0072199 chickadee<br>0.0034962 brambling, Fringilla montifringilla<br>0.0016226 water ouzel, dipper<br>0.0012858 indigo bunting, indigo finch, indigo bird, Passerina cyanea |
efficientnet-b0 | PyTorch | 0.8903497 junco, snowbird<br />0.0147084 water ouzel, dipper <br />0.0074830 chickadee <br />0.0044766 brambling, Fringilla montifringilla <br />0.0027406 goldfinch, Carduelis carduelis | 0.8903519 junco, snowbird<br>0.0147081 water ouzel, dipper<br>0.0074829 chickadee<br>0.0044765 brambling, Fringilla montifringilla<br>0.0027406 goldfinch, Carduelis carduelis | 0.8903498 junco, snowbird<br>0.0147084 water ouzel, dipper<br>0.0074830 chickadee<br>0.0044766 brambling, Fringilla montifringilla<br>0.0027406 goldfinch, Carduelis carduelis |
googlenet-v1 | PyTorch | 0.6449553 junco, snowbird<br />0.0752306 chickadee <br />0.0480572 brambling, Fringilla montifringilla <br />0.0298399 goldfinch, Carduelis carduelis <br />0.0126128 house finch, linnet, Carpodacus mexicanus | 0.6461055 junco, snowbird<br>0.0772564 chickadee<br>0.0468782 brambling, Fringilla montifringilla<br>0.0295897 goldfinch, Carduelis carduelis<br>0.0123322 house finch, linnet, Carpodacus mexicanus | 0.6461049 junco, snowbird<br>0.0772565 chickadee<br>0.0468783 brambling, Fringilla montifringilla<br>0.0295897 goldfinch, Carduelis carduelis<br>0.0123323 house finch, linnet, Carpodacus mexicanus |
resnet-50 | PyTorch | 0.9809760 junco, snowbird<br />0.0049167 goldfinch, Carduelis carduelis <br />0.0036987 chickadee <br />0.0036697 water ouzel, dipper <br />0.0029304 brambling, Fringilla montifringilla | 0.9805012 junco, snowbird<br>0.0049154 goldfinch, Carduelis carduelis<br>0.0039196 chickadee<br>0.0038098 water ouzel, dipper<br>0.0028983 brambling, Fringilla montifringilla | 0.3845567 junco, snowbird<br>0.0091156 water ouzel, dipper<br>0.0054526 chickadee<br>0.0026206 indigo bunting, indigo finch, indigo bird, Passerina cyanea<br>0.0023612 brambling, Fringilla montifringilla |
squeezenet1.1 | PyTorch | 0.9609295 junco, snowbird<br />0.0248581 chickadee <br />0.0042597 brambling, Fringilla montifringilla <br />0.0037157 goldfinch, Carduelis carduelis <br />0.0033528 ruffed grouse, partridge, Bonasa umbellus | 0.9614577 junco, snowbird<br>0.0250981 chickadee<br>0.0040701 brambling, Fringilla montifringilla<br>0.0035156 goldfinch, Carduelis carduelis<br>0.0030858 ruffed grouse, partridge, Bonasa umbellus | - |

#### Test image #3

Data source: [ImageNet][imagenet]

Image resolution: 333 x 500
﻿
<div style='float: center'>
<img width="150" src="images\ILSVRC2012_val_00018592.JPEG">
</div>

Model | Source Framework | Python API (source framework) | Python API (IREE, PyTorch) | Python API (IREE, ONNX) |
-|-|-|-|-|
densenet-121 | PyTorch | 0.3047960 liner, ocean liner<br />0.1327189 breakwater, groin, groyne, mole, bulwark, seawall, jetty <br />0.1180288 container ship, containership, container vessel <br />0.0794686 drilling platform, offshore rig <br />0.0718431 dock, dockage, docking facility | 0.3022414 liner, ocean liner<br>0.1322474 breakwater, groin, groyne, mole, bulwark, seawall, jetty<br>0.1194614 container ship, containership, container vessel<br>0.0795042 drilling platform, offshore rig<br>0.0723073 dock, dockage, docking facility | 0.3022407 liner, ocean liner<br>0.1322481 breakwater, groin, groyne, mole, bulwark, seawall, jetty<br>0.1194605 container ship, containership, container vessel<br>0.0795041 drilling platform, offshore rig<br>0.0723069 dock, dockage, docking facility |
efficientnet-b0 | PyTorch | 0.4476882 breakwater, groin, groyne, mole, bulwark, seawall, jetty<br />0.0953832 container ship, containership, container vessel <br />0.0872342 beacon, lighthouse, beacon light, pharos <br />0.0559825 drilling platform, offshore rig <br />0.0441807 liner, ocean liner | 0.4476875 breakwater, groin, groyne, mole, bulwark, seawall, jetty<br>0.0953838 container ship, containership, container vessel<br>0.0872344 beacon, lighthouse, beacon light, pharos<br>0.0559831 drilling platform, offshore rig<br>0.0441806 liner, ocean liner | 0.4476894 breakwater, groin, groyne, mole, bulwark, seawall, jetty<br>0.0953836 container ship, containership, container vessel<br>0.0872341 beacon, lighthouse, beacon light, pharos<br>0.0559827 drilling platform, offshore rig<br>0.0441803 liner, ocean liner |
googlenet-v1 | PyTorch | 0.1330581 liner, ocean liner<br />0.0796951 drilling platform, offshore rig <br />0.0680323 container ship, containership, container vessel <br />0.0588053 breakwater, groin, groyne, mole, bulwark, seawall, jetty <br />0.0365606 fireboat | 0.1323653 liner, ocean liner<br>0.0796393 drilling platform, offshore rig<br>0.0678083 container ship, containership, container vessel<br>0.0585719 breakwater, groin, groyne, mole, bulwark, seawall, jetty<br>0.0366882 fireboat | 0.1323648 liner, ocean liner<br>0.0796394 drilling platform, offshore rig<br>0.0678085 container ship, containership, container vessel<br>0.0585720 breakwater, groin, groyne, mole, bulwark, seawall, jetty<br>0.0366881 fireboat |
resnet-50 | PyTorch | 0.4818293 liner, ocean liner<br />0.0992477 breakwater, groin, groyne, mole, bulwark, seawall, jetty <br />0.0687505 container ship, containership, container vessel <br />0.0517874 dock, dockage, docking facility <br />0.0483462 pirate, pirate ship | 0.4759648 liner, ocean liner<br>0.1025407 breakwater, groin, groyne, mole, bulwark, seawall, jetty<br>0.0689996 container ship, containership, container vessel<br>0.0524496 dock, dockage, docking facility<br>0.0473777 pirate, pirate ship | 0.1220204 lifeboat<br>0.0430796 breakwater, groin, groyne, mole, bulwark, seawall, jetty<br>0.0360478 beacon, lighthouse, beacon light, pharos<br>0.0335465 dock, dockage, docking facility<br>0.0251255 liner, ocean liner |
squeezenet1.1 | PyTorch | 0.4393108 liner, ocean liner<br />0.1895231 container ship, containership, container vessel <br />0.1506845 pirate, pirate ship <br />0.0962459 fireboat <br />0.0199389 drilling platform, offshore rig | 0.4413096 liner, ocean liner<br>0.1931005 container ship, containership, container vessel<br>0.1459103 pirate, pirate ship<br>0.0937753 fireboat<br>0.0198682 drilling platform, offshore rig | - |

<!-- LINKS -->
[imagenet]: http://www.image-net.org
[onnx-models]: https://github.com/onnx/models/tree/main
