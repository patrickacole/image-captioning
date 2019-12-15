import os
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

coco = COCO(os.path.expanduser("~/Projects/datasets/COCO/annotations/captions_val2017.json"))
cocoRes = coco.loadRes(os.path.expanduser("~/Projects/datasets/COCO/results_3.json"))

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()
