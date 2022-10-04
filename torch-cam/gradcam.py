import torch
from torchvision import datasets, models, transforms
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
import matplotlib.pyplot as plt
from torchcam.cams import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from mpl_toolkits.axes_grid1 import ImageGrid

"""
Local Dependencies
pip install torchcam
pip install torchvision

Class activation explorer using TorchCAM.
Iterate through validation database and overlay images with class activation
heatmap.

Reference: https://github.com/frgfm/torch-cam
"""
def runGradCam():
  _, model, _, _, _, _ = reloadModel()
  if model:
    model.eval()
    cam_extractor = SmoothGradCAMpp(model)

    DATASET_TYPE = 'cam'
    num_iter = 2
    # pred_probs = getPredProbs(model, DATASET_TYPE, num_iter)

    # NOTE: file path is only valid if dataset type is cam
    for i, (img, cls, ori_path) in enumerate(data[DATASET_TYPE]):
      if i >= num_iter:
        break

      img = img.to(device)
      # Preprocess your data and feed it to the model
      out = model(img.unsqueeze(0))
      # Retrieve the CAM by passing the class index and the model output
      activation_map = cam_extractor(cls, out)

      # Resize the CAM and overlay it
      # https://matplotlib.org/stable/tutorials/colors/colormaps.html#miscellaneous
      ori_img = to_pil_image(img)
      result = overlay_mask(ori_img, to_pil_image(activation_map, mode='F'), colormap='jet', alpha=0.5)

      # Display it
      # https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html
      fig = plt.figure(figsize=(20., 20.))
      grid = ImageGrid(fig, 111,  # similar to subplot(111)
                      nrows_ncols=(1, 2),  # creates 1x2 grid of axes
                      axes_pad=0.1,  # pad between axes in inch.
                      )
      for ax, im in zip(grid, [ori_img, result]):
        # Iterating over the grid returns the Axes.
        ax.axis('off')
        classStr = "Covid"
        if cls:
          classStr = "NonCovid"
        # ax.set_title("{:.4f}% Covid, \n{:.4f}% NonCovid, Actual:{}, ImagePath:{}".format(100*pred_probs[i,0],
        #                                                               100*pred_probs[i,1],
        #                                                               classStr, ori_path), fontsize=10)
        ax.imshow(im)
      title = f'{curr_model} - GradCam++ Visualization'
      full_path = os.path.join(result_dir, f'{title}.png')
      plt.savefig(full_path)
      plt.show()


# using gpu or else cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
curr_model = models.densenet201.__name__
runGradCam()
