import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

# figure size in inches optional
rcParams['figure.figsize'] = 11 ,8

# read images
img_A = mpimg.imread(r'Z:\louisccc\av\synthesis_data\new_recording_3\0\carla_visualize\17794852.png')
img_B = mpimg.imread(r'Z:\louisccc\av\synthesis_data\new_recording_3\0\raw_images\17794852.jpg')

# display images
fig, ax = plt.subplots(1,2)
# plt.axis('off')
ax[0].set_axis_off()
ax[0].imshow(img_A);
ax[1].set_axis_off()
ax[1].imshow(img_B);



plt.show()