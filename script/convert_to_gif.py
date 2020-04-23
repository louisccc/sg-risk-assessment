import imageio
from pathlib import Path

def convert_gif(path):
	folder_path = path / 'raw_images'
	img_path = folder_path.glob('**/*.png')
	images = []
	for filename in img_path:
		images.append(imageio.imread(str(filename)))
	imageio.mimsave(path / 'movie.gif', images, format='GIF')
