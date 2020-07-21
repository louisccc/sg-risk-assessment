from tkinter import *
from PIL import Image, ImageTk
from pathlib import Path

class GUI():
    def __init__(self, root_dir, graph_type="carla_visualize/*.png"):
        self.root_dir = Path(root_dir).resolve()
        self.resize = (500, 375)
        self.folder_paths =  [f for f in self.root_dir.iterdir() if f.stem.isnumeric()]
        self.folder_paths = sorted(self.folder_paths, key=lambda x : int(x.stem))

        self.lane_change_index = 0
        self.frame_index = 0
        self.graph_index = 0
        self.graph_type = graph_type

        self.root = Tk()
        self.canvas = Canvas(self.root, width = 400, height = 300)
        self.canvas.pack()
        
        self.load_paths()
        self.load_images()

        self.raw_image = ImageTk.PhotoImage(self.raw_ims[0])
        self.graph_image = ImageTk.PhotoImage(self.graph_ims[0])

        self.raw_label = Label(self.canvas, image=self.raw_image)
        self.raw_label.grid(row=0, column=0, columnspan=2)
        self.graph_label = Label(self.canvas, image=self.graph_image)
        self.graph_label.grid(row=0, column=2, columnspan=2)

        Button(self.canvas, text='Prev Clip', command=self.prevClip).grid(row=1, column=0)
        Button(self.canvas, text='Prev Frame', command=self.prevFrame).grid(row=1, column=1)
        Button(self.canvas, text='Next Frame', command=self.nextFrame).grid(row=1, column=2)
        Button(self.canvas, text='Next Clip', command=self.nextClip).grid(row=1, column=3)
        
        self.entry = Entry(self.canvas)
        self.entry.grid(row=2, column=1)
        Button(self.canvas, text='Jump', command=self.jump2Idx).grid(row=2, column=2)

    def load_paths(self):
        self.graph_paths = list(self.folder_paths[self.lane_change_index].glob(self.graph_type))
        self.image_paths = []
        image_frames = [i.stem for i in self.folder_paths[self.lane_change_index].glob("raw_images/*.jpg")]
        for path in self.graph_paths:
            img_path = path.parent.parent / "raw_images" / (path.stem + ".jpg")
            if path.stem in image_frames:
                self.image_paths.append(img_path)
            else:
                self.image_paths.append(path)

    def get_image(self, path):
        image = Image.open(path)
        image = image.resize(self.resize, Image.ANTIALIAS)
        return image

    def load_images(self):
        self.raw_ims = [self.get_image(path) for path in self.image_paths]
        self.graph_ims = [self.get_image(path) for path in self.graph_paths]

    def destroy_images(self):
        for im in self.raw_ims:
            del im
        for im in self.graph_ims:
            del im

    def update_title(self):
        self.root.title("Lane Change ID: {} / {}, Frame: {} / {}".format(self.folder_paths[self.lane_change_index].stem, 
                                                                        self.folder_paths[-1].stem, self.frame_index, 
                                                                        len(self.graph_paths) - 1))

    def prevClip(self):
        if self.lane_change_index > 0:
            self.lane_change_index -= 1
            self.update_clip()

    def nextClip(self):
        if self.lane_change_index < len(self.folder_paths) - 1:
            self.lane_change_index += 1
            self.update_clip()

    def jump2Idx(self):
        jmp_idx = int(eval(self.entry.get()))
        if (0 <= jmp_idx < len(self.folder_paths)):
            self.lane_change_index = jmp_idx
            self.update_clip()
        else:
            print("Error: Invalid index!")
        self.entry.delete(0, END)

    def prevFrame(self):
        if self.frame_index > 0:
            self.frame_index -= 1
            self.graph_index -= 1
            self.update_images()
            self.update_title()

    def nextFrame(self):
        if self.frame_index < len(self.image_paths) - 1:
            self.frame_index += 1
            self.graph_index += 1
            self.update_images()
            self.update_title()

    def update_images(self):
        self.raw_image.paste(self.raw_ims[self.frame_index])
        self.graph_image.paste(self.graph_ims[self.graph_index])

    def update_clip(self):
        self.frame_index = 0
        self.destroy_images()
        self.load_paths()
        self.load_images()
        self.update_images()
        self.update_title()

    def start(self):
        self.update_title()
        self.root.mainloop()

if __name__ == "__main__":
    gui = GUI("/home/louisccc/NAS/louisccc/av/synthesis_data/new_recording_3/")
    gui.start()