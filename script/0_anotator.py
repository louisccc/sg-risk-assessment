import imageio
from pathlib import Path
import sys, os
from argparse import ArgumentParser
from tqdm import tqdm
from tkinter import *
from PIL import Image, ImageTk
Image.DEBUG = 0


class Config:

    def __init__(self, args):
        self.parser = ArgumentParser(description='The parameters for creating gifs of input videos.')
        self.parser.add_argument('--input_path', type=str, default="/home/aung/NAS/louisccc/av/synthesis_data/lane-change-100-old/", help="Path to data directory.")

        args_parsed = self.parser.parse_args(args)
        
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

        self.input_base_dir = Path(self.input_path).resolve()


def show_video(canvas, clip_folder):
	im = Image.open(str(clip_folder / "lane_change.gif"))
	UI(canvas, im).grid(row=0)

def anotate_task(root_folder):
    foldernames = [f for f in root_folder.iterdir() if f.stem.isnumeric()]
    foldernames = sorted(foldernames, key=lambda x : int(x.stem))

    idx = -1

    def nextClip():
        print("Loading next clip...")
        nonlocal idx
        idx += 1
        if (idx >= len(foldernames)):
            root.destroy()
        clip_canvas.delete('all')
        root.title("Lane Change {} Evaluation: Clip {} / {}".format(foldernames[idx].stem, idx, len(foldernames)))
        show_video(clip_canvas, foldernames[idx])

    def replayClip():
        clip_canvas.delete('all')
        show_video(clip_canvas, foldernames[idx])

    def saveScore():  
        x = int(eval(entry.get()))
        if (1 <= x <= 5):
            score = x - 3
            label_file = foldernames[idx] / "label.txt"
            prev_avg_score = 0
            num_of_scores = 0
            if label_file.exists():
                with open(str(label_file), 'r') as f:
                    label_data = [int(x) for x in f.read().split(',')]
                    if len(label_data) == 2:
                        prev_avg_score, num_of_scores = label_data
            
            avg_score = ((prev_avg_score * num_of_scores) + score) / (num_of_scores + 1)
            num_of_scores += 1

            with open(str(label_file), 'w') as f:
                f.write("{}, {}".format(int(avg_score), num_of_scores))
            
            entry.delete(0, END)
            nextClip()
        else:
            print("Error: Score not in range.")

    root = Tk()
    clip_canvas = Canvas(root, width = 400, height = 300)
    clip_canvas.pack()
    util_canvas = Canvas(root, width = 400, height = 200)
    util_canvas.pack()
    Label(util_canvas, text="Enter a score from 1-5: ").grid(row=0)
    entry = Entry(util_canvas)
    entry.grid(row=1)
    Button(util_canvas, text='Save Score', command=saveScore).grid(row=1, column=1)
    Button(util_canvas, text='Replay Clip', command=replayClip).grid(row=2, column=0)
    Button(util_canvas, text='Next Clip', command=nextClip).grid(row=2, column=1)
    root.mainloop()

class AppletDisplay:
    def __init__(self, ui):
        self.__ui = ui
    def paste(self, im, bbox):
        self.__ui.image.paste(im, bbox)
    def update(self):
        self.__ui.update_idletasks()

# --------------------------------------------------------------------
# an image animation player

class UI(Label):

    def __init__(self, master, im):
        if type(im) == type([]):
            # list of images
            self.im = im[1:]
            im = self.im[0]
        else:
            # sequence
            self.im = im

        if im.mode == "1":
            self.image = ImageTk.BitmapImage(im, foreground="white")
        else:
            self.image = ImageTk.PhotoImage(im)

        # APPLET SUPPORT (very crude, and not 100% safe)
        global animation_display
        animation_display = AppletDisplay(self)

        Label.__init__(self, master, image=self.image, bg="black", bd=0)

        self.update()

        try:
            duration = im.info["duration"]
        except KeyError:
            duration = 100
        self.after(duration, self.next)

    def next(self):

        if type(self.im) == type([]):

            try:
                im = self.im[0]
                del self.im[0]
                self.image.paste(im)
            except IndexError:
                return # end of list

        else:

            try:
                im = self.im
                im.seek(im.tell() + 1)
                self.image.paste(im)
            except EOFError:
                return # end of file

        try:
            duration = im.info["duration"]
        except KeyError:
            duration = 100
        self.after(duration, self.next)

        self.update_idletasks()


if __name__ == "__main__":
	config = Config(sys.argv[1:])
	anotate_task(config.input_base_dir)
	

