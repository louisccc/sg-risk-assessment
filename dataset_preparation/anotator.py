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
    im = []
    image_list = list(clip_folder.glob("raw_images/*.jpg"))
    for img in image_list:
        im.append(Image.open(str(img)))
    UI(canvas, im, image_list).grid(row=0)

def anotate_task(root_folder):
    foldernames = [f for f in root_folder.iterdir() if f.stem.isnumeric()]
    foldernames = sorted(foldernames, key=lambda x : int(x.stem))

    idx = -1

    def read_score(path):
        prev_avg_score = 0
        num_of_scores = 0
        if path.exists():
            with open(str(path), 'r') as f:
                label_data = [x for x in f.read().split(',')]
                if len(label_data) == 2:
                    prev_avg_score, num_of_scores = float(label_data[0]), int(label_data[1])
        return prev_avg_score+3, num_of_scores

    def nextClip():
        print("Loading next clip...")
        nonlocal idx
        idx += 1
        if (idx >= len(foldernames)):
            root.destroy()
        clip_canvas.delete('all')
        
        prev_avg_score, num_of_scores = read_score(foldernames[idx] / "label.txt")
        
        root.title("Lane Change {} Evaluation: Clip {} / {}, curr:({}, {})".format(foldernames[idx].stem, idx, len(foldernames), prev_avg_score, num_of_scores))
        show_video(clip_canvas, foldernames[idx])
    
    def prevClip():
        print("Loading next clip...")
        nonlocal idx
        if (idx > 0):
            idx -= 1
            clip_canvas.delete('all')

            prev_avg_score, num_of_scores = read_score(foldernames[idx] / "label.txt")

            root.title("Lane Change {} Evaluation: Clip {} / {}, curr:({}, {})".format(foldernames[idx].stem, idx, len(foldernames), prev_avg_score, num_of_scores))
            show_video(clip_canvas, foldernames[idx])

    def replayClip():
        clip_canvas.delete('all')

        prev_avg_score, num_of_scores = read_score(foldernames[idx] / "label.txt")
        root.title("Lane Change {} Evaluation: Clip {} / {}, curr:({}, {})".format(foldernames[idx].stem, idx, len(foldernames), prev_avg_score, num_of_scores))
        show_video(clip_canvas, foldernames[idx])

    def jump2Idx():
        jmp_idx = int(eval(entry.get()))
        nonlocal idx
        if (0 <= jmp_idx < len(foldernames)):
            idx = jmp_idx
            replayClip()
        else:
            print("Error: Invalid index!")

    def saveScore():  
        x = int(eval(entry.get()))
        if (1 <= x <= 5):
            score = x - 3
            label_file = foldernames[idx] / "label.txt"
            prev_avg_score = 0
            num_of_scores = 0
            if label_file.exists():
                with open(str(label_file), 'r') as f:
                    label_data = [x for x in f.read().split(',')]
                    if len(label_data) == 2:
                        prev_avg_score, num_of_scores = float(label_data[0]), int(label_data[1])
            
            avg_score = ((prev_avg_score * num_of_scores) + score) / (num_of_scores + 1)
            num_of_scores += 1
            
            print("%f stored to %s" %(avg_score, label_file))
            with open(str(label_file), 'w') as f:
                f.write("{}, {}".format(avg_score, num_of_scores))
            
            entry.delete(0, END)
            nextClip()
        else:
            print("Error: Score not in range.")

    root = Tk()
    clip_canvas = Canvas(root, width = 400, height = 300)
    clip_canvas.pack()
    util_canvas = Canvas(root, width = 400, height = 200)
    util_canvas.pack()
    Label(util_canvas, text="Enter a score from 1-5(safe-risk): ").grid(row=0)
    entry = Entry(util_canvas)
    entry.grid(row=1)
    Button(util_canvas, text='Save Score', command=saveScore).grid(row=1, column=1)
    Button(util_canvas, text='Jump to Index', command=jump2Idx).grid(row=1, column=2)
    Button(util_canvas, text='Replay Clip', command=replayClip).grid(row=2, column=0)
    Button(util_canvas, text='Prev Clip', command=prevClip).grid(row=2, column=1)
    Button(util_canvas, text='Next Clip', command=nextClip).grid(row=2, column=2)
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

    def __init__(self, master, im, image_path_list):
        if type(im) == type([]):
            # list of images
            self.im = im
            im = self.im[0]
            self.image_path_list = image_path_list
            self.index = 0
        else:
            # sequence
            self.im = im

        if im.mode == "1":
            self.image = ImageTk.BitmapImage(self.im[0], foreground="white")
        else:
            self.image = ImageTk.PhotoImage(self.im[0])

        # APPLET SUPPORT (very crude, and not 100% safe)
        global animation_display
        animation_display = AppletDisplay(self)

        Label.__init__(self, master, image=self.image, bg="black", bd=0)
        self.grid(row=0, columnspan=4)
        Button(master, text='Prev Frame', command=self.previousFrame).grid(row=1, column=0)
        Button(master, text='Pause', command=self.pause).grid(row=1, column=1)
        Button(master, text='Resume', command=self.resume).grid(row=1, column=2)
        Button(master, text='Next Frame', command=self.nextFrame).grid(row=1, column=3)
        Button(master, text="Del Before", command=self.deleteBefore).grid(row=2, column=1)
        Button(master, text="Del After", command=self.deleteAfter).grid(row=2, column=2)

        self.update()

        try:
            duration = im.info["duration"]
        except KeyError:
            duration = 100
        self.paused = False
        self.after(duration, self.next)
    
    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        try:
            im = self.im[self.index]
            duration = im.info["duration"]
        except KeyError:
            duration = 100
        self.after(duration, self.next)

    def previousFrame(self):
        if self.paused and self.index > 0:
            self.index -= 1
            im = self.im[self.index]
            self.image.paste(im)
    
    def nextFrame(self):
        if self.paused and (self.index < len(self.image_path_list) - 1):
            self.index += 1
            im = self.im[self.index]
            self.image.paste(im)
    
    def deleteBefore(self):
        # delete every images before current index
        if self.paused:
            delete_path_list = self.image_path_list[:self.index]

            for img_path in delete_path_list:
                img_path.unlink()
            
            self.im = self.im[self.index:]
            self.image_path_list = self.image_path_list[self.index:]
            self.index = 0

    def deleteAfter(self):
        # delete every images after current index
        if self.paused:
            delete_path_list = self.image_path_list[self.index + 1:]

            for img_path in delete_path_list:
                img_path.unlink()
            
            self.im = self.im[:self.index + 1]
            self.image_path_list = self.image_path_list[:self.index + 1]
            self.index = 0

    def next(self):
        if not self.paused:
            if type(self.im) == type([]):

                try:
                    self.index += 1
                    im = self.im[self.index]
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
	

