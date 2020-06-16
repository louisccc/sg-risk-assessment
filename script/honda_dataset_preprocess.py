from tqdm import tqdm
from pathlib import Path


import argparse, glob, cv2, pprint
import pickle as pkl


class HondaDataSetHandler:
    def __init__(self):
        # ############## Global Parameters ##############
        self.root = Path(r"/home/aung/NAS/louisccc/av/honda_data/release_2019_01_20").resolve()
        self.session_template = "{0}/{1}_{2}_{3}_ITS1/{4}/"
        self.annotation_dir = self.root / "EAF"
        self.sampling_frequency = 3
        self.video_framerate = 30
        self.index_path = Path(r"/home/aung/NAS/louisccc/av/honda_data/EAF_parsing/saved_index.pkl").resolve()
        self.cache_data = pkl.load(open(str(self.index_path), 'rb'))
        self.dest = Path(r'/home/aung/NAS/louisccc/av/honda_data/lane-change-clips').resolve()

        print_metadata(cfg.cache_data)

    def print_metadata(self):
        # layer_ix is the 4-layer representation proposed in the paper
        # Goal: 0, Stimulus: 6; Cause: 1; Attention: 3, 5; the rest are used for additional note: 2, 4
        pprint.pprint(self.cache_data['layer_ix'])
        
        # event_type_ix is all the categories defined in the 4-layer representation
        # Event_type_ix 0~12 belongs to those events defined in "Goal-oriented" layer. Note that we exclude 9 from the experiments conducted in CVPR'18.
        # Event type_ix 16 (congestion), 17 (sign), 18 (red light), 19 (crossing vehicle), 20 (parked vehicle), 22 (crossing pedestrian) 
        # are those objects annotated in "Cause" layer 
        pprint.pprint(self.cache_data['event_type_ix'])

    def get_video_path(self, session_id):
        session_template = self.session_template  # "{0}/{1}_{2}_{3}_ITS_data_collection/{4}_ITS/"
        preview_template = session_template.format(self.root,
                                                session_id[:4],
                                                session_id[4:6],
                                                session_id[6:8],
                                                session_id) + "camera/center/*.mp4"
        video_full_path = glob.glob(preview_template)[0]
        
        return video_full_path

    def capture_video_clips(self, event_types):

        events = self.cache_data['events_pd'].loc[self.cache_data['events_pd']["event_type"].isin(event_types)]

        frame_padding = 30
        frame_rate = self.video_framerate
        sampling_freq = self.sampling_frequency
        
        for idx, row in tqdm(events.iterrows()):
            start, end = int(row["start"] / 1000 * frame_rate), int(row["end"] / 1000 * frame_rate)
            session_id = row['session_id']

            out_path =  self.dest / (str(idx)+"_"+ session_id)
            out_path.mkdir(exist_ok=True)

            video_path = self.get_video_path(session_id)

            cap = cv2.VideoCapture(video_path)
            
            for frame_no in range(start - frame_padding, end + frame_padding, sampling_freq):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                outname = out_path /  (str(frame_no)+'.jpg')
                # frame = cv2.resize(frame,(1024,1024))
                cv2.imwrite(str(outname), frame)
            
            cap.release()


if __name__ == "__main__":
    data_handler = HondaDataSetHandler()
    event_types = [3, 5]
    data_handler.capture_video_clips(event_types)
    