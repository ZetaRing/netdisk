import numpy as np
import multiprocessing
from typing import List
import json
import os
import pandas as pd
import copy
from collections import defaultdict
import itertools
from tqdm import tqdm
import librosa
from datetime import datetime


# TODO: include evaluation set
class AudioSet(object):
    def __init__(self, training_set=True):
        self.dir_path = "/media/zfchen/048c6a59-054e-45ae-aa0e-1e437096b891/zeta/audio_set"

        # Load
        ontology = json.load(open(os.path.join(self.dir_path, "ontology.json"), "r"))
        if training_set:
            meta_file = "unbalanced_train_segments.csv"
        else:
            meta_file = "eval_segments.csv"
        meta = pd.read_csv(os.path.join(self.dir_path, meta_file), sep=", ", engine='python', skiprows=2)

        # Set selected categories
        # Ontology is a graph, not a tree. query_handles is visible tree roots.
        query_handles = ["Music", "Sounds of things"]
        valid_cate = ["Musical instrument", "Domestic sounds, home sounds", "Liquid", "Glass", "Printer",
                      "Air conditioning", "Mechanical fan", "Clock", "Fire alarm", "Smoke detector, smoke alarm",
                      "Doorbell", "Alarm clock", "Ringtone", "Telephone bell ringing", "Domestic sounds, home sounds",
                      "Loudspeaker", "Radio", "Television", "MP3", "Domestic animals, pets"]
        block_cate = ["Human sounds", "Vehicle"]

        # Put audios on the node.
        name2node = {x["name"]: x["id"] for x in ontology}
        node2child = {x["id"]: x["child_ids"] for x in ontology}
        valid_nodes = self.iterative_query([name2node[x] for x in valid_cate], query_dict=node2child)
        block_nodes = self.iterative_query([name2node[x] for x in block_cate], query_dict=node2child)

        self._node2audio = defaultdict(list)
        for id, labels in zip(meta["# YTID"], meta["positive_labels"]):
            labels = set(labels.strip('"').split(","))
            if len(labels & block_nodes): continue
            for i in (labels & valid_nodes):
                self._node2audio[i].append(id)

        # Pruning nodes without audios
        self.nodes = list()
        for i in [x["id"] for x in ontology]:
            nodes = self.iterative_query([i], node2child)
            if any(len(self._node2audio[x]) for x in nodes):
                self.nodes.append(i)

        query_nodes = self.iterative_query([name2node[x] for x in query_handles], query_dict=node2child)
        self.nodes = list(set(self.nodes) & query_nodes)

        filtered_ontology = [x for x in ontology if x["id"] in self.nodes]
        self.node2name = {x["id"]: x["name"] for x in filtered_ontology}
        self.node2description = {x["id"]: x["description"] for x in filtered_ontology}
        self.node2child = {x["id"]: (set(x["child_ids"]) & set(self.nodes)) for x in filtered_ontology}
        self.node2father = defaultdict(list)
        for k, v in self.node2child.items():
            for i in v:
                self.node2father[i].append(k)

        # Others
        self.audio_ids = self.get_ids(self.nodes)
        self.meta = meta[meta["# YTID"].isin(self.audio_ids)]
        self.downloader = os.path.join("third_party", "youtube-dl")
        print(f"AudioSet {meta_file}: {len(self.meta)} / {len(meta)}, cate {len(self.nodes)} / {len(ontology)}")

        # _str = self.meta.to_csv(index=False, sep="\t") # Stupid Lib
        # _str = _str.replace("\t", ", ")
        # with open(os.path.join(self.dir_path, f"filtered_{meta_file}"), "w") as f:
        #     f.write(_str)

        # Display
        # root_nodes = set(self.nodes).difference(set(itertools.chain.from_iterable(self.node2child.values())))
        # self.print_tree(root_nodes)

    def get_ids(self, nodes: List[str]):
        nodes = self.iterative_query(nodes, self.node2child)
        return list(set(itertools.chain.from_iterable(self._node2audio[x] for x in nodes)))

    def get_audio(self, audio_id):
        assert audio_id in self.audio_ids # YTID is unique in training set
        info = self.meta[self.meta["# YTID"] == audio_id]
        assert len(info) == 1
        _path = os.path.join(self.dir_path, f"{audio_id}.wav")
        if not os.path.exists(_path):
            os.system(f"sh {os.path.join('third_party', 'fetch_audio.sh')} "
                      f"{audio_id} {info['start_seconds'].values[0]} {info['end_seconds'].values[0]} "
                      f"{_path} {self.downloader}")
        else:
            print(f"skip {audio_id}")

        audio_data = None
        success = False
        # if os.path.exists(_path):
        #     audio_data, _ = librosa.load(_path, sr=config.RIR_SAMPLING_RATE)
        #     success = True
        return audio_data, success

    def check_length(self, audio_id):
        assert audio_id in self.audio_ids # YTID is unique in training set
        info = self.meta[self.meta["# YTID"] == audio_id]
        assert len(info) == 1
        _path = os.path.join(self.dir_path, f"{audio_id}.wav")

        if os.path.exists(_path):
            audio_data, _ = librosa.load(_path, sr=16000)
            _t = info['end_seconds'].values[0] - info['start_seconds'].values[0]
            print(f"{audio_id} {len(audio_data)} {_t}")
            # if len(audio_data) != (_t * 16000):
            #     print(f"{audio_id} {len(audio_data)} {_t}")
                # os.system(f"sh {os.path.join('third_party', 'fetch_audio.sh')} "
                #           f"{audio_id} {info['start_seconds'].values[0]} {info['end_seconds'].values[0]} "
                #           f"{_path} {self.downloader}")
        return None, False

    @staticmethod
    def iterative_query(nodes: List[str], query_dict: dict[str, List[str]], include_root=True) -> set:
        q = copy.deepcopy(nodes)
        res = []
        while len(q):
            node = q.pop()
            res.append(node)
            q.extend(query_dict[node])

        if not include_root:
            for i in nodes:
                res.remove(i)
        return set(res)

    def print_tree(self, nodes, max_depth=100):
        q = []
        for i in nodes:
            q.append((i, 0))
        while len(q):
            node, depth = q.pop()
            for i in self.node2child[node]:
                q.append((i, depth+1))
            if depth < max_depth:
                num = len(set(itertools.chain.from_iterable(
                    self._node2audio[x] for x in self.iterative_query([node], self.node2child))))
                print(f'{"--" * depth} {self.node2name[node]}: {num}, {self.node2description[node]}')

    @property
    def meta_info(self):
        info = {}
        for i in self.nodes:
            path = self.iterative_query([i], self.node2father)
            tags = [self.node2name[x] for x in path]
            description = self.node2description[i]
            info[self.node2name[i]] = f"tags={tags}, description='{description}'"
        return info


if __name__ == "__main__":
    # TODO: spilt train and test set (use src_file label for audio files)
    audio_set = AudioSet(training_set=True)
    pool = multiprocessing.Pool(12)

    _dir = "/media/zfchen/048c6a59-054e-45ae-aa0e-1e437096b891/zeta/audio_set"
    _f = [i.strip(".wav") for i in os.listdir(_dir) if i.endswith(".wav")]
    _f = list(set(_f) & set(audio_set.audio_ids))
    print(len(_f))
    r = list(tqdm(pool.imap(audio_set.check_length, _f), total=len(_f)))
    # r = list(tqdm(pool.imap(audio_set.check_length, audio_set.audio_ids), total=len(audio_set.audio_ids)))


    # def func(x):
    #     os.system('duration=$(ffprobe "' + x + '" 2>&1 | awk "/Duration/ { print $2 }") && echo -e "$duration\t"' +x)
    # r = list(tqdm(pool.map(func, _f), total=len(_f)))

    # with open("all.txt") as f:
    #     all = f.readlines()
    # all = [x.strip().removesuffix(".wav") for x in all]
    #
    # with open("empty.txt") as f:
    #     empty = f.readlines()
    # empty = [x.strip().removesuffix(".wav").removeprefix("/media/zfchen/048c6a59-054e-45ae-aa0e-1e437096b891/zeta/audio_set/") for x in empty]
    #
    # with open("list.txt") as f:
    #     duration = f.readlines()
    # duration = [x.strip() for x in duration]
    #
    # date_format = '%H:%M:%S.%f'
    # std = datetime.strptime("00:00:10.00", date_format)
    # dur = []
    # wro = []
    # for i in duration:
    #     d, w = i.split("\t")
    #     w = w.removesuffix(".wav").removeprefix("/media/zfchen/048c6a59-054e-45ae-aa0e-1e437096b891/zeta/audio_set/")
    #     d = d.removeprefix("-e ").removesuffix(",")
    #     dur.append(w)
    #
    #     if (len(d) != 0) and (d != "N/A"):
    #         d = datetime.strptime(d, date_format)
    #         if d != std:
    #             wro.append(w)
    #     else:
    #         wro.append(w)
    # pass