#!/usr/bin/env python3

import os
import struct
import threading
import time
import collections
import random
import pickle
import yaml

# 00 string Level;
# 08 int Chapter;
# 0c int Mode;
# 10 bool TimerActive;
# 11 bool ChapterStarted;
# 12 bool ChapterComplete;
# 18 long ChapterTime;
# 20 int ChapterStrawberries;
# 24 bool ChapterCassette;
# 25 bool ChapterHeart;
# 28 long FileTime;
# 30 int FileStrawberries;
# 34 int FileCassettes;
# 38 int FileHearts;
# 40 int CurrentChapterCheckpoints;

# get autosplitter info path
asi_path = os.environ.get('ASI_PATH', '/dev/shm/autosplitterinfo')


# check if the selected file is pickle or yaml, and load it as necessary
# NOTE: this is used for route, pb and gold files
def open_pickle_or_yaml(filename):
    try:
        with open(filename, 'rb') as fp:
            return pickle.load(fp)
    except pickle.PickleError:
        try:
            with open(filename, 'r', encoding='utf-8') as fp:
                return yaml.load(fp, Loader=MyUnsafeLoader)  # yikes!!
        except yaml.YAMLError:
            raise TypeError("Cannot load this file as either pickle or yaml")


# save the yaml file after exiting
def save_yaml(filename, data):
    with open(filename, 'w', encoding='utf-8') as fp:
        yaml.dump(data, fp, Dumper=MyDumper)

# patching around yaml dumper and setting ignore_aliases to True
# for more info: https://stackoverflow.com/a/58091449
class MyDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True


# hack around https://github.com/yaml/pyyaml/issues/620
class MyUnsafeConstructor(yaml.constructor.UnsafeConstructor):
    def construct_object(self, node, deep=False):
        return super().construct_object(node, deep=True)


# a yaml loader, using the unsafe constructor above
class MyUnsafeLoader(
        yaml.reader.Reader,
        yaml.scanner.Scanner,
        yaml.parser.Parser,
        yaml.composer.Composer,
        MyUnsafeConstructor,
        yaml.resolver.Resolver
):
    def __init__(self, stream):
        yaml.reader.Reader.__init__(self, stream)
        yaml.scanner.Scanner.__init__(self)
        yaml.parser.Parser.__init__(self)
        yaml.composer.Composer.__init__(self)
        MyUnsafeConstructor.__init__(self)
        yaml.resolver.Resolver.__init__(self)


# this function is used by yaml module to serialize the data, so that older and newer pickle files are equally compatible with this program, and this enables us to maintain only one set of de/serialization functions
def represent_pickle(self, data):
    data_type = type(data)
    tag = 'tag:yaml.org,2002:python/object:%s.%s' % (
        data_type.__module__, data_type.__name__)
    value = data.__getstate__()
    return self.represent_mapping(tag, value)


# convert filetime into hours, minutes, seconds, milliseconds and return as a tuple
def split_time(filetime):
    neg = filetime < 0
    if neg:
        filetime = -filetime
    ms = filetime % 1000
    se = filetime // 1000 % 60
    mi = filetime // 1000 // 60 % 60
    hr = filetime // 1000 // 60 // 60
    return (neg, hr, mi, se, ms)


# format filetime as human-readable +/- hh:mm:ss.ms
def fmt_time(tup, ms_decimals=3, full_width=False, sign=False):
    if tup is None:
        return None
    if type(tup) is int:
        tup = split_time(tup)

    neg, hr, mi, se, ms = tup
    if ms_decimals > 0:
        if ms_decimals == 1:
            ms //= 100
        elif ms_decimals == 2:
            ms //= 10
        ms_str = ('.%%0%dd' % ms_decimals) % ms
    else:
        ms_str = ''

    if hr or mi or full_width:
        se_str = '%02d' % se
    else:
        se_str = '%d' % se

    if hr or full_width:
        mi_str = '%02d:' % mi
    else:
        if mi:
            mi_str = '%d:' % mi
        else:
            mi_str = ''

    if hr or full_width:
        hr_str = '%d:' % hr
    else:
        hr_str = ''

    if sign or neg:
        sign_str = '-' if neg else '+'
    else:
        sign_str = ''

    return sign_str + hr_str + mi_str + se_str + ms_str


# convert hh:mm:ss.ms into filetime
def parse_time(string):
    if string is None:
        return None
    try:
        hms, ms = string.split('.')
        hr, mi, se = hms.split(':')
        ms = int(ms)
        hr = int(hr)
        mi = int(mi)
        se = int(se)
    except Exception as e:
        raise TypeError(
            "Cannot parse %s as time - expected format is 00:00:00.000") from e

    return ms + se * 1000 + mi * 1000 * 60 + hr * 1000 * 60 * 60


class AutoSplitterInfo: # this class has the function to read the autosplitter info and get all the current values e.g. chapter name, death count, chapter time etc. from it
    def __init__(self, filename=asi_path):
        self.all_attrs = ('chapter', 'mode', 'timer_active', 'chapter_started', 'chapter_complete', 'chapter_time', 'chapter_strawberries', 'chapter_cassette',
                          'chapter_heart', 'file_time', 'file_strawberries', 'file_cassettes', 'file_hearts', 'chapter_checkpoints', 'in_cutscene', 'death_count', "level_name")
        self.chapter = 0
        self.mode = 0
        self.timer_active = False
        self.in_cutscene = False
        self.death_count = 0
        self.level_name = ""

        self.chapter_started = False
        self.chapter_complete = False
        self.chapter_time = 0
        self.chapter_strawberries = 0
        self.chapter_cassette = False
        self.chapter_heart = False
        self.chapter_checkpoints = 0

        self.file_time = 0
        self.file_strawberries = 0
        self.file_cassettes = 0
        self.file_hearts = 0

        if not os.path.exists(filename):
            print('waiting for', filename, '...')
            while not os.path.exists(filename):
                time.sleep(1)

        self.fp = open(filename, 'rb') # load /dev/shm/autosplitterinfo as read-only binary file object
        self.live = True

        self.thread = threading.Thread(target=self.update_loop)
        self.thread.daemon = True
        self.thread.start()

    @property
    # getting the chapter name (for interpreting chapter 0 as 'Prologue' and 8 as 'Epilogue'), and also the level side
    def chapter_name(self):
        if self.chapter == 0:
            return 'Prologue'
        if self.chapter == 8:
            return 'Epilogue'
        if self.chapter == 10:
            return '9'
        if self.mode == 0:
            side = 'a'
        elif self.mode == 1:
            side = 'b'
        else:
            side = 'c'
        return '%d%s' % (self.chapter, side)

    def __getitem__(self, k):
        try:
            return getattr(self, k)
        except AttributeError as e:
            raise KeyError(k) from e

    @property
    def dict(self):
        return {x: getattr(self, x) for x in self.all_attrs}

    # getting value from autosplitterinfo file every 0.001s by unpacking the framebuffer
    def update_loop(self):
        fmtstring = struct.Struct('Qii???QI??QIIIxxxxI?i100s') # a framebuffer of 176 bytes, i suppose
        while self.live:
            last_tick = time.time()
            self.fp.seek(0)
            dat = self.fp.raw.read(fmtstring.size) # read the framebuffer from /dev/shm/autosplitterinfo
            _, self.chapter, self.mode, self.timer_active, \
                self.chapter_started, self.chapter_complete, \
                chapter_time, self.chapter_strawberries, \
                self.chapter_cassette, self.chapter_heart, file_time, \
                self.file_strawberries, self.file_cassettes, self.file_hearts, \
                self.chapter_checkpoints, self.in_cutscene, self.death_count, level_name \
                = fmtstring.unpack(dat) # unpack the framebuffer into property variables

            self.chapter_time = chapter_time // 10000 # formatting chapter time gotten from autosplitterinfo
            self.file_time = file_time // 10000 # formatting file time gotten from autosplitterinfo
            self.level_name = level_name.split(b'\0')[0].decode() # get utf-8 string from level name data

            timeout = last_tick + 0.001 - time.time()
            if timeout > 0:
                time.sleep(timeout)


class Trigger:
    def __init__(self, name, end_trigger):
        self.name = name
        self.end_trigger = end_trigger

    def check_trigger(self, asi):  # pylint: disable=unused-argument
        return eval(self.end_trigger)  # pylint: disable=eval-used
        # ?

    def __repr__(self): # formats trigger name
        return '<Trigger %s>' % self.name


class Split: # an important class for editing splits
    def __init__(self, names, level=0):
        if type(names) == str:
            names = [names] # in the previous versions of this program, the splits used to have only one name, but now they can have more than one, so this maintains backwards compatibility by loading multiple names into a list
        if len(names) == 0:
            raise ValueError("Need at least one name")
        self.names = names
        self.level = level
        self.identity = random.randrange(2**64)

    def level_name(self, level): # levels are a kind of hierarchy inside splits, for example, chapter splits are level 0 and checkpoint splits are level 1, and this functions the level of the split
        if level < self.level:
            raise ValueError(
                "Why are you trying to render %s at level %d?" % (self, level))
        try:
            return self.names[level - self.level]
        except IndexError:
            return self.names[-1]
    

    """
    reference: https://rszalski.github.io/magicmethods/
    basically, these are called magic methods, and they are used to implement the functionality of the operators +, -, *, /, **, etc.
    """

    def __eq__(self, other): # this checks if two splits' identity property is equal, meaning it checks if they are the same split
        return hasattr(other, 'identity') and self.identity == other.identity

    def __hash__(self): # this hashes a given split, to quickly check if two keys in (probably) two dict objects are equal
        return hash(self.identity)

    def __repr__(self): # this sets what the string representation of a split class object is
        return '<Split %s>' % self.names[0]

    def __getstate__(self): # for pickling (meaning that the split class object can be saved to a file), the split is serialized into a dictionary, and the dictionary is returned
        return self.__dict__

    def __setstate__(self, state): # for unpickling (meaning that the split class object can be loaded from a file), the split is deserialized from a dictionary, the split names are loaded from a list to a dictionary in reverse order (that's why the pop() function is used, i suppose), and the dictionary is returned
        if 'name' in state:
            state['names'] = [state.pop('name')]
        self.__dict__.update(state)


yaml.representer.Representer.add_representer(Split, represent_pickle) # setting yaml equivalent pickle for autosplitter splits? or parsing Split class object using represent_pickle?


# this StartTimer class has the str representation as <StartTimer>, and when the autosplitter detects this are the current split, the timer resets to zero. this hasnt been used anywhere as this was implemented at the start of the code's development, and it was mainly used for 1a arb runs. ig it can be useful for my chapterwise practice, maybe?
class StartTimer: 
    def __repr__(self):
        return '<StartTimer>'


notpassed = object() # this is a blank object which will never be used unless there is some kind of exceptional error in the program and then this object will be used to detect that something exceptional happened

"""
NOTE: we should clarify what 'pb' and 'gold' means in this program's context.
here pb means your best total time, and the splitsrecord class processes this data. this data, after pickling or serializing, is stored in a ,pb file.
on the other hand, gold means a run composed of all the best checkpoint and chapter times, and the goldsrecord class processes this data. this data, after pickling or serializing, is stored in a ,gold file. 
"""

# this class interacts with the pb splits
class SplitsRecord(collections.OrderedDict): # declaring that this class takes an ordered dictionary as parameter
    def segment_time(self, split, level=0, fallback=notpassed): # returns the time of the split (here split can mean either chapters or checkpoints, depends on the implementation), or the time of the previous split if the split doesn't exist?
        found_prev = None
        for cur in self:
            if cur == split:
                break
            if cur.level <= level:
                found_prev = cur
        else:
            if fallback is not notpassed:
                return fallback
            raise KeyError(split)

        if found_prev is None:
            return self[split]
        elif self[split] is None or self[found_prev] is None:
            return None
        else:
            return self[split] - self[found_prev]

    # serializing the new pb splits that the SplitsRecord class object has into a dict, for saving as pickle or yaml .pb file
    # source: https://forums.fast.ai/t/what-do-getstate-and-setstate-do/76128
    def __getstate__(self): 
        return {
            'version': 1,
            'splits': {split: fmt_time(time, full_width=True) for split, time in self.items()}
        }

    # deserializing the pb splits that the SplitsRecord class object has from the dict, for loading from pickle or yaml .pb file
    def __setstate__(self, state):
        if type(state) is not dict:
            raise TypeError(
                "Cannot deserialize this SplitsRecord - are you sure it's a record file?")
        version = state.get('version', 0)
        if version == 0:
            self.__dict__.update(state)
        elif version == 1:
            self.__init__({split: parse_time(time)
                          for split, time in state['splits'].items()})
        else:
            raise TypeError(
                "Cannot deserialize this SplitsRecord - try updating the autosplitter")

    def update_identity(self, route):
        """
        Replace the splits here with the splits from the route with the same identity, meaning with this we update the the split times found in the pb file with the new pb splits having the same identity. e.g. if the .pb file has the split time of prologue as 31s and in the new run, we got a new pb record and that has the split time of prologue as 32s, then we update the .pb file with the new split time of prologue as 32s.
        """
        collection = dict(self)
        self.clear()
        for split in route.splits:
            self[split] = collection.get(split, None)


yaml.representer.Representer.add_representer(SplitsRecord, represent_pickle) # again, a yaml equivalent for saving and reading splitsrecord files


# this class interacts with the golds splits
class GoldsRecord(collections.UserDict): # so the GoldsRecord class object is a user defined dictionary
    
    # serializing the new gold splits that the GoldsRecord class object has into a dict, for saving as pickle or yaml .gold file
    def __getstate__(self):
        return {
            'version': 1,
            'splits': {split: fmt_time(time, full_width=True) for split, time in self.items()}
        }

    # deserializing the gold splits that the GoldsRecord class object has from the dict, for loading from pickle or yaml .gold file
    def __setstate__(self, state):
        if type(state) is not dict:
            raise TypeError(
                "Cannot deserialize this GoldsRecord - are you sure it's a golds file?")
        version = state.get('version', 0)
        if version == 1:
            self.__init__({split: parse_time(time)
                          for split, time in state['splits'].items()})
        else:
            raise TypeError(
                "Cannot deserialize this GoldsRecord - try updating the autosplitter")

    def update_identity(self, route):
        """
        Replace the splits here with the splits from the route with the same identity, meaning with this we update the the split times found in the golds file with the new gold splits having the same identity. e.g. if the .gold file has the split time of prologue as 31s and in the new run, we got a new gold record and that has the split time of prologue as 30s, then we update the .gold file with the new split time of prologue as 30s.
        """
        collection = dict(self)
        self.clear()
        for segment in route.all_subsegments:
            self[segment] = collection.get(segment, None)


yaml.representer.Representer.add_representer(GoldsRecord, represent_pickle)

# this is the class that interacts with the route file
class Route(collections.UserList): # the route class object is a user defined list
    
    # setting class variables from the route list object
    def __init__(self, name, time_field, pieces, level_names, reset_trigger):
        if type(pieces[-1]) is not Split or pieces[-1].level != 0:
            raise TypeError("Last piece of route must be top-level Split")
        super().__init__(pieces)
        self.name = name
        self.time_field = time_field
        self.levels = max(
            piece.level for piece in pieces if type(piece) is Split) + 1
        self.splits = [x for x in self if type(x) is Split]
        self.level_names = level_names
        self.reset_trigger = reset_trigger

    # serializing the route as a dict object, for saving as pickle or yaml route file, and this is probably used in edit_splits.py and make_*_splits.py
    def __getstate__(self):
        return {
            'version': 1,
            'name': self.name,
            'time_field': self.time_field,
            'level_names': self.level_names,
            'reset_trigger': self.reset_trigger,
            'pieces': list(self),
        }

    # deserializing the route file from a dict to a list of triggers and splits, for reading from a pickle or yaml file
    def __setstate__(self, state):
        if type(state) is dict:
            version = state.get("version", 0)
            if version == 0:
                self.__dict__.update(state)
            elif version == 1:
                self.__init__(state['name'], state['time_field'], state['pieces'],
                              state['level_names'], state['reset_trigger'])
            else:
                raise TypeError(
                    "Cannot deserialize this Route - try updating the autosplitter")
        elif len(state) == 3:
            self.__init__(state[1], state[2], state[0], [
                          'Segment', 'Subsegment'], None)
        elif len(state) == 5:
            self.__init__(state[1], state[2], state[0], state[3], state[4])
        else:
            raise TypeError(
                "Cannot deserialize this Route - are you sure it's a route file?")

    # this parses the list from __setstate__, and returns the split from the list that is of the specified level and contains the specified index
    def split_idx(self, i, level=0):
        while type(self[i]) is not Split or self[i].level > level:
            i += 1
            if i >= len(self):
                return None
        return self.splits.index(self[i])

    @property
    # defining the subsplits?
    def all_subsegments(self):
        prev = None
        for split in self.splits:
            if prev is not None:
                for level in range(prev.level, split.level, -1):
                    yield (split, level)

            # yield is like return, but it can be used in a for loop, or it can return different values different times
            # reference: https://docs.python.org/3/reference/expressions.html#yieldexpr
            yield (split, split.level)
            prev = split


"""
from rhelmot xemself:
"SplitsManager is the big guy, the class that observes an AutoSplitterInfo updating and uses that to tell where you are in the route, what your current splits are, whether a split is a gold, etc. All the methods on it are about answering questions about the current run, e.g. are we done? what's the next trigger we're waiting for? what's the next/previous split of a given level? It also has methods for updating clerical information about the current run, e.g. resetting the run, marking when you finish a split, noting when you get a gold split, etc."
couldn't have said it better myself.
"""
class SplitsManager:
    def __init__(self, asi, route, compare_pb=None, compare_best=None):
        self.asi = asi
        self.route = route
        # i've expanded the one liners that were here underneath, so I can understand what they are doing
        if compare_pb is not None:
            self.compare_pb = compare_pb
        else:
            self.compare_pb = SplitsRecord()
        if compare_best is not None:
            self.compare_best = compare_best
        else:
            self.compare_best = {}
        self.current_times = SplitsRecord()
        self.current_piece_idx = 0
        self.start_time = 0
        self.started = False

        # importing the golds file dict as a golds record object
        if type(self.compare_best) is dict:
            self.compare_best = GoldsRecord(self.compare_best)

        self.compare_pb.update_identity(self.route)
        self.compare_best.update_identity(self.route)

        parents = {}
        for split in self.route.splits:
            parents[split.level] = split

            if split not in self.compare_pb:
                self.compare_pb[split] = None
            else:
                self.compare_pb.move_to_end(split)

    @property
    # this returns True if the route is done
    def done(self):
        return self.current_piece_idx >= len(self.route)

    @property
    def current_piece(self):
        if self.done:
            return None
        return self.route[self.current_piece_idx]

    # this returns the index of the current split given the level of the split
    def _current_split_idx(self, level=0):
        idx = self.route.split_idx(self.current_piece_idx, level)
        if idx is None:
            return None
        while self.route.splits[idx].level > level:
            idx += 1
        return idx

    # this is probably called when the current split is done, and the run moves forward, so we have to update the current split index by adding 1
    def _forward_split(self, idx, level=0):
        idx += 1
        if idx >= len(self.route.splits):
            return None
        while self.route.splits[idx].level > level:
            idx += 1
            if idx >= len(self.route.splits):
                return None
        return idx

    # TODO: update what this does
    def _backwards_split(self, idx, level=0):
        idx -= 1
        if idx < 0:
            return None
        while self.route.splits[idx].level > level:
            idx -= 1
            if idx < 0:
                return None
        return idx

    # this returns the current split using the split index from the _current_split_idx function
    def current_split(self, level=0):
        if self.done:
            return None
        idx = self._current_split_idx(level)
        return self.route.splits[idx]

    # this returns what the previous split is, given the level of the split
    def previous_split(self, level=0):
        if self.done: # meaning if the run is done, then we return None
            idx = len(self.route.splits)
        else:
            idx = self._current_split_idx(level)
        idx = self._backwards_split(idx, level)
        if idx is None:
            return None
        return self.route.splits[idx]

    # this checks if the current split is done
    def is_segment_done(self, split):
        return self.current_piece_idx > self.route.index(split)

    @property
    # this subtracts the the start time from the current unix time to get the actual ingame elapsed time
    def current_time(self):
        return self.asi[self.route.time_field] - self.start_time

    # this basically does the same thing as the current_time function, but it subtracts the split start time from the current time
    def current_segment_time(self, level=0):
        if self.done:
            return None
        prev_split = self.previous_split(level)
        if prev_split is None:
            return self.current_time
        split_start = self.current_times[prev_split]
        if split_start is None:
            return None
        return self.current_time - split_start

    # TODO: update what this does
    def best_possible_time(self):
        return None

    # this gets the elapsed time of the current split
    def split(self, split):
        self.current_times[split] = self.current_time

    # commit() is used by the update() function to update the current split times, new pb and gold splits and so on and so forth 
    def commit(self):
        if self.route.splits[-1] in self.current_times:
            cur_time = self.current_times[self.route.splits[-1]]
            pb_time = self.compare_pb[self.route.splits[-1]]
            if pb_time is None or cur_time < pb_time:
                self.compare_pb = self.current_times

        # TODO: do we care about not mutating this reference?
        self.compare_best = GoldsRecord(self.compare_best)
        for key in self.route.all_subsegments:
            split, level = key
            seg = self.current_times.segment_time(split, level, None)
            best = self.compare_best[key]
            if seg is not None and (best is None or seg < best):
                self.compare_best[key] = seg

    # this resets the timer when the player selects a new file or deletes the current file to start over
    def reset(self):
        self.current_piece_idx = 0
        self.current_times = SplitsRecord()
        self.started = False
        self.start_time = 0

    # this is used when the player manually skips the current split, and the run moves forward, adding 1 to the current split index
    def skip(self, n=1):
        while not self.done:
            if type(self.current_piece) is Split:
                self.current_times[self.current_piece] = None
                self.current_piece_idx += 1
            elif type(self.current_piece) is StartTimer:
                self.start_time = self.asi[self.route.time_field]
                self.current_piece_idx += 1
            else:
                if n:
                    self.started = True
                    self.current_piece_idx += 1
                    n -= 1
                else:
                    break

    # this is used when the player manually rewinds the run to redo the split, and this subtracts 1 from the current split index
    def rewind(self, n=1):
        while self.current_piece_idx:
            if type(self.current_piece) is Split:
                del self.current_times[self.current_piece]
                self.current_piece_idx -= 1
            elif type(self.current_piece) is StartTimer:
                self.current_piece_idx -= 1
                self.started = False
            else:
                if n:
                    self.current_piece_idx -= 1
                    n -= 1
                else:
                    if self.current_piece.check_trigger(self.asi):
                        self.current_piece_idx -= 1
                    else:
                        break

    
    """
    from rhelmot xemself:
    "update is the heart of the heart, the core of the SplitsManager. It is called once per frame and is in charge of noticing when you've completed a split. Additionally, it's in charge of noticing when you've reset a run, so that's what the commit() reset() stuff is about."
    """
    def update(self):
        if type(self.route.reset_trigger) is Trigger and self.route.reset_trigger.check_trigger(self.asi):
            self.commit()
            self.reset()

        if self.done:
            return

        while not self.done:
            if type(self.current_piece) is Split:
                self.split(self.current_piece)
                self.current_piece_idx += 1
            elif type(self.current_piece) is StartTimer:
                self.start_time = self.asi[self.route.time_field]
                self.current_piece_idx += 1
            else:
                if self.current_piece.check_trigger(self.asi):
                    self.started = True
                    self.current_piece_idx += 1
                else:
                    break


# oh, this parses the name of the chapter
def parse_mapname(line):
    if line.lower() == 'farewell':
        return 10, 0
    if line.lower() == 'prologue':
        return 0, 0

    if line.isdigit():
        side = 'a'
    else:
        line, side = line[:-1], line[-1]
        side = side.lower()
    assert side in ('a', 'b', 'c')
    mode = ord(side) - ord('a')
    chapter = int(line)
    if chapter >= 8:
        chapter += 1
    return chapter, mode


# and this is the demo we see when we run celeste_timer.py
def _main():
    asi = AutoSplitterInfo()
    max_width = max(len(attr) for attr in asi.all_attrs)
    while True:
        data = '\x1b\x5b\x48\x1b\x5b\x4a' # sorry what
        time.sleep(0.01)
        for attr in asi.all_attrs:
            val = asi.dict[attr]
            if attr.endswith('_time'):
                val = fmt_time(val)
            data += attr.ljust(max_width) + ': ' + str(val) + '\n'
        print(data)


if __name__ == '__main__':
    _main()
