"""Common code for reading NTU-RGBD videos (hint: it's a pain)."""

import re
import subprocess
import shutil
import tempfile
import zipfile
import os

from scipy.misc import imread


class ZipVideo:
    FRAME_TEMPLATE = '%04d.jpg'

    def __init__(self, zip_path, path_in_zip):
        # read the video
        zf = zipfile.ZipFile(zip_path)
        inner_fp = zf.open(path_in_zip)

        # make temporary file and extract video to it
        ext = os.path.splitext(path_in_zip)[1]
        with tempfile.NamedTemporaryFile(suffix=ext) as intermediate_fp:
            while True:
                # copy across 1MiB at a time
                buf = inner_fp.read(1024 * 1024)
                if not buf:
                    break
                intermediate_fp.write(buf)
            intermediate_fp.flush()

            # just copy the frames out to disk, I give up trying to do anything
            # more fancy
            self.tmp_dir = intermediate_fp.name + '-frames'
            os.makedirs(self.tmp_dir, exist_ok=True)
            subprocess.run(
                [
                    'ffmpeg', '-i', intermediate_fp.name,
                    os.path.join(self.tmp_dir, self.FRAME_TEMPLATE)
                ],
                check=True)

    def get_frame(self, fr_ind):
        # ffmpeg starts at 1, we start at 0
        fr_num = fr_ind + 1
        frame_path = os.path.join(self.tmp_dir, self.FRAME_TEMPLATE % fr_num)
        return imread(frame_path)

    def __del__(self):
        if self.tmp_dir is not None:
            shutil.rmtree(self.tmp_dir)


_vid_name_re = re.compile(r'^S(?P<setup>\d{3})C(?P<camera>\d{3})'
                          r'P(?P<performer>\d{3})R(?P<replication>\d{3})'
                          r'A(?P<action>\d{3})_I(?P<orig_id>\d+)'
                          r'SF(?P<start_frame>\d+)EF(?P<end_frame>\d+)$')


def parse_name(name):
    # takes a sequence name and returns video path, start frame, and end frame
    groups = _vid_name_re.match(name).groupdict()
    # this holds videos for the current setup
    archive_name = 'nturgbd_rgb_s%s.zip' % groups['setup']
    # name of video in archive
    name_pfx, _ = name.split('_', 1)
    video_name = 'nturgb+d_rgb/%s_rgb.avi' % name_pfx
    start_frame = int(groups['start_frame'])
    end_frame = int(groups['end_frame'])
    return archive_name, video_name, start_frame, end_frame
