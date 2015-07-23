
import sys
import json
import subprocess

# bounding box - represents a bounding box in an image
class BoundingBox:
    from_x = 0
    from_y = 0
    to_x = 0
    to_y = 0

    def __init__(self, from_x = (-sys.maxint - 1), to_x = sys.maxint, from_y = (-sys.maxint - 1), to_y = sys.maxint):
        self.from_x = float(from_x)
        self.to_x = float(to_x)
        self.from_y = float(from_y)
        self.to_y = float(to_y)
        if not self.validate():
            raise "Invalid bounding box values: {0}, {1}, {2}, {3} (should be {0} < {1}, and {2} < {3}".format(
                self.from_x, self.from_y, self.to_x, self.to_y) 

    @classmethod
    def fromList(cls, bbox_list):
        return cls(bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3])


    @classmethod
    def fromStr(cls, bbox_str):
        return cls.fromList(bbox_str.split(" "))

    def validate(self):
        # TODO: check that the bounding box values are valid
        if (self.from_x > self.to_x) or (self.from_y > self.to_y):
            return False
        return True

    def overlap(self, other_bbox):
        # Returns true if there is intersection between the bboxes or a full containment
        if (self.from_x < other_bbox.to_x) and (self.to_x > other_bbox.from_x) and \
           (self.from_y < other_bbox.to_y) and (self.to_y > other_bbox.from_y):
            return True
        return False

    def extend(self, other_bbox):
        # updates the current bounding box by extending it to include the other_bbox
        if self.from_x > other_bbox.from_x:
            self.from_x = other_bbox.from_x
        if self.from_y > other_bbox.from_y:
            self.from_y = other_bbox.from_y
        if self.to_x < other_bbox.to_x:
            self.to_x = other_bbox.to_x
        if self.to_y < other_bbox.to_y:
            self.to_y = other_bbox.to_y

    def toStr(self):
        return '{0} {1} {2} {3}'.format(self.from_x, self.to_x, self.from_y, self.to_y)

    def toArray(self):
        return [self.from_x, self.to_x, self.from_y, self.to_y]

    @classmethod
    def parse_bbox_lines(cls, bbox_lines):
        str = ''.join(bbox_lines)
        str = str[str.find('[') + 1:str.find(']')]
        str = str.replace(',', ' ')
        return str

    @classmethod
    def read_bbox(cls, tiles_spec_fname):
        cmd = "grep -A 5 \"bbox\" {}".format(tiles_spec_fname)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Parse all bounding boxes in the given json file
        ret_val = None
        cur_bbox_lines = []
        for line in iter(p.stdout.readline, ''):
            if line.startswith("--"):
                cur_bbox = BoundingBox.fromStr(BoundingBox.parse_bbox_lines(cur_bbox_lines))
                if ret_val is None:
                    ret_val = cur_bbox
                else:
                    ret_val.extend(cur_bbox)
                cur_bbox_lines = []
            else:
                cur_bbox_lines.append(line.strip(' \n'))
        if len(cur_bbox_lines) > 0:
            cur_bbox = BoundingBox.fromStr(BoundingBox.parse_bbox_lines(cur_bbox_lines))
            if ret_val is None:
                ret_val = cur_bbox
            else:
                ret_val.extend(cur_bbox)
        return ret_val


    def __getitem__(self, i):
        return [self.from_x, self.to_x, self.from_y, self.to_y][i]
    
