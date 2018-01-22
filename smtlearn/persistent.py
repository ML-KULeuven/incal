import json
import os


class Data(object):
    def __init__(self, filename, initial_structure):
        self.filename = filename
        self.initial_structure = initial_structure

    def load(self):
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                json.dump({"files": {}}, f)

        with open(self.filename, "r") as f:
            flat = json.load(f)
        return flat

    def dump(self, flat):
        temp_name = "{}.tmp".format(self.filename)
        with open(temp_name, "w") as f:
            json.dump(flat, f)
        os.rename(temp_name, self.filename)
