import json
import math
from collections import OrderedDict


RESOURCE_PATH = "resources/video_1_detections.json"

class_color = {
    "car":(255,0,0), #blue
    "person":(0,255,255), #yellow
    "truck":(0,128,255), #
    "bicycle":(0,255,0), #green
    "bus":(255,0,255), #violet
    "motorbike":(0,0,255) # red
}

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)

    return data

class_forget_time_frames = {
    "car"      : 180,
    "person"   : 480, 
    "truck"    : 340,
    "bicycle"  : 60, 
    "bus"      : 340,
    "motorbike": 60 
}

class_distance = {
    "car"      : 150,
    "person"   : 40, 
    "truck"    : 500,
    "bicycle"  : 60, 
    "bus"      : 500,
    "motorbike": 60  
}



class bounding_box():
    def __init__(self, frame, det_class, x, y, width, height, score, color):

        self.frame = frame
        self.det_class = det_class
        self.x = x
        self.width = width
        self.y = y
        self.height = height
        self.score = score
        self.color = color

        self.xCentre = (self.x + self.x + self.width) //2
        self.yCentre = (self.y + self.y + self.height) //2

    def centre_point(self):
        return (self.xCentre, self.yCentre)

    def add_id(self, id):
        self.id = id

    def get_id(self):
        return self.id


    def print_example(self):
        print("frame",self.frame, self.x,self.y, "class", self.det_class)

    
def load_bounding_boxes(resource_path):
    
    Video = OrderedDict()

    data = load_json(resource_path)

    for frame, value in data.items():
        frame = int(frame)

        for i in range(0,len(value["bounding boxes"])):

            a = bounding_box(int(frame),
                        det_class=value["detected classes"][i],
                        x = value["bounding boxes"][i][0],
                        y = value["bounding boxes"][i][1],
                        width = value["bounding boxes"][i][2],
                        height = value["bounding boxes"][i][3],
                        score = value["detection scores"][i],
                        color= class_color[value["detected classes"][i]])
            # a.print_example()

        if Video.get(frame) is None:
            Video[frame] = [a]
        else:
            Video[frame].append(a) 
    
    return Video



# object tracking

def euclidean_distance(x1, x2, y1, y2):

    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)

    return distance

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.centre_point_update = {}
        self.det_class = {}


def assign_id(Video):

    tracker = EuclideanDistTracker()

    tagged_video = OrderedDict()
    # max_frame = Video.keys()

    for frame, objects in Video.items():
        

        if frame == 1:
            for obj in objects:

                obj.add_id(tracker.id_count)
                tracker.center_points[tracker.id_count] = obj.centre_point()
                
                tracker.centre_point_update[tracker.id_count] = frame

                tracker.det_class[tracker.id_count] = obj.det_class

                tracker.id_count += 1

                

                if tagged_video.get(frame) is None:
                    tagged_video[frame] = [obj]
                else:
                    tagged_video[frame].append(obj) 

        else:
            for obj in objects:


                new_centre_points = tracker.center_points.copy()
                new_centre_points_update = tracker.centre_point_update.copy()
                new_det_class = tracker.det_class.copy()
                same_object_detected = False
                for id, cp in tracker.center_points.items():
                    
                    # distance from object to each saved point
                    distance = euclidean_distance(obj.xCentre, cp[0], obj.yCentre, cp[1])


                    # same object?
                    if distance < class_distance[obj.det_class] and (frame - tracker.centre_point_update[id]) <= class_forget_time_frames[obj.det_class] and (obj.det_class == tracker.det_class[id]):
                        new_centre_points[id] = (obj.xCentre, obj.yCentre)
                        new_centre_points_update[id] = frame
                        obj.add_id(id)

                        if tagged_video.get(frame) is None:
                            tagged_video[frame] = [obj]
                        else:
                            tagged_video[frame].append(obj)
                        
                        same_object_detected = True

                        
                if same_object_detected is False:
                    obj.add_id(tracker.id_count)
                    new_centre_points[tracker.id_count] = obj.centre_point()
                    new_centre_points_update[tracker.id_count] = frame
                    new_det_class[tracker.id_count] = obj.det_class
                    tracker.id_count += 1

                    if tagged_video.get(frame) is None:
                        tagged_video[frame] = [obj]
                    else:
                        tagged_video[frame].append(obj) 
                
                # tracker.center_points.clear()
                # tracker.centre_point_update.clear()
                # tracker.det_class.clear()
                tracker.center_points = new_centre_points.copy()
                tracker.centre_point_update = new_centre_points_update.copy()
                tracker.det_class = new_det_class.copy()

                new_centre_points = new_centre_points.clear()
                new_centre_points_update = new_centre_points_update.clear()
                new_det_class = new_det_class.clear()


        


    return tagged_video





if __name__ == "__main__":
    
    a = load_json(RESOURCE_PATH)
    x = load_bounding_boxes(RESOURCE_PATH)
    tagged_x = assign_id(x)
    print(x)
    print(tagged_x)


    # print(a["1"]['bounding boxes'][0][1])
