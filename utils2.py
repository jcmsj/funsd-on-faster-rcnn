# Pascal VOC: x1,y1,x2,y2
# YOLO: x_center, y_center, width, height
# COCO: x1, y1, width, height

def x1y1wh_to_cxcywh(bbox: tuple[float, float, float, float], img_width: int, img_height: int):
    x1, y1, w, h = bbox
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx/img_width, cy/img_height, w/img_width, h/img_height

def x1y1x2y2_to_x1y1wh(bbox: tuple[float, float, float, float]):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h

def cxcywh_to_x1y1x2y2(
    yolo: tuple[float, float, float, float], img_width: int, img_height: int
):
    """
    scales the yolo coordinates to the image size
    """
    cx, cy, w, h = yolo
    x1 = (cx - (w / 2)) * img_width
    y1 = (cy - (h / 2)) * img_height
    x2 = (cx + (w / 2)) * img_width
    y2 = (cy + (h / 2)) * img_height
    return x1, y1, x2, y2
# Normalize bounding box coordinates
def limit_to_0_1(val:float) -> float:
    return 1 if val > 1 else 0 if val < 0 else val
def limit_bbox(x):
    return tuple(map(limit_to_0_1, x))
def test_x1y1wh_to_cxcywh():
    """
    >>> '%.4f, %.4f, %.4f, %.4f' % x1y1wh_to_cxcywh((50.58, 316.75, 240.12,24.81),601, 792)
    '0.2839, 0.4156, 0.3995, 0.0313'
    """

def test_cxcywh_to_x1y1x2y2():
    """
    >>> '%.2f, %.2f, %.2f, %.2f' % cxcywh_to_x1y1x2y2((0.2839, 0.4156, 0.3995, 0.0313),601, 792)
    '50.57, 316.76, 290.67, 341.55'
    """

if __name__ == "__main__":
    import doctest

    doctest.testmod()
