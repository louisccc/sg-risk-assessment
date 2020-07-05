import math

xval = 410
yval = -84
xval2 = 384
yval2 = -121
ego_yaw = math.radians(-104)
cos_term = math.cos(ego_yaw)
sin_term = math.sin(ego_yaw)

def rotate_coords(x, y): 
    new_x = (x*cos_term) + (y*sin_term)
    new_y = ((-x)*sin_term) + (y*cos_term)
    return new_x, new_y

x1, y1 = rotate_coords(xval, yval)
x2, y2 = rotate_coords(xval2, yval2)
print(rotate_coords(xval, yval))
print(rotate_coords(xval2, yval2))
print(x2-x1)
print(y2-y1)