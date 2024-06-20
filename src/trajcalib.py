import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np
import copy

import cv2

import math
def rotvec2rotationMat(rotvec, format='degree'):
    R = cv2.Rodrigues(rotvec)[0]
    return R

data1 = np.loadtxt("./pcds_poses_tum.txt")
print(data1.shape)
data2 = np.loadtxt("./F_POS_lidar_sync_veh_pose.txt") # xyzrpy
print(data2.shape)

ax = plt.figure().add_subplot(projection='3d')


cam2veh = np.array([[ 0.9990,        0, 0.0439,     0],
                    [-0.0157,  0.9333, 0.3586, -0.322],
                    [ 0.0410, -0.3590, 0.9324,  3.642],
                    [      0,       0,      0,      1]])

init_lidar2cam = np.asarray([[-0.00478373,  1.00005229,  0.00839116,  0.05791383],
                            [-0.94033487, -0.00692047, -0.33991561, -0.03897246],
                            [-0.34008549, -0.01175839,  0.94032928, -3.41434962],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])

init_extrinsic = cam2veh @ init_lidar2cam

init_rotvec = cv2.Rodrigues(init_extrinsic[0:3, 0:3])[0]

print(init_rotvec)


ax_slide0 = plt.axes([.05, .2, .85, .02]) # [left, bottom, width, height]
rx_bar = Slider(ax_slide0,"rx changing value",valmin=-1.0*np.pi,valmax=1.0*np.pi, valinit=init_rotvec[0, 0],valstep=.001)
ax_slide1 = plt.axes([.05, .175, .85, .02])
ry_bar = Slider(ax_slide1,"ry changing value",valmin=-1.0*np.pi,valmax=1.0*np.pi, valinit=init_rotvec[1, 0],valstep=.001)
ax_slide2 = plt.axes([.05, .15, .85, .02])
rz_bar = Slider(ax_slide2,"rz changing value",valmin=-1.0*np.pi,valmax=1.0*np.pi, valinit=init_rotvec[2, 0],valstep=.001)

ax_slide3 = plt.axes([.05, .125, .85, .02]) # [left, bottom, width, height]
x_bar = Slider(ax_slide3,"x changing value",valmin=-5.0,valmax=5.0,valinit=0,valstep=.05)
ax_slide4 = plt.axes([.05, .10, .85, .02])
y_bar = Slider(ax_slide4,"y changing value",valmin=-5.0,valmax=5.0,valinit=0,valstep=.05)
ax_slide5 = plt.axes([.05, .075, .85, .02])
z_bar = Slider(ax_slide5,"z changing value",valmin=-5.0,valmax=5.0,valinit=0,valstep=.05)

def generate_data(rx=0.0, ry=0.0, rz=0.0, x=0.0, y=0.0, z=0.0):
    rot_vec = np.array([rx, ry, rz])
    t_vec = np.array([x, y, z])

    extrinsic = np.eye(4)
    extrinsic[0:3, 0:3] = rotvec2rotationMat(rot_vec)
    extrinsic[0:3, 3] = t_vec.T

    lidar2veh = extrinsic
    lidar2cam = np.linalg.inv(cam2veh) @ lidar2veh
    print("lidar2cam : ", lidar2cam)



    xyz = copy.deepcopy(data1[:, 1:4].copy())
    xyzw = np.concatenate([xyz, np.ones_like(xyz[:, 0:1])], axis=1)

    xyzw_new = (extrinsic@xyzw.T).T
    xyz_new = xyzw_new[:, 0:3]
    xyz = xyz_new

    data1x = xyz[:, 0]
    data1y = xyz[:, 1]
    data1z = xyz[:, 2]

    data1i = data1[:, 4]
    data1j = data1[:, 5]
    data1k = data1[:, 6]
    data1w = data1[:, 7]

    data2x = data2[:, 0]
    data2y = data2[:, 1]
    data2z = data2[:, 2]

    return data1x, data1y, data1z, data2x, data2y, data2z


ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")



def update(val):
    ax.clear()

    ax.set_xlim(-50, 300)
    ax.set_ylim(-50, 300)
    # ax.set_zlim(-5, 5)

    # ax.set_xticks(np.arange(-500, 500, 100))
    # ax.set_yticks(np.arange(-500, 500, 100))
    # ax.set_zticks(np.arange(-500, 500, 100))

    print(rx_bar.val, ry_bar.val, rz_bar.val, x_bar.val, y_bar.val, z_bar.val)

    data1x, data1y, data1z, data2x, data2y, data2z = generate_data(rx_bar.val, ry_bar.val, rz_bar.val, x_bar.val, y_bar.val, z_bar.val)
    ax.plot(data1x, data1y, data1z, "blue")
    ax.plot(data2x, data2y, data2z, "orange")


rx_bar.on_changed(update)
ry_bar.on_changed(update)
rz_bar.on_changed(update)
x_bar.on_changed(update)
y_bar.on_changed(update)
z_bar.on_changed(update)

plt.show()
