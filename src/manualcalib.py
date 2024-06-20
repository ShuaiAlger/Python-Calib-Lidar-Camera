import rospy

import math
from math import *
import numpy as np

import cv2

import copy

import time

np.set_printoptions(precision=4, suppress=True)

count = 0


import open3d as o3d





def eulerAngles2rotationMat(theta, format='degree'):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R



rot_vec = np.array([0.0, 0.0, 0.0])
t_vec = np.array([0.0, 0.0, 0.0])

cam_bev_rot_vec = np.array([pi/2, 0.0, 0.0])
cam_bev_rot_mat = cv2.Rodrigues(cam_bev_rot_vec)[0]
cam_bev_extrinsic = np.eye(4)
cam_bev_extrinsic[:3, :3] = cam_bev_rot_mat


def rot_x(x):
    global rot_vec
    rot_vec[0] = (x/10)*pi/180

def rot_y(x):
    global rot_vec
    rot_vec[1] = (x/10)*pi/180

def rot_z(x):
    global rot_vec
    rot_vec[2] = (x/10)*pi/180

def t_x(x):
    global rot_vec
    t_vec[0] = (x - 500)/100

def t_y(x):
    global rot_vec
    t_vec[1] = (x - 500)/100

def t_z(x):
    global rot_vec
    t_vec[2] = (x - 500)/100



cv2.namedWindow("show_img", 0)
cv2.createTrackbar('rx', 'show_img', 3586, 3600, rot_x)
cv2.createTrackbar('ry', 'show_img', 213, 3600, rot_y)
cv2.createTrackbar('rz', 'show_img', 2700, 3600, rot_z)
cv2.createTrackbar('tx', 'show_img', 500, 1000, t_x)
cv2.createTrackbar('ty', 'show_img', 496, 1000, t_y)
cv2.createTrackbar('tz', 'show_img', 162, 1000, t_z)




intrisic = np.array([   [289.21   ,        0.,   640.822],
                        [       0.,   289.665,   480.529],
                        [       0.,        0.,        1.]
                    ])

distortion = np.array([[0.134212, 0.018733, -0.0252689, 0.00412384]])



fx = intrisic[0, 0]
fy = intrisic[1, 1]
cx = intrisic[0, 2]
cy = intrisic[1, 2]


from pypcd4 import PointCloud
import tqdm
def main():
    dataset_root = "/media/shuai/Correspondence/Calib/data/dominant3062/"

    gt_file = dataset_root + "F_POS.txt"

    frame_num = 0

    poses = []

    with open(gt_file) as f:
        lines = f.readlines()
        for i in tqdm.trange(0, len(lines), 1):
            line = lines[i].split("\n")[0]
            x, y, z, roll, pitch, yaw = line.split(" ")
            pose = [float(x), float(y), float(z),
                    float(roll), float(pitch), float(yaw)]
            frame_num = frame_num + 1
            poses.append(pose)

    speeds = []
    for i in tqdm.trange(0, len(poses)-1, 1):
        vx = poses[i+1][0] - poses[i][0]
        vy = poses[i+1][1] - poses[i][1]
        vz = poses[i+1][2] - poses[i][2]
        speeds.append([vx, vy, vz])
    
    speeds = np.array(speeds)

    index_list = []

    with open(dataset_root+"Dataset.txt") as f:
        lines = f.readlines()
        for i in tqdm.trange(1, len(lines), 1):
            line = lines[i].split("\n")[0]
            lidar_id, _, cam_id = line.split(",")
            lidar_id = int(lidar_id)
            cam_id = int(cam_id)
            index_list.append([lidar_id, cam_id])
            print(lidar_id, cam_id)

    for j in range(800, len(index_list)):
        img_filename = "/media/shuai/Correspondence/Calib/data/dominant3062/images/output_" + str(index_list[j][1]+1).zfill(4) + ".png"
        pcd_filename = "/media/shuai/Correspondence/Calib/data/dominant3062/pcds/" + str(index_list[j][0]) + ".pcd"

        cv_img_src = cv2.imread(img_filename)
        pc: PointCloud = PointCloud.from_path(pcd_filename)

        pcd = o3d.io.read_point_cloud(pcd_filename)

        distance_threshold = 0.2    # The maximum distance from the interior point to the planar model
        ransac_n = 3                # Number of sampling points used to fit the plane
        num_iterations = 1000       # Maximum number of iterations

        # Return the model coefficients plane_model and inliers, and assign values to them
        plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)

        # Output plane equation
        [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        # inliers = np.asarray(inliers).reshape(len(inliers), 1)

        inlier_cloud = pcd.select_by_index(inliers)

        inlier_cloud = np.asarray(inlier_cloud.points)

        # inlier_cloud.paint_uniform_color([0, 0, 1.0])

        # outlier_cloud = pcd.select_by_index(inliers, invert=True)
        # outlier_cloud.paint_uniform_color([1.0, 0, 0])



        print(j, speeds[j])

        laser = np.array(pc.numpy()[:, 0:4])

        laser_points = laser
        # laser_points = laser[inliers]

        # print(laser_points.shape)

        distances = np.sqrt(laser_points[:, 0]*laser_points[:, 0]+
                            laser_points[:, 1]*laser_points[:, 1]+
                            laser_points[:, 2]*laser_points[:, 2]
                                ).reshape(laser_points.shape[0], 1)

        max_distance_thres = 50
        min_distance_thres = 1

        mask = np.asarray(distances < max_distance_thres)
        filtered_laser_points = copy.deepcopy(laser_points[mask[:, 0], :].copy())
        all_num = filtered_laser_points.shape[0]

        distances = np.sqrt(filtered_laser_points[:, 0]*filtered_laser_points[:, 0]+
                            filtered_laser_points[:, 1]*filtered_laser_points[:, 1]+
                            filtered_laser_points[:, 2]*filtered_laser_points[:, 2]
                                ).reshape(filtered_laser_points.shape[0], 1)
        
        distance_indices = np.argsort((-distances).reshape(filtered_laser_points.shape[0]), axis=0)

        distances[distances < min_distance_thres] = min_distance_thres
        distances = distances - min_distance_thres
        distances = np.sqrt(distances) 
        thres_color = np.sqrt(max_distance_thres)
        distances = distances/thres_color*255.0


        intensity = laser_points[mask[:, 0], 3]
        max_intensity = intensity.max()
        intensity = intensity/max_intensity*255.0
        intensity = np.uint8(intensity).reshape(len(intensity), 1)
        color_intensity = cv2.applyColorMap(intensity, cv2.COLORMAP_PLASMA)

        # print(intensity.shape, color_intensity.shape)

        RUN = 1
        while(RUN):
        # if (RUN):

            cv_img = copy.deepcopy(cv_img_src.copy())
            canvas_depth = np.zeros_like(cv_img)
            canvas_intensity = np.zeros_like(cv_img)
            
            xyzhomo = np.concatenate([filtered_laser_points[:, 0:3],
                                    np.ones((all_num, 1))], axis=1)

            rot_mat = eulerAngles2rotationMat(rot_vec)

            # rot_mat = cv2.Rodrigues(rot_vec)[0]
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rot_mat
            extrinsic[:3, 3] = t_vec

            # # # print(extrinsic)

            xyzhomo = (extrinsic @ xyzhomo.T).T


            p3ds = xyzhomo[:, 0:3]
            length = p3ds.shape[0]
            x = p3ds[:, 0].reshape((-1, 1))
            y = p3ds[:, 1].reshape((-1, 1))
            z = p3ds[:, 2].reshape((-1, 1))
            chi = np.sqrt(x*x + y*y)
            theta = np.arctan2(chi, z)
            rho = theta + distortion[0, 0] * theta ** 3 + distortion[0, 1] * theta ** 5 \
                        + distortion[0, 2] * theta ** 7 + distortion[0, 3] * theta ** 9
            phi = np.arctan2(y, x)

            points_distort_homo = np.concatenate((rho * np.cos(phi), 
                                                rho * np.sin(phi),
                                                np.ones(shape=(length,1))), axis=1)
            uv = np.dot(points_distort_homo, intrisic.T)

            for i in range(distance_indices.shape[0]):
                index = int(distance_indices[i])
                u = int(uv[index, 0])
                v = int(uv[index, 1])

                cir_size = 3
                if distances[index] > 100:
                    cir_size = 2
                else:
                    if distances[index] < 10:
                        cir_size = 4
                    else:
                        if distances[index] < 20:
                            cir_size = 4
                        else:
                            if distances[index] < 40:
                                cir_size = 3

                ###  USE distance color visualization
                color = (int(distances[index]), int(distances[index]), int(distances[i]))
                cv2.circle(canvas_depth, (u, v), cir_size, color, -1)

                ###  USE intensity color visualization
                color = (int(color_intensity[index, 0, 0]), int(color_intensity[index, 0, 1]), int(color_intensity[index, 0, 2]))
                cv2.circle(canvas_intensity, (u, v), cir_size, color, -1)

            img_depth_weighted = cv2.addWeighted(cv_img, 0.6, canvas_depth, 0.4, 1.0)
            img_intensity_weighted = cv2.addWeighted(cv_img, 0.6, canvas_intensity, 0.4, 1.0)
            

            gray = cv2.cvtColor(cv_img_src, cv2.COLOR_BGR2GRAY)
            binary_img = copy.deepcopy(gray.copy())
            binary_img[binary_img < 180] = 0
            binary_img[0:300, :] = 0
            binary_img[450:, :] = 0
            binary_img_c3 = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

            edge_img = cv2.Canny(gray, 100, 200)
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)

            # edge_intensity_weighted = cv2.addWeighted(edge_img, 0.6, canvas_intensity, 0.4, 1.0)

            gray_intensity = cv2.cvtColor(canvas_intensity, cv2.COLOR_BGR2GRAY)
            binary_inten_img = copy.deepcopy(gray_intensity.copy())
            binary_inten_img[binary_inten_img < 170] = 0
            binary_inten_img[0:300, :] = 0
            binary_inten_img[450:, :] = 0
            binary_inten_img_c3 = cv2.cvtColor(binary_inten_img, cv2.COLOR_GRAY2BGR)


            edge_intensity_img = cv2.Canny(gray_intensity, 180, 200)
            edge_intensity_img = cv2.cvtColor(edge_intensity_img, cv2.COLOR_GRAY2BGR)


            iou = np.uint8(0.5*binary_img + 0.5*binary_inten_img)
            iou = cv2.applyColorMap(iou, cv2.COLORMAP_JET)

            projection = cv2.addWeighted(cv_img_src, 0.5,
                                         canvas_intensity, 0.5, 1.0)



            cv2.putText(cv_img_src, "raw img", (100, 100), 0, 1.0, (0, 255, 125), 3)
            cv2.putText(binary_img_c3, "binary img", (100, 100), 0, 1.0, (0, 255, 125), 3)
            cv2.putText(iou, "binary (img + intensity)", (100, 100), 0, 1.0, (0, 255, 125), 3)



            cv2.putText(canvas_intensity, "road intensity", (100, 100), 0, 1.0, (0, 255, 125), 3)
            cv2.putText(binary_inten_img_c3, "binary road intensity", (100, 100), 0, 1.0, (0, 255, 125), 3)
            cv2.putText(projection, "projection", (100, 100), 0, 1.0, (0, 255, 125), 3)


            show_img = np.concatenate([
                                        np.concatenate([cv_img_src, binary_img_c3, iou], axis=1),
                                        np.concatenate([canvas_intensity, binary_inten_img_c3, projection], axis=1)
                                    ], axis=0)

            cv2.imwrite("img_src.png", cv_img_src[300:450, :, :])
            cv2.imwrite("canvas_intensity.png", canvas_intensity[300:450, :, :])

            if 1:
                cv2.imshow("show_img", show_img)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    exit()
                if key == ord('n'):
                    break
            else:
                cv2.imwrite("edge_map/" + str(j).zfill(4)+".png", show_img)


    
    




if __name__ == "__main__":



    main()



