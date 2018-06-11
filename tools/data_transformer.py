import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random
import copy

# Get the all files in the specified directory (path).
def get_recursive_file_list(path, file_lst, ext):
    current_files = os.listdir(path)
    for file_name in current_files:
        full_file_name = path + "/" + file_name
        if os.path.isdir(full_file_name):
            get_recursive_file_list(full_file_name, file_lst, ext)
        elif full_file_name[-len(ext):]==ext:
            file_lst.append(full_file_name)
        else:
            None

class img_transformer:
    def __init__(self, rotation_range=[0, 0, -10, 10], aspect_range=[0.95, 1.05], shearx_range=[-0.01, 0.01], sheary_range=[-0.01, 0.01],
                 gamma_range=[0.9, 1.1], saturate_range=[0.8, 1.2], noise_range=[0, 0.05], intensity_range=[0.8, 1.2], perspective_configure=[0.3,0.2],
                 downsampling_range=[0.5,2.5], perspective=False, flip=False, downsampling=False):
        self.rotation_range = rotation_range
        self.aspect_range = aspect_range
        self.shearx_range = shearx_range
        self.sheary_range = sheary_range
        self.gamma_range = gamma_range
        self.saturate_range = saturate_range
        self.intensity_range = intensity_range
        self.noise_range = noise_range
        self.flip=flip
        self.perspective = perspective
        self.perspective_configure = perspective_configure
        self.downsampling = downsampling
        self.downsampling_range = downsampling_range
        random.seed(0)

    def perspective_img(self, img):
        img_size = np.array([img.shape[1],img.shape[0]]).astype(np.float32)
        img_center = img_size/2
        offset = img_size*self.perspective_configure[0]

        mode = random.choice([0,1,2,3])
        perturb = np.round(max(offset)*self.perspective_configure[1] * random.uniform(0, 1))
        if mode == 0:
            pt1 = [img_center[0] + offset[0], img_center[1] - offset[1]]
            pt2 = [img_center[0] + offset[0], img_center[1] + offset[1]]
            pt3 = [img_center[0], img_center[1] - offset[1]]
            pt4 = [img_center[0], img_center[1] + offset[1]]
            pt1_new = [pt1[0], pt1[1]+ perturb]
            pt2_new = [pt2[0], pt2[1]- perturb]
            pt3_new = pt3
            pt4_new = pt4
        elif mode == 1:
            pt1 = [img_center[0] - offset[0], img_center[1] - offset[1]]
            pt2 = [img_center[0] - offset[0], img_center[1] + offset[1]]
            pt3 = [img_center[0], img_center[1] - offset[1]]
            pt4 = [img_center[0], img_center[1] + offset[1]]
            pt1_new = [pt1[0], pt1[1]+ perturb]
            pt2_new = [pt2[0], pt2[1]- perturb]
            pt3_new = pt3
            pt4_new = pt4
        elif mode == 2:
            pt1 = [img_center[0] - offset[0], img_center[1] - offset[1]]
            pt2 = [img_center[0] + offset[0], img_center[1] - offset[1]]
            pt3 = [img_center[0] - offset[0], img_center[1]]
            pt4 = [img_center[0] + offset[0], img_center[1]]
            pt1_new = [pt1[0]+ perturb, pt1[1]]
            pt2_new = [pt2[0]- perturb, pt2[1]]
            pt3_new = pt3
            pt4_new = pt4
        else:
            pt1 = [img_center[0] - offset[0], img_center[1] + offset[1]]
            pt2 = [img_center[0] + offset[0], img_center[1] + offset[1]]
            pt3 = [img_center[0] - offset[0], img_center[1]]
            pt4 = [img_center[0] + offset[0], img_center[1]]
            pt1_new = [pt1[0]+ perturb, pt1[1]]
            pt2_new = [pt2[0]- perturb, pt2[1]]
            pt3_new = pt3
            pt4_new = pt4
        pt_ori = np.float32([pt1,pt2,pt3,pt4])
        pt_new = np.float32([pt1_new, pt2_new, pt3_new, pt4_new])
        perspective_mat= cv2.getPerspectiveTransform(pt_ori,pt_new)
        img_perspective = cv2.warpPerspective(img, perspective_mat, (img.shape[1], img.shape[0]))
        return img_perspective

    def add_downsampling(self, img, noise):
        down_method = random.choice([0, 1, 2, 3, 4])
        up_method = random.choice([0, 1, 2, 3, 4])
        downsample_rate = self.downsampling_range[0] + (self.downsampling_range[1]-self.downsampling_range[0]) * random.uniform(0,1)
        if downsample_rate<=1:
            img_downsampling = copy.deepcopy(img)
        else:
            img_downsampling = cv2.resize(img, dsize=(np.int32(img.shape[1]/downsample_rate), np.int32(img.shape[0]/downsample_rate)), interpolation=down_method)
        img_downsampling = self.add_noise(img_downsampling, noise)
        img_upsampling = cv2.resize(img_downsampling, dsize=(img.shape[1], img.shape[0]), interpolation=up_method)
        return img_upsampling

    def add_noise(self, img, noise):
        noise_level = np.max(img)*noise
        noise_mat = np.random.normal(0, noise_level, size=(img.shape[0], img.shape[1], img.shape[2]))
        img_noise = np.round(np.clip(img + noise_mat, 0, 255)).astype(np.uint8)
        return img_noise

    def adjust_gamma(self, img, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        img_gamma = cv2.LUT(img, table)
        return img_gamma

    def adjust_HSV(self, img, saturate, intensity):
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        HSV_adjust=copy.deepcopy(HSV)
        HSV_adjust[:,:,1] = HSV[:,:,1] * saturate
        HSV_adjust[:, :, 2] = HSV[:, :, 2] * intensity
        HSV_adjust = np.clip(HSV_adjust, 0, 255).astype(np.uint8)
        img_adjust = cv2.cvtColor(HSV_adjust, cv2.COLOR_HSV2BGR)
        return img_adjust

    def get_shear_mat(self, shearx, sheary, aspect, centerx, centery):
        pt1 = [centerx, centery]
        pt2 = [centerx - 100, centery - 100]
        pt3 = [centerx - 100, centery + 100]
        pt1_new = [centerx, centery]
        pt2_new = [centerx - (100 + 100 * sheary)*aspect, centery - 100 - 100 * shearx]
        pt3_new = [centerx - (100 - 100 * sheary)*aspect , centery + 100 - 100 * shearx]
        affine_mat = np.eye(3,dtype=np.float32)
        affine_mat[0:2,:] = cv2.getAffineTransform(np.float32([pt1, pt2, pt3]), np.float32([pt1_new, pt2_new, pt3_new]))
        #print(aspect, shearx, sheary)
        return affine_mat

    def transform_img(self, img, rotation=None, aspect=None, shearx=None, sheary=None, gamma=None, saturate=None, intensity=None, noise=None):
        if rotation==None:
            rotation=[self.rotation_range[0], self.rotation_range[1], self.rotation_range[2]+(self.rotation_range[3]-self.rotation_range[2])*random.uniform(0,1)]
        if aspect == None:
            aspect= self.aspect_range[0] + (self.aspect_range[1]-self.aspect_range[0])*random.uniform(0,1)
        if shearx == None:
            shearx = self.shearx_range[0] + (self.shearx_range[1]-self.shearx_range[0])*random.uniform(0,1)
        if sheary == None:
            sheary = self.sheary_range[0] + (self.sheary_range[1]-self.sheary_range[0])*random.uniform(0,1)
        if gamma == None:
            gamma = self.gamma_range[0] + (self.gamma_range[1]-self.gamma_range[0])*random.uniform(0,1)
        if saturate == None:
            saturate = self.saturate_range[0] + (self.saturate_range[1]-self.saturate_range[0])*random.uniform(0,1)
        if intensity == None:
            intensity = self.intensity_range[0] + (self.intensity_range[1]-self.intensity_range[0])*random.uniform(0,1)
        if noise == None:
            noise = self.noise_range[0] + (self.noise_range[1] - self.noise_range[0]) * random.uniform(0,1)

        rotation_mat_2D = cv2.getRotationMatrix2D((rotation[0], rotation[1]),rotation[2],1)
        rotation_mat = np.zeros(shape=(3, 3), dtype=np.float32)
        rotation_mat[0:2,:] = rotation_mat_2D

        shear_aspect_mat = self.get_shear_mat(shearx, sheary, aspect, rotation[0], rotation[1])

        affine_mat = np.dot(rotation_mat, shear_aspect_mat)
        affine_mat = affine_mat[0:2, :]
        transformed_img = cv2.warpAffine(img, affine_mat, (img.shape[1], img.shape[0]))

        img_HSV = self.adjust_HSV(transformed_img, saturate, intensity)
        img_gamma = self.adjust_gamma(img_HSV, gamma)

        if self.downsampling==True:
            img_final = self.add_downsampling(img_gamma, noise)
        else:
            img_final = self.add_noise(img_gamma, noise)
        if self.perspective == True:
            img_final = self.perspective_img(img_final)
        if self.flip == True:
            if random.uniform(0,1)>0:
                img_final = cv2.flip(img_final, 1)
        #cv2.imshow("transformed_img_full", img_final)
        #cv2.imshow("transformed_img", img_final[100:200,100:200,:])
        #cv2.waitKey(-1)
        return img_final

if __name__ == "__main__":
    transformer = img_transformer(rotation_range=[151, 151, -10, 10], aspect_range=[0.8, 1.2], shearx_range=[-0.2, 0.2], sheary_range=[-0.2, 0.2], intensity_range=[0.6, 1.5], saturate_range=[0.3, 1.3], noise_range=[0, 0.1], gamma_range=[0.6,1.7], perspective_configure=[0.2, 0.1], downsampling_range=[0.6, 2.5], perspective=True, flip=False, downsampling=True)
    img = cv2.imread("H:\projects\data/traffic_sign_from_Tsinghua_tencent_100K\patch/test.jpg")
    for i in range(1000):
        transformer.transform_img(img)