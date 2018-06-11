import os
import random
import cv2
import sys
import shutil
#reload(sys)
#sys.setdefaultencoding('utf-8')
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

# Get the all sub folders in the "path" folder
def get_sub_folders(path, sub_folder):
    current_files = os.listdir(path)
    for file_name in current_files:
        full_file_name = os.path.join(path, file_name)
        if os.path.isdir(full_file_name):
            sub_folder.append(full_file_name)
        else:
            None

def load_caffe_idx_dict(idx_sequence_lst):
    idx_dict = dict()
    idx_sequence_file = open(idx_sequence_lst, "r")
    lines = idx_sequence_file.readlines()
    for line in lines:
        eles = line.split("\t")
        idx_dict[eles[1].strip()] = int(eles[0])-1   #start from 0
    idx_sequence_file.close()
    return idx_dict

def make_caffe_val_lst(root_path, val_ground_truth_lst, val_lst):
    val_labels = list()
    val_gt = open(val_ground_truth_lst,"r")
    lines = val_gt.readlines()
    val_labels = list()
    for line in lines:
        val_labels.append(int(line)-1) #start from 0
    val_gt.close()
    assert (len(val_labels) == 50000)

    val_file = open(val_lst, "w")
    for i in xrange(len(val_labels)):
        file_path = root_path + ("/ILSVRC2012_val_%08d.JPEG" % (i+1))
        if os.access(file_path, os.R_OK) is False:
            raise Exception(file_path + " does not exist")
        label_line = file_path.replace(root_path+"/", "") + " " + str(val_labels[i]) + "\n"
        val_file.write(label_line)
    val_file.close()

# the image lst is for caffe training
def make_caffe_train_lst_cls(root_folder, train_lst_name, idx_dict_file):
    idx_dict = load_caffe_idx_dict(idx_dict_file)
    class_folders = []
    get_sub_folders(root_folder, class_folders)
    train_lst = open(train_lst_name,'w')

    img_labels = list()
    for i in xrange(len(class_folders)):
        img_lst = []
        class_folder = class_folders[i]
        class_name = class_folder[class_folder.rfind("/")+1:]
        class_idx = idx_dict[class_name]
        get_recursive_file_list(class_folder, img_lst, '.jpg')
        get_recursive_file_list(class_folder, img_lst, '.jpeg')
        get_recursive_file_list(class_folder, img_lst, '.JPEG')
        get_recursive_file_list(class_folder, img_lst, '.bmp')
        for img in img_lst:
            img_labels.append([img, str(class_idx)])
        print (str(i)+'->'+class_folders[i])
    random.shuffle(img_labels)
    for img_label in img_labels:
        train_lst.write(img_label[0].replace(root_folder+"/", "")+" "+img_label[1]+"\n")
    train_lst.close()

def make_mxnet_val_lst(root_path, val_ground_truth_lst, val_lst):
    val_labels = list()
    val_gt = open(val_ground_truth_lst,"r")
    lines = val_gt.readlines()
    val_labels = list()
    for line in lines:
        val_labels.append(int(line)-1) #start from 0
    val_gt.close()
    assert (len(val_labels) == 50000)

    val_file = open(val_lst, "w")
    for i in xrange(len(val_labels)):
        file_path = root_path + ("/ILSVRC2012_val_%08d.JPEG" % (i+1))
        if os.access(file_path, os.R_OK) is False:
            raise Exception(file_path + " does not exist")
        label_line = str(i) + "\t" + ("%0.6f"%float(val_labels[i])) + "\t" + file_path.replace(root_path+"/", "") + "\n"
        val_file.write(label_line)
    val_file.close()

# the image lst is for caffe training
def make_mxnet_train_lst_cls(root_folder, train_lst_name, idx_dict_file):
    idx_dict = load_caffe_idx_dict(idx_dict_file)
    class_folders = []
    get_sub_folders(root_folder, class_folders)
    train_lst = open(train_lst_name,'w')

    img_labels = list()
    for i in xrange(len(class_folders)):
        img_lst = []
        class_folder = class_folders[i]
        class_name = class_folder[class_folder.rfind("/")+1:]
        class_idx = idx_dict[class_name]
        get_recursive_file_list(class_folder, img_lst, '.jpg')
        get_recursive_file_list(class_folder, img_lst, '.jpeg')
        get_recursive_file_list(class_folder, img_lst, '.JPEG')
        for img in img_lst:
            img_labels.append([img, str(class_idx)])
        print (str(i)+'->'+class_folders[i])
    random.shuffle(img_labels)
    for i in xrange(len(img_labels)):
        train_lst.write(str(i) + "\t" + ("%0.6f")%(float(img_labels[i][1])) + "\t" + img_labels[i][0].replace(root_folder+"/", "") + "\n")
    train_lst.close()

def  rename_files_convert_jpg(path):
    file_lst = []
    get_recursive_file_list(path, file_lst, ".bmp")
    get_recursive_file_list(path, file_lst, ".jpg")
    get_recursive_file_list(path, file_lst, ".png")
    idx = 0
    for img_path in file_lst:
        idx1 = img_path.rfind("/")
        idx2 = img_path.rfind(".")
        path = img_path[0:idx1]
        new_name = ("%07d"%idx)
        ext = img_path[idx2:]
        new_path = path + "/" + new_name + ".jpg"
        if ext == ".jpg":
            os.rename(img_path, new_path)
        else:
            img = cv2.imread(img_path)
            cv2.imwrite(new_path,img)
            os.remove(img_path)
        idx += 1

def make_bg_img_label_file(img_root_path, label, label_file):
    img_lst = []
    get_recursive_file_list(img_root_path, img_lst, '.jpg')
    lines = []
    for img_path in img_lst:
        line = img_path + " " + str(label) + "\n"
        lines.append(line)
    with open(label_file, "w") as lf:
        lf.writelines(lines)

def make_caffe_lst_file(img_folders, root_folder, train_ratio, train_lst, val_lst):
    class_idx = 0
    train_lines = []
    val_lines = []
    for img_folder in img_folders:
        img_lst = []
        get_recursive_file_list(img_folder, img_lst, '.bmp')
        get_recursive_file_list(img_folder, img_lst, '.jpg')
        random.shuffle(img_lst)
        img_num = len(img_lst)
        train_num = int(img_num*train_ratio)
        for i in range(0, train_num):
            line = img_lst[i][len(root_folder):] + (" %d\n" % class_idx)
            train_lines.append(line)
        for i in range(train_num, img_num):
            line = img_lst[i][len(root_folder):] + (" %d\n" % class_idx)
            val_lines.append(line)
        class_idx += 1
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    with open(train_lst, "w") as trainf:
        trainf.writelines(train_lines)
    with open(val_lst, "w") as valf:
        valf.writelines(val_lines)

def make_lst(root_path, ext, lst_path):
    lst_file = open(lst_path, "w")
    file_lst = []
    get_recursive_file_list(root_path, file_lst, ext)
    for file_path in file_lst:
        lst_file.write(file_path+"\n")
    lst_file.close()


def copy_file_from_lst(img_lst, input_root, output_root):
    g_idx = 0
    with open(img_lst, "r") as lst:
        lines = lst.readlines()

        for line in lines:
            [path, label] = line.split(" ")
            label = label.strip("\n\r")
            abs_path = input_root + "/" + path
            os.access(abs_path, os.R_OK)
            new_path = output_root + "/" + label
            if os.access(new_path, os.W_OK) == False:
                os.mkdir(new_path)
            new_abs_path = new_path + "/" + ("%08d.jpg"%g_idx)
            g_idx += 1
            shutil.copy(abs_path, new_abs_path)

def copy_original_imgs(det_res_folder, ori_img_folder):
    det_res_lst = []
    ori_img_lst = []
    get_recursive_file_list(det_res_folder, det_res_lst, ".jpg")
    get_recursive_file_list(ori_img_folder, ori_img_lst, ".jpg")
    for det_res_img_path in det_res_lst:
        det_res_img_name = os.path.basename(det_res_img_path)
        for ori_img_path in ori_img_lst:
            if det_res_img_name in ori_img_path:
                shutil.copy(ori_img_path, det_res_img_path)
                break

def batch_resize_imgs(path, size=(48, 48)):
    img_lst = []
    get_recursive_file_list(path, img_lst, ".jpg")
    for img_path in img_lst:
        img = cv2.imread(img_path)
        img_dst = cv2.resize(img, size, interpolation=random.choice([0,1,2,3,4]))
        cv2.imwrite(img_path, img_dst)


if __name__ == "__main__":
    batch_resize_imgs("/data5/tmp/wzheng/traffic_sign_cls/classification/patches48", size=(48, 48)); exit(0)
    copy_original_imgs("/data/wzheng/data/lane_detect/images/lane_videos/883_video_20180324/883_labeled_pos_2018_03_24/",
                       "/data/wzheng/data/lane_detect/images/lane_videos/883_video_20170802/rec_20170802_081516"); exit(0)
    make_lst("/data/wzheng/data/lane_detect/images/lane_videos/883_video_20170324/rec_20180324_144801", ".jpg", \
             "/data/wzheng/data/lane_detect/images/lane_videos/883_video_20170324/rec_20180324_144801.txt"); exit(0)

    copy_file_from_lst("/data/wczhang/car_type_proj/togo/gen_six_data_margin/val_list.txt",
                       "/data/wczhang/car_type_proj/togo/gen_six_data_margin/", "/data/wzheng/data/car_cls_val")
    copy_file_from_lst("/data/wczhang/car_type_proj/togo/gen_six_data_margin/train_list.txt", "/data/wczhang/car_type_proj/togo/gen_six_data_margin/", "/data/wzheng/data/car_cls_train"); exit(0)


    #rename_files_convert_jpg("/data/wzheng/test_rename")
    #make_bg_img_label_file("/data2/public_datasets/detection_from_nanjing/person_background/", 0, "/data2/public_datasets/detection_from_nanjing/person_background/bg_label.txt")
    make_caffe_lst_file(["/data/wzheng/data/side_person/patch/patch_24/negative_stage_1",
                         "/data/wzheng/data/side_person/patch/patch_24/positive_stage_1",
                         "/data/wzheng/data/side_person/patch/patch_24/negative_stage_1_bootstrapped"],
                        "/data/wzheng/data/side_person/patch/patch_24/",
                        0.8,
                        "/data/wzheng/data/side_person/patch/patch_24/stage_1_caffe_train_lst.txt",
                        "/data/wzheng/data/side_person/patch/patch_24/stage_1_caffe_val_lst.txt")


