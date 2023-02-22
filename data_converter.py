from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
from open3d import *
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import points_in_box
import os.path as osp
from functools import partial
from utils.points_process import *
from sklearn.neighbors import KDTree
import open3d as o3d
import argparse
INTER_STATIC_POINTS = {}
INTER_STATIC_POSE = {}
INTER_STATIC_LABEL = {}

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./project/data/nuscenes/',
        help='specify the root path of dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./project/data/nuscenes//occupancy2/',
        required=False,
        help='specify sweeps of lidar per example')
    parser.add_argument(
        '--num_sweeps',
        type=int,
        default=10,
        required=False,
        help='specify sweeps of lidar per example')
    args = parser.parse_args()
    return args

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def align_dynamic_thing(box, prev_instance_token, nusc, prev_points, ego_frame_info):
        if prev_instance_token not in ego_frame_info['instance_tokens']:
            box_mask = points_in_box(box,
                                    prev_points[:3, :])
            return np.zeros((prev_points.shape[0], 0)), np.zeros((0, )), box_mask
        
        box_mask = points_in_box(box,
                                    prev_points[:3, :])
        box_points = prev_points[:, box_mask].copy()
        prev_bbox_center = box.center
        prev_rotate_matrix = box.rotation_matrix

        box_points = rotate(box_points, np.linalg.inv(prev_rotate_matrix), center=prev_bbox_center)
        target = ego_frame_info['instance_tokens'].index(prev_instance_token)
        ego_boxes_center = ego_frame_info['boxes'][target].center
        box_points = translate(box_points, ego_boxes_center-prev_bbox_center)
        box_points = rotate(box_points, ego_frame_info['boxes'][target].rotation_matrix, center=ego_boxes_center)
        box_points_mask = filter_points_in_ego(box_points, ego_frame_info, prev_instance_token)
        box_points = box_points[:, box_points_mask]
        box_label = np.full_like(box_points[0], nusc.lidarseg_name2idx_mapping[box.name]).copy()
        return box_points, box_label, box_mask


def get_frame_info(frame, nusc: NuScenes, gt_from='lidarseg'):
    '''
    get frame info
    return: frame_info (Dict):

    '''
    sd_rec = nusc.get('sample_data', frame['data']['LIDAR_TOP'])
    lidar_path, boxes, _ = nusc.get_sample_data(frame['data']['LIDAR_TOP'])
    # lidarseg_labels_filename = os.path.join(nusc.dataroot,
    #                                             nusc.get(gt_from, sd_rec)['filename'])
    lidarseg_labels_filename = osp.join(nusc.dataroot,
                                    nusc.get(gt_from, frame['data']['LIDAR_TOP'])['filename'])

    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

    pc = LidarPointCloud.from_file(nusc.dataroot+sd_rec['filename']) 

    # pc = LidarPointCloud.from_file(nusc.dataroot+sd_rec['filename']) 
    cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    velocities = np.array(
                [nusc.box_velocity(token)[:2] for token in frame['anns']])
    velocities = np.concatenate((velocities, np.zeros_like(velocities[:, 0:1])), axis=-1)
    velocities = velocities.transpose(1, 0)
    instance_tokens = [nusc.get('sample_annotation', token)['instance_token'] for token in frame['anns']]
    frame_info = {
        'pc': pc,
        'token': frame['token'],
        'lidar_token': frame['data']['LIDAR_TOP'],
        'cs_record': cs_record,
        'pose_record': pose_record,
        'velocities': velocities,
        'lidarseg': points_label,
        'boxes': boxes,
        'anno_token': frame['anns'],
        'instance_tokens': instance_tokens,
        'timestamp': frame['timestamp'],
    }
    return frame_info


def get_intermediate_frame_info(nusc: NuScenes, prev_frame_info, lidar_rec, flag):
    intermediate_frame_info = dict()
    pc = LidarPointCloud.from_file(nusc.dataroot+lidar_rec['filename']) 
    intermediate_frame_info['pc'] = pc
    intermediate_frame_info['pc'].points = remove_close(intermediate_frame_info['pc'].points)
    intermediate_frame_info['lidar_token'] = lidar_rec['token']
    intermediate_frame_info['cs_record'] = nusc.get('calibrated_sensor',
                             lidar_rec['calibrated_sensor_token'])
    sample_token = lidar_rec['sample_token']
    frame = nusc.get('sample', sample_token)
    instance_tokens = [nusc.get('sample_annotation', token)['instance_token'] for token in frame['anns']]
    intermediate_frame_info['pose_record'] = nusc.get('ego_pose', lidar_rec['ego_pose_token'])
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_rec['token'])
    intermediate_frame_info['boxes'] = boxes
    intermediate_frame_info['instance_tokens'] = instance_tokens
    assert len(boxes) == len(instance_tokens) , print('erro')
    return intermediate_frame_info

def intermediate_keyframe_align(nusc: NuScenes, prev_frame_info, ego_frame_info, cur_sample_points, cur_sample_labels):
    ''' align prev_frame points to ego_frame
    return: points (np.array) aligned points of prev_frame
            pc_segs (np.array) label of aligned points of prev_frame
    '''
    prev_frame_info['pc'].points = remove_close(prev_frame_info['pc'].points, (1, 2))
    pcs, labels, masks = multi_apply(align_dynamic_thing, prev_frame_info['boxes'], prev_frame_info['instance_tokens'], nusc=nusc, prev_points=prev_frame_info['pc'].points, ego_frame_info=ego_frame_info)

    # for box, instance_token in zip(prev_frame_info['boxes'], prev_frame_info['instance_tokens']):
    #     align_dynamic_thing(box, instance_token, nusc=nusc, prev_points=prev_frame_info['pc'].points, ego_frame_info=ego_frame_info)

    masks = np.stack(masks, axis=-1)
    masks = masks.sum(axis=-1)
    masks = ~(masks>0)
    prev_frame_info['pc'].points = prev_frame_info['pc'].points[:, masks]

    
    if  prev_frame_info['lidar_token'] in INTER_STATIC_POINTS:
        static_points = INTER_STATIC_POINTS[prev_frame_info['lidar_token']].copy()
        static_points = prev2ego(static_points, INTER_STATIC_POSE[prev_frame_info['lidar_token']], ego_frame_info)
        static_points_label = INTER_STATIC_LABEL[prev_frame_info['lidar_token']].copy()
        assert static_points_label.shape[0] == static_points.shape[1], f"{static_points_label.shape, static_points.shape}"
    else:
        static_points = prev2ego(prev_frame_info['pc'].points, prev_frame_info, ego_frame_info)
        static_points_label = np.full_like(static_points[0], -1)
        static_points, static_points_label = search_label(cur_sample_points, cur_sample_labels, static_points, static_points_label)
        INTER_STATIC_POINTS[prev_frame_info['lidar_token']] = static_points.copy()
        INTER_STATIC_LABEL[prev_frame_info['lidar_token']] = static_points_label.copy()
        INTER_STATIC_POSE[prev_frame_info['lidar_token']] = {'cs_record': ego_frame_info['cs_record'],
                                                            'pose_record': ego_frame_info['pose_record'],
                                                            }
    pcs.append(static_points)
    labels.append(static_points_label)
    return np.concatenate(pcs, axis=-1), np.concatenate(labels)

def nonkeykeyframe_align(nusc: NuScenes, prev_frame_info, ego_frame_info, flag='prev', cur_sample_points=None, cur_sample_labels=None):
    ''' align non keyframe points to ego_frame
    return: points (np.array) aligned points of prev_frame
            pc_segs (np.array) seg of aligned points of prev_frame
    '''
    pcs = []
    labels = []
    start_frame = nusc.get('sample', prev_frame_info['token'])
    end_frame = nusc.get('sample', start_frame[flag])
    # next_frame_info = get_frame_info(end_frame, nusc)
    start_sd_record = nusc.get('sample_data', start_frame['data']['LIDAR_TOP'])
    start_sd_record = nusc.get('sample_data', start_sd_record[flag])
    # end_sd_record = nusc.get('sample_data', end_frame['data']['LIDAR_TOP'])
    # get intermediate frame info
    while start_sd_record['token'] != end_frame['data']['LIDAR_TOP']:
        intermediate_frame_info = get_intermediate_frame_info(nusc, prev_frame_info, start_sd_record, flag)
        pc, label = intermediate_keyframe_align(nusc, intermediate_frame_info, ego_frame_info, cur_sample_points, cur_sample_labels)
        start_sd_record = nusc.get('sample_data', start_sd_record[flag])
        pcs.append(pc)
        labels.append(label)
    return np.concatenate(pcs, axis=-1), np.concatenate(labels)


def prev2ego(points, prev_frame_info, income_frame_info, velocity=None, time_gap=0.0):
    ''' translation prev points to ego frame
    '''
    # prev_sd_rec = nusc.get('sample_data', prev_frame_info['data']['LIDAR_TOP'])

    prev_cs_record = prev_frame_info['cs_record']
    prev_pose_record = prev_frame_info['pose_record']

    points = transform(points, Quaternion(prev_cs_record['rotation']).rotation_matrix, np.array(prev_cs_record['translation']))
    points = transform(points, Quaternion(prev_pose_record['rotation']).rotation_matrix, np.array(prev_pose_record['translation']))

    if velocity is not None:
        points[:3, :] = points[:3, :] + velocity*time_gap

    ego_cs_record = income_frame_info['cs_record']
    ego_pose_record = income_frame_info['pose_record']
    points = transform(points, Quaternion(ego_pose_record['rotation']).rotation_matrix, np.array(ego_pose_record['translation']), inverse=True)
    points = transform(points, Quaternion(ego_cs_record['rotation']).rotation_matrix, np.array(ego_cs_record['translation']), inverse=True)
    return points.copy()


def filter_points_in_ego(points, frame_info, instance_token):
    '''
    filter points in this frame box
    '''
    index = frame_info['instance_tokens'].index(instance_token)
    box = frame_info['boxes'][index]
    # print(f"ego box pos {box.center}")
    box_mask = points_in_box(box, points[:3, :])
    return box_mask

def keyframe_align(prev_frame_info, ego_frame_info):
    ''' align prev_frame points to ego_frame
    return: points (np.array) aligned points of prev_frame
            pc_segs (np.array) seg of aligned points of prev_frame
    '''
    pcs = []
    pc_segs = []
    lidarseg_prev = prev_frame_info['lidarseg']
    ego_vehicle_mask = (lidarseg_prev == 31) | (lidarseg_prev == 0)
    lidarseg_prev = lidarseg_prev[~ego_vehicle_mask]
    prev_frame_info['pc'].points = prev_frame_info['pc'].points[:, ~ego_vehicle_mask]

    # translation prev static points to ego
    static_mask = (lidarseg_prev >= 24) & (lidarseg_prev <= 30)

    static_points = prev_frame_info['pc'].points[:, static_mask]
    static_seg = lidarseg_prev[static_mask]
    static_points = prev2ego(static_points, prev_frame_info, ego_frame_info)
    pcs.append(static_points.copy())
    pc_segs.append(static_seg.copy())
    prev_frame_info['pc'].points = prev_frame_info['pc'].points[:, ~static_mask].copy()
    lidarseg_prev = lidarseg_prev[~static_mask]
    # translation prev moving points to ego
    for index_anno in range(len(prev_frame_info['boxes'])):
        if prev_frame_info['instance_tokens'][index_anno] not in ego_frame_info['instance_tokens']:
            continue
        box_mask = points_in_box(prev_frame_info['boxes'][index_anno],
                                    prev_frame_info['pc'].points[:3, :])
        box_points = prev_frame_info['pc'].points[:, box_mask].copy()
        boxseg_prev = lidarseg_prev[box_mask].copy()
        prev_bbox_center = prev_frame_info['boxes'][index_anno].center

        prev_rotate_matrix = prev_frame_info['boxes'][index_anno].rotation_matrix
        box_points = rotate(box_points, np.linalg.inv(prev_rotate_matrix), center=prev_bbox_center)

        target = ego_frame_info['instance_tokens'].index(prev_frame_info['instance_tokens'][index_anno])
        ego_boxes_center = ego_frame_info['boxes'][target].center
        box_points = translate(box_points, ego_boxes_center-prev_bbox_center)
        box_points = rotate(box_points, ego_frame_info['boxes'][target].rotation_matrix, center=ego_boxes_center)

        box_points_mask = filter_points_in_ego(box_points, ego_frame_info, prev_frame_info['instance_tokens'][index_anno])
        box_points = box_points[:, box_points_mask]
        boxseg_prev = boxseg_prev[box_points_mask]

        pcs.append(box_points)
        pc_segs.append(boxseg_prev)
    return np.concatenate(pcs, axis=-1), np.concatenate(pc_segs, axis=-1)


def search_label(points, lidar_seg, intermediate_pcs, intermediate_labels, max_dist=0.5):
    unlabel_mask = intermediate_labels == -1
    thing_mask = (lidar_seg >= 24) & (lidar_seg <=30)
    thing_label = lidar_seg[thing_mask]
    thing_points = points[:, thing_mask]
    unlabeled_points = intermediate_pcs[:, unlabel_mask]
    tree = KDTree(thing_points.transpose(1, 0)[:, :3])
    unlabeled_points = unlabeled_points.transpose(1, 0)
    dists, inds = tree.query(unlabeled_points[:, :3], k=1)
    inds = np.reshape(inds, (-1,))
    dists = np.reshape(dists, (-1,))
    dists = dists<max_dist
    intermediate_labels[unlabel_mask] = np.take_along_axis(thing_label, inds, axis=-1)
    return intermediate_pcs[:, dists], intermediate_labels[dists]


def generate_occupancy_data(nusc: NuScenes, cur_sample, num_sweeps, save_path='./occupacy/', gt_from: str = 'lidarseg'):
    pcs =[] # for keyframe points
    pc_segs = []

    intermediate_pcs = [] # # for non keyfrme points
    intermediate_labels = []
    lidar_data = nusc.get('sample_data',
                            cur_sample['data']['LIDAR_TOP'])
    pc = LidarPointCloud.from_file(nusc.dataroot+lidar_data['filename'])
    filename = os.path.split(lidar_data['filename'])[-1]
    lidar_sd_token = cur_sample['data']['LIDAR_TOP']
    
    lidarseg_labels_filename = os.path.join(nusc.dataroot,
                                                nusc.get(gt_from, lidar_sd_token)['filename'])
    lidar_seg = load_bin_file(lidarseg_labels_filename, type=gt_from)

    # align keyframes
    count_prev_frame = 0
    prev_frame = cur_sample.copy()

    while num_sweeps > 0:
        if prev_frame['prev'] == '':
            break
        prev_frame = nusc.get('sample', prev_frame['prev'])
        count_prev_frame += 1
        if count_prev_frame == num_sweeps:
            break
    cur_sample_info = get_frame_info(cur_sample, nusc=nusc)
    # convert prev keyframe to ego frame
    if count_prev_frame > 0:
        prev_info = get_frame_info(prev_frame, nusc)
    pc_points = None
    pc_seg = None
    while count_prev_frame > 0:
        income_info = get_frame_info(frame =prev_frame, nusc=nusc)
        prev_frame = nusc.get('sample', prev_frame['next'])
        prev_info = income_info
        pc_points, pc_seg = keyframe_align(prev_info, cur_sample_info)
        pcs.append(pc_points)
        pc_segs.append(pc_seg)
        count_prev_frame -= 1

    # convert next frame to ego frame
    next_frame = cur_sample.copy()
    pc_points = None
    pc_seg = None
    count_next_frame = 0
    while num_sweeps > 0:
        if next_frame['next'] == '':
            break
        next_frame = nusc.get('sample', next_frame['next'])
        count_next_frame += 1
        if count_next_frame == num_sweeps:
            break

    if count_next_frame > 0:
        prev_info = get_frame_info(next_frame, nusc=nusc)

    while count_next_frame > 0:
        
        income_info = get_frame_info(frame=next_frame, nusc=nusc)
        prev_info = income_info
        next_frame =  nusc.get('sample', next_frame['prev'])
        pc_points, pc_seg = keyframe_align(prev_info, cur_sample_info)
        pcs.append(pc_points)
        pc_segs.append(pc_seg)
        count_next_frame -= 1
    pcs = np.concatenate(pcs, axis=-1)
    pc_segs = np.concatenate(pc_segs)



    pc.points = np.concatenate((pc.points, pcs), axis=-1)
    lidar_seg = np.concatenate((lidar_seg, pc_segs))


    range_mask = (pc.points[0,:]<= 60) &  (pc.points[0,:]>=-60)\
     &(pc.points[1,:]<= 60) &  (pc.points[1,:]>=-60)\
      &(pc.points[2,:]<= 10) &  (pc.points[2,:]>=-10)
    pc.points = pc.points[:, range_mask]
    lidar_seg = lidar_seg[range_mask]


    # align nonkeyframe
    count_prev_frame = 0
    prev_frame = cur_sample.copy()

    while num_sweeps > 0:
        if prev_frame['prev'] == '':
            break
        prev_frame = nusc.get('sample', prev_frame['prev'])
        count_prev_frame += 1
        if count_prev_frame == num_sweeps:
            break
    cur_sample_info = get_frame_info(cur_sample, nusc=nusc)
    # convert prev frame to ego frame
    if count_prev_frame > 0:
        prev_info = get_frame_info(prev_frame, nusc)
    while count_prev_frame > 0:
        income_info = get_frame_info(frame =prev_frame, nusc=nusc)
        prev_frame = nusc.get('sample', prev_frame['next'])
        prev_info = income_info
        intermediate_pc, intermediate_label = nonkeykeyframe_align(nusc, prev_info, cur_sample_info, 'next', pc.points, lidar_seg)
        intermediate_pcs.append(intermediate_pc)
        intermediate_labels.append(intermediate_label)
        count_prev_frame -= 1

    next_frame = cur_sample.copy()
    count_next_frame = 0
    while num_sweeps > 0:
        if next_frame['next'] == '':
            break
        next_frame = nusc.get('sample', next_frame['next'])
        count_next_frame += 1
        if count_next_frame == num_sweeps:
            break

    if count_next_frame > 0:
        prev_info = get_frame_info(next_frame, nusc=nusc)

    while count_next_frame > 0:
        
        income_info = get_frame_info(frame =next_frame, nusc=nusc)
        prev_info = income_info
        next_frame =  nusc.get('sample', next_frame['prev'])
        intermediate_pc, intermediate_label = nonkeykeyframe_align(nusc, prev_info, cur_sample_info, 'prev', pc.points, lidar_seg)
        intermediate_pcs.append(intermediate_pc)
        intermediate_labels.append(intermediate_label)
        count_next_frame -= 1
    intermediate_pcs = np.concatenate(intermediate_pcs, axis=-1)
    intermediate_labels = np.concatenate(intermediate_labels)
    intermediate_labels = np.reshape(intermediate_labels, (1, -1))
    intermediate_pcs = np.concatenate((intermediate_pcs, intermediate_labels), axis=0)
    lidar_seg = np.reshape(lidar_seg, (1, -1))
    pc.points = np.concatenate((pc.points, lidar_seg), axis=0)
    pc.points = np.concatenate((pc.points, intermediate_pcs), axis=1)

    # removed too dense point
    raw_point = pc.points.transpose(1,0)[:,:3]
    fake_colors = pc.points.transpose(1,0)[:,3:]/255 
    assert pc.points.transpose(1,0)[:,3:].max()<=255
    n, _ = fake_colors.shape
    fake_colors = np.concatenate((fake_colors, np.zeros((n,1))), axis=1)
    pcd=o3d.open3d.geometry.PointCloud()
    
    pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    pcd.colors = o3d.open3d.utility.Vector3dVector(fake_colors)
    pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, 0.2)
    new_points = np.asarray(pcd_new.points)
    fake_colors = np.asarray(pcd_new.colors)[:,:2]*255
    new_points = np.concatenate((new_points, fake_colors), axis=1)

    range_mask = (new_points[:,0]<= 60) &  (new_points[:,0]>=-60)\
     &(new_points[:,1]<= 60) &  (new_points[:,1]>=-60)\
      &(new_points[:,2]<= 10) &  (new_points[:,2]>=-10)
    new_points = new_points[range_mask]
    new_points = new_points.astype(np.float16)
    new_points.tofile(save_path +filename)
    return pc.points, lidar_seg

def convert2occupy(dataroot,
                        save_path, num_sweeps=10,):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cnt = 0
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    for scene in nusc.scene:
        INTER_STATIC_POINTS.clear()
        INTER_STATIC_LABEL.clear()
        INTER_STATIC_POSE.clear()
        sample_token = scene['first_sample_token']
        cur_sample = nusc.get('sample', sample_token)
        while True:
            cnt += 1
            print(cnt)
            generate_occupancy_data(nusc, cur_sample, num_sweeps, save_path=save_path)
            if cur_sample['next'] == '':
                break
            cur_sample = nusc.get('sample', cur_sample['next'])

if __name__ == "__main__":
    args = parse_args()
    convert2occupy(args.dataroot, args.save_path, args.num_sweeps)

